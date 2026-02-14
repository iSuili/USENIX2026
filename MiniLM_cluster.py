import os
import json
import re
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
from tqdm import tqdm

DATA_PATH = "MaliciousInstruct.txt"
TEXT_COL = "goal"

MODEL_DIR = "modelscope_models/AI-ModelScope/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 2
CLUSTER_SELECTION_METHOD = "leaf"   

USE_UMAP = True
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.0

RANDOM_SEED = 42
HOLDOUT_RATIO = 0.2
MIN_TEST_CLUSTERS = 3

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "maliciousinstruct_with_minilm_clusters.csv")
CLUSTER_SUMMARY_CSV = os.path.join(RESULT_DIR, "maliciousinstruct_cluster_summary.csv")
CLUSTER_STATS_JSON = os.path.join(RESULT_DIR, "maliciousinstruct_cluster_stats.json")
OOD_SPLITS_JSON = os.path.join(RESULT_DIR, "maliciousinstruct_ood_domain_splits.json")

TEMPLATE_PATTERNS = [
    r"^\s*how\s+to\s+",
    r"^\s*ways\s+to\s+",
    r"^\s*tips\s+for\s+(using\s+)?",
    r"^\s*ideas\s+for\s+(using\s+)?",
    r"^\s*methods\s+to\s+",
    r"^\s*techniques\s+to\s+",
]

def strip_template(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[?]+$", "", s)
    for p in TEMPLATE_PATTERNS:
        s = re.sub(p, "", s)
    s = re.sub(r"\b(make|get)\s+someone\b", "someone", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_text_lines(path: str, text_col: str = "goal") -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"DATA_PATH not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f]
    lines = [x for x in lines if x]
    if lines and lines[0].lower() == text_col.lower():
        lines = lines[1:]
    return pd.DataFrame({text_col: lines})

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def encode_prompts(tokenizer, model, prompts, batch_size=32, max_len=128, device="cpu"):
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding"):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            sent_emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            embs.append(sent_emb.cpu().numpy())
    return np.vstack(embs)

def cluster_intra_sim(emb, labels, cid, max_items=200, seed=42):
    idx = np.where(labels == cid)[0]
    if len(idx) <= 1:
        return None
    if len(idx) > max_items:
        idx = np.random.RandomState(seed).choice(idx, size=max_items, replace=False)
    sims = cosine_similarity(emb[idx], emb[idx])
    tri = sims[np.triu_indices_from(sims, k=1)]
    return float(tri.mean()) if tri.size > 0 else None

EXTRA_STOP = {
    "how", "ways", "way", "tips", "tip", "ideas", "idea", "methods", "method", "techniques", "technique",
    "guide", "tutorial", "instruction", "instructions", "step", "steps", "step-by-step",
    "create", "write", "generate", "provide", "develop", "design", "explain", "make", "teach", "outline",
    "someone", "people", "person", "your", "you", "user", "users", "use", "using"
}

def top_keywords_per_cluster(df, text_col, label_col="cluster_id", topk=8, min_df=2):
    kw = {}
    for cid in sorted(df[label_col].unique()):
        if cid == -1:
            continue
        texts = df.loc[df[label_col] == cid, text_col].astype(str).tolist()
        if len(texts) < 2:
            kw[cid] = []
            continue

        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df
        )
        X = vec.fit_transform(texts)
        vocab = np.array(vec.get_feature_names_out())

        keep_mask = []
        for t in vocab:
            bad = False
            for s in EXTRA_STOP:
                if s in t:
                    bad = True
                    break
            keep_mask.append(not bad)
        keep_mask = np.array(keep_mask, dtype=bool)

        if keep_mask.sum() == 0:
            kw[cid] = []
            continue

        X = X[:, keep_mask]
        vocab = vocab[keep_mask]
        if vocab.size == 0:
            kw[cid] = []
            continue

        scores = np.asarray(X.mean(axis=0)).ravel()
        top_idx = scores.argsort()[::-1][:topk]
        kw[cid] = vocab[top_idx].tolist()
    return kw

def main():
    df = load_text_lines(DATA_PATH, TEXT_COL)
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(strip_template)
    prompts = df[TEXT_COL].tolist()
    print(f"Loaded {len(prompts)} lines from {DATA_PATH}")

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
    model.to(DEVICE)
    model.eval()

    embeddings = encode_prompts(
        tokenizer=tokenizer,
        model=model,
        prompts=prompts,
        batch_size=32,
        max_len=MAX_LEN,
        device=DEVICE
    )
    print("Embedding shape:", embeddings.shape)

    emb_for_cluster = embeddings
    umap_ok = False
    if USE_UMAP:
        try:
            import umap
            reducer = umap.UMAP(
                n_components=UMAP_N_COMPONENTS,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                metric="cosine",
                random_state=RANDOM_SEED
            )
            emb_for_cluster = reducer.fit_transform(embeddings)
            umap_ok = True
            print("UMAP reduced shape:", emb_for_cluster.shape)
        except Exception as e:
            print(f"[WARN] UMAP failed ({e}). Continue without UMAP.")
            emb_for_cluster = embeddings

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=CLUSTER_SELECTION_METHOD
    )
    cluster_labels = clusterer.fit_predict(emb_for_cluster)
    df["cluster_id"] = cluster_labels

    valid = df[df["cluster_id"] != -1]
    cluster_ids = sorted(valid["cluster_id"].unique().tolist())
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(cluster_ids)

    n_hold = int(len(cluster_ids) * HOLDOUT_RATIO) if len(cluster_ids) > 0 else 0
    n_hold = max(1, n_hold) if len(cluster_ids) > 0 else 0
    if len(cluster_ids) >= MIN_TEST_CLUSTERS:
        n_hold = max(MIN_TEST_CLUSTERS, n_hold)
    n_hold = min(n_hold, max(1, len(cluster_ids) - 1)) if len(cluster_ids) > 1 else n_hold

    test_cids = set(cluster_ids[:n_hold])
    train_cids = set(cluster_ids[n_hold:])

    df["ood_domain_split"] = "outlier"
    df.loc[df["cluster_id"].isin(train_cids), "ood_domain_split"] = "train_cluster"
    df.loc[df["cluster_id"].isin(test_cids), "ood_domain_split"] = "test_cluster"

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] Saved clustered prompts to {OUTPUT_CSV}")

    keywords = top_keywords_per_cluster(df[df["cluster_id"] != -1], TEXT_COL, "cluster_id", topk=8, min_df=2)

    summary_rows = []
    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            continue
        size = int((df["cluster_id"] == cid).sum())
        mean_sim = cluster_intra_sim(embeddings, cluster_labels, cid, max_items=200, seed=RANDOM_SEED)
        examples = df.loc[df["cluster_id"] == cid, TEXT_COL].head(3).astype(str).tolist()
        summary_rows.append({
            "cluster_id": int(cid),
            "size": size,
            "mean_cos_sim": mean_sim,
            "top_keywords": "; ".join(keywords.get(cid, [])),
            "example_1": examples[0] if len(examples) > 0 else "",
            "example_2": examples[1] if len(examples) > 1 else "",
            "example_3": examples[2] if len(examples) > 2 else "",
            "split": "test_cluster" if cid in test_cids else "train_cluster"
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["size", "mean_cos_sim"], ascending=[False, False])
    summary_df.to_csv(CLUSTER_SUMMARY_CSV, index=False, encoding="utf-8")
    print(f"[OK] Saved {CLUSTER_SUMMARY_CSV}")

    n_total = len(df)
    n_out = int((df["cluster_id"] == -1).sum())
    cluster_sizes = valid["cluster_id"].value_counts().values if len(valid) else np.array([])

    cohs = []
    for cid in sorted(valid["cluster_id"].unique()):
        ms = cluster_intra_sim(embeddings, cluster_labels, cid, max_items=200, seed=RANDOM_SEED)
        if ms is not None:
            cohs.append(ms)

    stats = {
        "N_total": int(n_total),
        "num_clusters": int(valid["cluster_id"].nunique()) if len(valid) else 0,
        "num_outliers": int(n_out),
        "outlier_ratio": float(n_out / max(1, n_total)),
        "cluster_size_min": int(cluster_sizes.min()) if cluster_sizes.size else 0,
        "cluster_size_median": float(np.median(cluster_sizes)) if cluster_sizes.size else 0.0,
        "cluster_size_mean": float(cluster_sizes.mean()) if cluster_sizes.size else 0.0,
        "cluster_size_max": int(cluster_sizes.max()) if cluster_sizes.size else 0,
        "mean_intra_cos_sim_avg": float(np.mean(cohs)) if len(cohs) else None,
        "mean_intra_cos_sim_std": float(np.std(cohs)) if len(cohs) else None,
        "ood_domain_split": {
            "random_seed": RANDOM_SEED,
            "holdout_ratio": HOLDOUT_RATIO,
            "min_test_clusters": MIN_TEST_CLUSTERS,
            "num_train_clusters": int(len(train_cids)),
            "num_test_clusters": int(len(test_cids)),
        },
        "params": {
            "DATA_PATH": DATA_PATH,
            "TEXT_COL": TEXT_COL,
            "MODEL_DIR": MODEL_DIR,
            "MAX_LEN": MAX_LEN,
            "DEVICE": DEVICE,
            "USE_UMAP": USE_UMAP,
            "UMAP_OK": umap_ok,
            "UMAP_N_COMPONENTS": UMAP_N_COMPONENTS,
            "UMAP_N_NEIGHBORS": UMAP_N_NEIGHBORS,
            "UMAP_MIN_DIST": UMAP_MIN_DIST,
            "MIN_CLUSTER_SIZE": MIN_CLUSTER_SIZE,
            "MIN_SAMPLES": MIN_SAMPLES,
            "HDBSCAN_metric": "euclidean",
            "cluster_selection_method": CLUSTER_SELECTION_METHOD,
        }
    }

    with open(CLUSTER_STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {CLUSTER_STATS_JSON}")
    print("[STATS]", json.dumps(stats, ensure_ascii=False, indent=2))

    splits = {
        "random_seed": RANDOM_SEED,
        "holdout_ratio": HOLDOUT_RATIO,
        "min_test_clusters": MIN_TEST_CLUSTERS,
        "train_clusters": sorted(list(train_cids)),
        "test_clusters": sorted(list(test_cids)),
    }
    with open(OOD_SPLITS_JSON, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {OOD_SPLITS_JSON}")

    print("\n==== Quick Overview (Top-10 clusters by size) ====")
    top10 = summary_df.head(10)
    for _, row in top10.iterrows():
        mean_sim = row["mean_cos_sim"]
        mean_sim_str = f"{float(mean_sim):.4f}" if pd.notna(mean_sim) else "None"
        print(
            f"\nCluster {int(row['cluster_id'])} | "
            f"size={int(row['size'])} | "
            f"mean_cos_sim={mean_sim_str} | "
            f"split={row['split']}"
        )
        print(f"  keywords: {row['top_keywords']}")
        print(f"  ex1: {row['example_1']}")
        print(f"  ex2: {row['example_2']}")
        print(f"  ex3: {row['example_3']}")

    print("\n==== Outlier examples ====")
    out_examples = df[df["cluster_id"] == -1][TEXT_COL].head(10).tolist()
    for e in out_examples:
        print("  -", e)

if __name__ == "__main__":
    main()