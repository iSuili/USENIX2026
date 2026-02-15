#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ===================== Feature config (front3 + mid5concat mean pooled) =====================
SEGMENT_IDX = 0          # front segment
GROUP_IDX = 1            # mid group
SUB_RANGE = slice(0, 5)  # 5 sublayers
TOKENS_USED = 3          # front3
# expected arr shape: (2, T, 3, 5, H)

# ===================== Utils =====================
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_hs_prompt_npy(root: str):
    """Recursively find hs_prompt*.npy, excluding hs_promptLast_prompt*.npy"""
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".npy"):
                continue
            if not fn.startswith("hs_prompt"):
                continue
            if fn.startswith("hs_promptLast_prompt"):
                continue
            out.append(os.path.join(r, fn))
    out.sort(key=natural_key)
    return out

def load_features_front3_mid5_concat(path: str):
    """
    Take segment=0, token t=0..2, group=1, sub=0..4 -> (5,H), concat -> (5H,)
    mean pooling over 3 tokens -> (5H,)
    """
    try:
        arr = np.load(path)
    except Exception as e:
        print(f"[WARNING] Failed to load {path}: {e}. Skip.")
        return None

    if arr.ndim != 5 or arr.shape[0] < 2:
        print(f"[WARNING] Unexpected shape {arr.shape} in {path}. Expect (2,T,3,5,H). Skip.")
        return None

    T = arr.shape[1]
    L = arr.shape[2]
    S = arr.shape[3]

    if T < TOKENS_USED:
        print(f"[WARNING] Not enough tokens T={T} in {path}. Need >= {TOKENS_USED}. Skip.")
        return None
    if L <= 1 or S != 5:
        print(f"[WARNING] Unexpected layer structure L={L}, S={S} in {path}. Skip.")
        return None

    token_feats = []
    for t in range(TOKENS_USED):
        feat5 = arr[SEGMENT_IDX, t, GROUP_IDX, SUB_RANGE, :]  # (5,H)
        token_feats.append(feat5.reshape(-1))                  # (5H,)
    pooled = np.mean(np.stack(token_feats, axis=0), axis=0)   # (5H,)
    return pooled.astype(np.float32)

def collect_features_from_files(files, label: int):
    X_list, y_list = [], []
    skipped = 0
    for p in files:
        feat = load_features_front3_mid5_concat(p)
        if feat is None:
            skipped += 1
            continue
        X_list.append(feat)
        y_list.append(label)
    return X_list, y_list, skipped

def collect_features_from_dirs(dirs, label: int):
    all_files = []
    raw = 0
    for d in dirs:
        fs = list_hs_prompt_npy(d)
        raw += len(fs)
        all_files.extend(fs)
    X_list, y_list, skipped = collect_features_from_files(all_files, label=label)
    return X_list, y_list, raw, skipped

# ===================== Model =====================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ===================== Threshold helpers (margin-based) =====================
@torch.no_grad()
def logits_to_margin(logits: np.ndarray) -> np.ndarray:
    """
    logits: (N,2) with class0=benign, class1=malicious
    margin = z1 - z0. predict malicious if margin >= tau
    """
    return logits[:, 1] - logits[:, 0]

def choose_tau_by_target_fpr(margins_benign: np.ndarray, target_fpr: float) -> float:
    """
    Choose tau such that FPR ~= target_fpr on benign margins:
      FPR = P(margin >= tau | benign)
    So tau is the (1-target_fpr) quantile.
    """
    margins = np.asarray(margins_benign, dtype=np.float64)
    if margins.size == 0:
        raise ValueError("Empty benign margins for threshold selection.")
    q = 1.0 - float(target_fpr)
    tau = float(np.quantile(margins, q, method="higher" if hasattr(np, "quantile") else "linear"))
    return tau

@torch.no_grad()
def eval_with_tau(model, X: np.ndarray, y: np.ndarray, tau: float, device: str, name: str):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits_t = model(X_t)
    logits = logits_t.detach().cpu().numpy()
    margins = logits_to_margin(logits)
    preds = (margins >= tau).astype(np.int64)

    # AUC uses soft score; use sigmoid(margin) (monotonic, OK)
    scores = 1.0 / (1.0 + np.exp(-margins))

    acc = float(np.mean(preds == y))
    cm = confusion_matrix(y, preds)
    if cm.shape == (2,2):
        TN, FP, FN, TP = cm.ravel()
        TPR = float(TP / (TP + FN + 1e-12))
        FPR = float(FP / (FP + TN + 1e-12))
    else:
        TPR = float("nan")
        FPR = float("nan")

    try:
        auc = float(roc_auc_score(y, scores))
    except Exception:
        auc = float("nan")

    rep = classification_report(y, preds, target_names=["benign", "malicious"], digits=4)
    return {
        "name": name,
        "tau": tau,
        "n": int(len(y)),
        "n_benign": int((y==0).sum()),
        "n_malicious": int((y==1).sum()),
        "acc": acc,
        "TPR": TPR,
        "FPR": FPR,
        "AUC": auc,
        "cm": cm,
        "report": rep,
    }

# ===================== Sampling plan for multi-attack training =====================
def balanced_multi_attack_sample(attack_to_files: dict, n_target: int, seed: int):
    """
    Sample from each attack uniformly to reach n_target total (without replacement).
    If some attack has too few, we:
      - take all from that attack
      - redistribute remaining quota to attacks with leftover
    """
    rng = random.Random(seed)
    attacks = sorted(list(attack_to_files.keys()))
    for a in attacks:
        rng.shuffle(attack_to_files[a])

    K = len(attacks)
    if K == 0:
        raise RuntimeError("No attack dirs found under train base.")

    # initial quota
    base = n_target // K
    rem = n_target % K
    quotas = {a: base for a in attacks}
    for a in attacks[:rem]:
        quotas[a] += 1

    chosen = []
    # first pass: fulfill quotas or exhaust
    leftovers = {}
    missing = 0
    for a in attacks:
        files = attack_to_files[a]
        q = quotas[a]
        take = min(q, len(files))
        chosen.extend(files[:take])
        if take < q:
            missing += (q - take)
        leftovers[a] = files[take:]  # remaining

    # second pass: fill missing from remaining pools
    if missing > 0:
        pool = []
        for a in attacks:
            pool.extend(leftovers[a])
        rng.shuffle(pool)
        fill = pool[:missing]
        chosen.extend(fill)

    # if still short (overall not enough), then we cap
    if len(chosen) < n_target:
        print(f"[WARN] Not enough total train attack samples: need {n_target}, got {len(chosen)}.")

    rng.shuffle(chosen)
    return chosen[:n_target]

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", type=str, default="/home/hyr/Code/Wwq/0201/result/gen_outputs_0201_hs/splits")
    ap.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--seed", type=int, default=42)

    # training benign sampling (optional)
    ap.add_argument("--train_benign_cap", type=int, default=-1,
                    help="If >0, downsample train/harmless to this many samples before balancing attacks. Default=-1 (use all).")

    # threshold selection
    ap.add_argument("--target_fpr", type=float, default=None,
                    help="If set, choose tau by target FPR on benign calibration set (margin-based).")
    ap.add_argument("--tau_from", type=str, default="test",
                    choices=["test", "train"],
                    help="Where to take benign calibration set for tau: test (recommended) or train.")
    ap.add_argument("--result_csv", type=str, default="result/qwen2_5_multiattack_balanced_front3_mid5concat_mlp_pairwise.csv")
    ap.add_argument("--include_test_harmful_pair", action="store_true",
                    help="Also evaluate harmless vs harmful(test) pair.")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_base = os.path.join(args.split_root, "train", args.model_name)
    test_base  = os.path.join(args.split_root, "test",  args.model_name)

    train_harmless = os.path.join(train_base, "harmless")
    test_harmless  = os.path.join(test_base,  "harmless")
    test_harmful   = os.path.join(test_base,  "harmful")

    # discover attack dirs
    train_attack_dirs = []
    for name in os.listdir(train_base):
        p = os.path.join(train_base, name)
        if os.path.isdir(p) and name.startswith("attack_"):
            train_attack_dirs.append(p)
    train_attack_dirs.sort()

    test_attack_dirs = []
    for name in os.listdir(test_base):
        p = os.path.join(test_base, name)
        if os.path.isdir(p) and name.startswith("attack_"):
            test_attack_dirs.append(p)
    test_attack_dirs.sort()

    print("\n==== Multi-attack balanced training (Qwen only) ====")
    print("[TRAIN harmless] ", train_harmless)
    print("[TRAIN attacks ] ")
    for d in train_attack_dirs:
        print("  -", d)
    print("[TEST harmless ] ", test_harmless)
    if os.path.isdir(test_harmful):
        print("[TEST harmful  ] ", test_harmful)
    print("[TEST attacks  ] ")
    for d in test_attack_dirs:
        print("  -", d)
    print("[FEAT] front3 + mid5 concat (mean pooled) -> MLP")
    print(f"[DEVICE] {device}  seed={args.seed}")

    if not os.path.isdir(train_harmless):
        raise FileNotFoundError(f"Missing: {train_harmless}")
    if not os.path.isdir(test_harmless):
        raise FileNotFoundError(f"Missing: {test_harmless}")
    if len(train_attack_dirs) == 0:
        raise FileNotFoundError(f"No train attack dirs under: {train_base}")
    if len(test_attack_dirs) == 0:
        raise FileNotFoundError(f"No test attack dirs under: {test_base}")

    # -------------------- Load train harmless files --------------------
    train_ben_files = list_hs_prompt_npy(train_harmless)
    if args.train_benign_cap and args.train_benign_cap > 0:
        rng = random.Random(args.seed)
        rng.shuffle(train_ben_files)
        train_ben_files = train_ben_files[:args.train_benign_cap]

    n_b = len(train_ben_files)
    if n_b == 0:
        raise RuntimeError("No train harmless .npy found.")
    print(f"\n[TRAIN benign] files={n_b} (cap={args.train_benign_cap})")

    # -------------------- Build attack->files dict and sample malicious = n_b --------------------
    attack_to_files = {}
    for d in train_attack_dirs:
        name = os.path.basename(d)
        files = list_hs_prompt_npy(d)
        attack_to_files[name] = files
        print(f"[TRAIN attack] {name} files={len(files)}")

    train_mal_files = balanced_multi_attack_sample(attack_to_files, n_target=n_b, seed=args.seed)
    print(f"[TRAIN malicious] sampled total={len(train_mal_files)} (target={n_b})")

    # -------------------- Extract features --------------------
    Xb, yb, sk_b = collect_features_from_files(train_ben_files, label=0)
    Xm, ym, sk_m = collect_features_from_files(train_mal_files, label=1)

    X_train_list = Xb + Xm
    y_train = np.array(yb + ym, dtype=np.int64)

    if len(X_train_list) == 0:
        raise RuntimeError("No training samples loaded after feature extraction.")

    X_train = np.stack(X_train_list, axis=0)
    print("\n==== Train stats ====")
    print(f"TRAIN benign  valid={int((y_train==0).sum())} skipped={sk_b}")
    print(f"TRAIN mal     valid={int((y_train==1).sum())} skipped={sk_m}")
    print(f"TRAIN total N={len(y_train)} dim={X_train.shape[1]}")

    # -------------------- Train MLP --------------------
    model = MLP(input_dim=X_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == y_train_t).float().mean().item()
            print(f"[Epoch {epoch+1:2d}/{args.epochs}] loss={loss.item():.6f} train_acc={acc:.4f}")

    # -------------------- Load test harmless once (for pairwise & calibration) --------------------
    Xte_b, _, raw_te_b, sk_te_b = collect_features_from_dirs([test_harmless], label=0)
    if len(Xte_b) == 0:
        raise RuntimeError("No test harmless samples loaded.")
    X_ben = np.stack(Xte_b, axis=0)
    y_ben = np.zeros((len(Xte_b),), dtype=np.int64)

    print("\n==== Test benign stats ====")
    print(f"TEST harmless raw={raw_te_b} valid={len(Xte_b)} skipped={sk_te_b}")

    # -------------------- Choose tau if needed --------------------
    tau = 0.0  # default margin threshold (z1-z0>=0)
    if args.target_fpr is not None:
        if args.tau_from == "test":
            calib_X = X_ben
        else:
            # use train benign features (already computed)
            if len(Xb) == 0:
                raise RuntimeError("No train benign features for tau_from=train.")
            calib_X = np.stack(Xb, axis=0)

        with torch.no_grad():
            calib_logits = model(torch.tensor(calib_X, dtype=torch.float32, device=device)).detach().cpu().numpy()
        calib_margins = logits_to_margin(calib_logits)
        tau = choose_tau_by_target_fpr(calib_margins, args.target_fpr)
        real_fpr = float(np.mean(calib_margins >= tau))
        print("\n==== Threshold selection (margin-based) ====")
        print(f"tau_from={args.tau_from} target_fpr={args.target_fpr:.6f} -> tau={tau:.6f} (real_fpr={real_fpr:.6f})")
        if args.tau_from == "train" and len(calib_margins) < 200:
            print(f"[WARN] train benign only {len(calib_margins)} samples; FPR step is coarse (~1/{len(calib_margins)}).")

    # -------------------- Pairwise tests: harmless vs each attack --------------------
    rows = []

    def eval_pair(pair_name: str, mal_dirs):
        Xm_list, _, raw_m, sk_m = collect_features_from_dirs(mal_dirs, label=1)
        if len(Xm_list) == 0:
            print(f"[WARN] no samples for {pair_name}, skip.")
            return
        X_m = np.stack(Xm_list, axis=0)
        y_m = np.ones((len(Xm_list),), dtype=np.int64)

        X_pair = np.concatenate([X_ben, X_m], axis=0)
        y_pair = np.concatenate([y_ben, y_m], axis=0)

        res = eval_with_tau(model, X_pair, y_pair, tau=tau, device=device, name=pair_name)
        print(f"\n==== PAIR: {pair_name} ====")
        print(res["report"])
        print("Confusion Matrix:\n", res["cm"])
        print(f"tau={res['tau']:.6f}  Acc={res['acc']:.4f}  TPR={res['TPR']:.4f}  FPR={res['FPR']:.4f}  AUC={res['AUC']:.4f}")
        print(f"[DATA] mal_raw={raw_m} mal_valid={len(Xm_list)} mal_skipped={sk_m}")

        rows.append({
            "pair": pair_name,
            "tau": res["tau"],
            "n": res["n"],
            "n_benign": res["n_benign"],
            "n_malicious": res["n_malicious"],
            "acc": res["acc"],
            "TPR": res["TPR"],
            "FPR": res["FPR"],
            "AUC": res["AUC"],
        })

    # optional: harmless vs harmful(test)
    if args.include_test_harmful_pair and os.path.isdir(test_harmful):
        eval_pair("harmless vs harmful(test)", [test_harmful])

    # each attack
    for ad in test_attack_dirs:
        name = os.path.basename(ad)
        eval_pair(f"harmless vs {name}", [ad])

    # -------------------- Save CSV --------------------
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.result_csv), exist_ok=True)
    df.to_csv(args.result_csv, index=False)
    print(f"\n[OK] Saved summary to: {args.result_csv}")

if __name__ == "__main__":
    main()
