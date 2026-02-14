import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

EPOCHS = 30
LR = 1e-3
TARGET_TPR = 1.0
SEGMENT_CONFIGS = [(0, "front5")]
MAX_TOKENS_TO_USE = 3

MODELS = {
    "llama3-8b-instruct": {
        "benign_dir": "hs_outputs_llama3-8b-instruct/benign",
        "root_dir":   "hs_outputs_llama3-8b-instruct",
        "layer_groups": [
            [2, 3, 4, 5, 6],
            [12, 13, 14, 15, 16],
            [28, 29, 30, 31, 32]
        ]
    },
    "mistral-7b-v0.2": {
        "benign_dir": "hs_outputs_mistral-7b-v0.2/benign",
        "root_dir":   "hs_outputs_mistral-7b-v0.2",
        "layer_groups": [
            [2, 3, 4, 5, 6],
            [12, 13, 14, 15, 16],
            [28, 29, 30, 31, 32]
        ]
    },
    "qwen2.5-7b-instruct": {
        "benign_dir": "hs_outputs_qwen2.5-7b-instruct/benign",
        "root_dir":   "hs_outputs_qwen2.5-7b-instruct",
        "layer_groups": [
            [2, 3, 4, 5, 6],
            [12, 13, 14, 15, 16],
            [24, 25, 26, 27, 28]
        ]
    }
}

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def load_pooled_features(path, segment_idx, num_tokens, group_idx, sub_idx, pool_type="mean"):
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr.ndim != 5:
        return None
    if segment_idx >= arr.shape[0]:
        return None
    if num_tokens > arr.shape[1]:
        return None
    if group_idx >= arr.shape[2] or sub_idx >= arr.shape[3]:
        return None
    slice_tokens = arr[segment_idx, :num_tokens, group_idx, sub_idx, :]
    if pool_type == "mean":
        feat = slice_tokens.mean(axis=0)
    elif pool_type == "max":
        feat = slice_tokens.max(axis=0)
    else:
        feat = slice_tokens.mean(axis=0)
    return feat.astype(np.float32)

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.net(x)

def process_model(model_name, cfg):
    benign_dir = cfg["benign_dir"]
    root_dir = cfg["root_dir"]
    layer_groups = cfg["layer_groups"]

    FLATTENED_LAYERS = [l for group in layer_groups for l in group]
    N_LAYERS = len(FLATTENED_LAYERS)
    assert N_LAYERS == 15, "layer_groups must be 3x5"

    SELECTED_FLAT_INDICES = [0, N_LAYERS // 2, N_LAYERS - 1]
    MID_LAYER_FLAT_IDX = N_LAYERS // 2

    os.makedirs("result", exist_ok=True)

    benign_files = sorted(
        [os.path.join(benign_dir, f) for f in os.listdir(benign_dir) if f.endswith(".npy")],
        key=natural_key
    )

    print(f"\n========== Model: {model_name} ==========")
    print(f"Benign samples: {len(benign_files)}")

    malicious_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.lower() != "benign"
    ]

    print("Detected malicious datasets:", malicious_dirs)

    fpr_by_dataset = {}

    for malicious_name in malicious_dirs:
        print("\n" + "#" * 100)
        print(f"[{model_name}] Processing malicious dataset: {malicious_name}")
        print("#" * 100)

        malicious_root = os.path.join(root_dir, malicious_name)

        malicious_files = sorted(
            [os.path.join(malicious_root, f) for f in os.listdir(malicious_root) if f.endswith(".npy")],
            key=natural_key
        )

        print(f"Malicious samples: {len(malicious_files)}")

        RESULT_CSV = f"result/{model_name}_{malicious_name}_pooled_frontK3_midlayer_linear.csv"
        results = []

        for segment_idx, segment_name in SEGMENT_CONFIGS:
            for num_tokens in range(1, MAX_TOKENS_TO_USE + 1):
                for flat_idx in SELECTED_FLAT_INDICES:
                    layer_id = FLATTENED_LAYERS[flat_idx]
                    group_idx = flat_idx // 5
                    sub_idx = flat_idx % 5

                    X_list, y_list = [], []

                    for p in benign_files:
                        feat = load_pooled_features(p, segment_idx, num_tokens, group_idx, sub_idx, pool_type="mean")
                        if feat is not None:
                            X_list.append(feat)
                            y_list.append(0)

                    for p in malicious_files:
                        feat = load_pooled_features(p, segment_idx, num_tokens, group_idx, sub_idx, pool_type="mean")
                        if feat is not None:
                            X_list.append(feat)
                            y_list.append(1)

                    if len(set(y_list)) < 2:
                        continue

                    X = np.stack(X_list)
                    y = np.array(y_list)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.6, stratify=y, random_state=42
                    )

                    model = LinearClassifier(input_dim=X.shape[1])
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=LR)

                    for _ in range(EPOCHS):
                        model.train()
                        optimizer.zero_grad()
                        loss = criterion(
                            model(torch.tensor(X_train, dtype=torch.float32)),
                            torch.tensor(y_train, dtype=torch.long)
                        )
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        logits = model(torch.tensor(X_test, dtype=torch.float32))
                        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
                        preds = logits.argmax(dim=1).numpy()

                    cm = confusion_matrix(y_test, preds)
                    TN, FP, FN, TP = cm.ravel()

                    TPR = TP / (TP + FN + 1e-12)
                    FPR = FP / (FP + TN + 1e-12)
                    ACC = (preds == y_test).mean()

                    try:
                        AUC = roc_auc_score(y_test, probs)
                    except Exception:
                        AUC = float("nan")

                    try:
                        fpr_arr, tpr_arr, thresholds = roc_curve(y_test, probs)
                        idx_best = np.argmin(np.abs(tpr_arr - TARGET_TPR))
                        TPR_at_target = tpr_arr[idx_best]
                        FPR_at_target = fpr_arr[idx_best]
                    except Exception:
                        TPR_at_target = float("nan")
                        FPR_at_target = float("nan")

                    results.append({
                        "model": model_name,
                        "malicious_dataset": malicious_name,
                        "segment_name": segment_name,
                        "num_tokens": num_tokens,
                        "layer_flat_idx": flat_idx,
                        "layer_id": layer_id,
                        "accuracy": ACC,
                        "TPR": TPR,
                        "FPR": FPR,
                        "AUC": AUC,
                        "TPR_at_1.0": TPR_at_target,
                        "FPR_at_1.0": FPR_at_target
                    })

        df = pd.DataFrame(results)
        df.to_csv(RESULT_CSV, index=False)
        print(f"Saved results to: {RESULT_CSV}")

        df_mid = df[(df["segment_name"] == "front5") & (df["layer_flat_idx"] == MID_LAYER_FLAT_IDX)]

        if df_mid.empty:
            print(f"[Warning] No mid-layer front5 data for {model_name}-{malicious_name}, skip in merged FPR plot.")
            continue

        fpr_series = df_mid.groupby("num_tokens")["FPR_at_1.0"].mean().reindex(range(1, MAX_TOKENS_TO_USE + 1))
        fpr_by_dataset[malicious_name] = fpr_series

    if len(fpr_by_dataset) == 0:
        print(f"No mid-layer FPR_at_1.0 data collected for model {model_name}. Skip merged figure.")
        return

    datasets = list(fpr_by_dataset.keys())
    n_ds = len(datasets)

    if n_ds <= 2:
        n_rows, n_cols = 1, n_ds
    else:
        n_rows, n_cols = 2, int(np.ceil(n_ds / 2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    for idx, (dataset_name, fpr_series) in enumerate(fpr_by_dataset.items()):
        ax = axes[idx]
        ax.plot(fpr_series.index, fpr_series.values, marker="o", linewidth=2)
        ax.set_title(dataset_name, fontsize=12)
        ax.set_xlabel("Number of pooled tokens (front-K, K≤3)", fontsize=11)
        ax.set_ylabel("FPR at TPR≈1.0", fontsize=11)
        ax.set_xticks(range(1, MAX_TOKENS_TO_USE + 1))
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.text(0.02, 0.95, panel_labels[idx], transform=ax.transAxes, fontsize=13, fontweight="bold", verticalalignment="top")

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f"{model_name}: FPR at TPR≈1.0 for Pooled Front-K (K≤3) Tokens (Mid-layer, Linear Classifier)",
        fontsize=14,
        y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    merged_fig_path = f"result/{model_name}_midlayer_frontK3_FPR_at_1.0_all_linear.png"
    plt.savefig(merged_fig_path, dpi=300)
    plt.close()
    print(f"Saved merged FPR figure for {model_name} to: {merged_fig_path}")

def main():
    for model_name, cfg in MODELS.items():
        process_model(model_name, cfg)

if __name__ == "__main__":
    main()
