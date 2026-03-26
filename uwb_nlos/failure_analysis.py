"""
CC-5: Failure-Mode Analysis for ECA-UWB paper.

Analyses the misclassified samples from the eWINE test set.
Groups by error type (FN vs FP) and environment.
Reports FP_AMP ratios, CIR_PWR statistics.

Usage:
  python failure_analysis.py
"""

import sys, json, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from preprocess import _extract_cir_window, remove_outliers
from ecauwb_model import ECAUWBNet

DEVICE = "cpu"
SEED = 42

# Same constants as train_ecauwb.py
AUX_COLS = [1, 8, 6, 3, 4, 5, 7]
N_CIR = 50
N_AUX = 7

HP = dict(
    cir_len=50, n_aux=7, ch1=16, ch2=16, k1=5, k2=3, eca_k=3,
    aux_hid=32, feat_dim=16, clf_hid=32, dropout=0.3,
)

# eWINE environment names (7 CSV files, alphabetical order)
ENV_NAMES = [
    "Apartment", "Bedroom", "Boiler Room",
    "Kitchen/Living Room", "Office 1", "Office 2", "Workshop"
]


def load_ewine_with_env():
    """Load eWINE data with environment labels preserved."""
    ewine_dir = Path(config.EWINE_DIR)
    csv_files = sorted(ewine_dir.glob("*.csv"))

    all_X, all_labels, all_envs, all_raw_diag = [], [], [], []
    fp_col = config.EWINE_DIAG_COLS["FP_IDX"]
    rx_col = config.EWINE_DIAG_COLS["RXPACC"]

    for env_idx, f in enumerate(csv_files):
        arr = pd.read_csv(f, header=0).values
        labels = arr[:, 0].astype(np.int8)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(np.float32)

        rows = []
        raw_diags = []
        for i in range(len(arr)):
            fp_idx = int(arr[i, fp_col])
            rxpacc = float(arr[i, rx_col]) + 1e-9
            cir_win = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc
            aux = arr[i, AUX_COLS].astype(np.float32)
            rows.append(np.concatenate([cir_win, aux]))

            # Raw diagnostics for analysis
            raw_diags.append({
                "FP_AMP1": float(arr[i, 3]),
                "FP_AMP2": float(arr[i, 4]),
                "FP_AMP3": float(arr[i, 5]),
                "CIR_PWR": float(arr[i, 7]),
                "STDEV_NOISE": float(arr[i, 6]),
                "MAX_NOISE": float(arr[i, 8]),
                "RXPACC": float(arr[i, rx_col]),
                "RANGE": float(arr[i, 1]),
            })

        all_X.append(np.array(rows, dtype=np.float32))
        all_labels.append(labels)
        all_envs.append(np.full(len(labels), env_idx, dtype=np.int8))
        all_raw_diag.extend(raw_diags)

    X = np.concatenate(all_X, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    envs = np.concatenate(all_envs, axis=0)

    # Remove outliers (same logic as training)
    cir_part = X[:, :N_CIR]
    aux_part = X[:, N_CIR:]
    mask = np.abs((cir_part - cir_part.mean(0)) / (cir_part.std(0) + 1e-9)).max(1) < 6
    X = X[mask]
    labels = labels[mask]
    envs = envs[mask]
    all_raw_diag = [all_raw_diag[i] for i in range(len(mask)) if mask[i]]

    return X, labels, envs, all_raw_diag


def main():
    print("Loading eWINE with environment labels ...")
    X, labels, envs, raw_diag = load_ewine_with_env()
    print(f"Total samples: {len(labels)}")

    # Same split as training (seed=42)
    indices = np.arange(len(labels))
    idx_tv, idx_te = train_test_split(
        indices, test_size=0.15, stratify=labels, random_state=SEED)
    idx_tr, idx_val = train_test_split(
        idx_tv, test_size=0.15/0.85, stratify=labels[idx_tv], random_state=SEED)

    X_tr, X_te = X[idx_tr], X[idx_te]
    y_te = labels[idx_te]
    envs_te = envs[idx_te]
    diag_te = [raw_diag[i] for i in idx_te]

    scaler = StandardScaler().fit(X_tr)
    X_te_scaled = scaler.transform(X_te).astype(np.float32)

    # Load model
    model_path = config.MODEL_DIR / "ecauwb.pt"
    model = ECAUWBNet(**HP)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Predict
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te_scaled))
        probs = logits.sigmoid().numpy()

    threshold = 0.54
    preds = (probs > threshold).astype(int).flatten()
    trues = y_te.astype(int)

    # Find misclassified
    misclassified_mask = preds != trues
    n_misc = misclassified_mask.sum()
    print(f"\nMisclassified: {n_misc} / {len(trues)} ({n_misc/len(trues)*100:.2f}%)")

    # Classify error types
    fn_mask = (trues == 1) & (preds == 0)  # NLOS predicted as LOS
    fp_mask = (trues == 0) & (preds == 1)  # LOS predicted as NLOS
    print(f"  False Negatives (NLOS→LOS): {fn_mask.sum()}")
    print(f"  False Positives (LOS→NLOS): {fp_mask.sum()}")

    # Analyse by environment
    print(f"\n{'='*70}")
    print(f"{'Environment':<22} {'Total':>6} {'FN':>5} {'FP':>5} {'FN%':>6} {'FP%':>6}")
    print(f"{'-'*70}")

    env_stats = {}
    for env_idx, env_name in enumerate(ENV_NAMES):
        env_mask = envs_te == env_idx
        total = env_mask.sum()
        fn_count = (fn_mask & env_mask).sum()
        fp_count = (fp_mask & env_mask).sum()
        fn_pct = fn_count / max(total, 1) * 100
        fp_pct = fp_count / max(total, 1) * 100
        print(f"{env_name:<22} {total:>6} {fn_count:>5} {fp_count:>5} "
              f"{fn_pct:>5.1f}% {fp_pct:>5.1f}%")
        env_stats[env_name] = {
            "total": int(total), "fn": int(fn_count), "fp": int(fp_count),
            "fn_pct": round(fn_pct, 2), "fp_pct": round(fp_pct, 2),
        }

    # Diagnostic feature analysis for misclassified vs correct
    print(f"\n{'='*70}")
    print("Diagnostic feature comparison: Misclassified vs Correctly classified")
    print(f"{'='*70}")

    for error_type, mask in [("False Negatives", fn_mask), ("False Positives", fp_mask)]:
        err_indices = np.where(mask)[0]
        correct_same_class = np.where(
            (~misclassified_mask) & (trues == (1 if "Neg" in error_type else 0)))[0]

        if len(err_indices) == 0:
            continue

        print(f"\n  {error_type} ({len(err_indices)} samples):")
        for feat in ["FP_AMP1", "FP_AMP2", "FP_AMP3", "CIR_PWR"]:
            err_vals = [diag_te[i][feat] for i in err_indices]
            cor_vals = [diag_te[i][feat] for i in correct_same_class]
            print(f"    {feat:>12}: err={np.mean(err_vals):>10.1f} ± {np.std(err_vals):>8.1f}"
                  f"  |  correct={np.mean(cor_vals):>10.1f} ± {np.std(cor_vals):>8.1f}")

    # FP_AMP ratio analysis (FP_AMP1 / CIR_PWR)
    print(f"\n{'='*70}")
    print("FP_AMP1/CIR_PWR ratio analysis")
    print(f"{'='*70}")
    for error_type, mask in [("False Negatives", fn_mask), ("False Positives", fp_mask)]:
        err_indices = np.where(mask)[0]
        correct_indices = np.where(~misclassified_mask)[0]
        if len(err_indices) == 0:
            continue
        err_ratios = [diag_te[i]["FP_AMP1"] / (diag_te[i]["CIR_PWR"] + 1e-9) for i in err_indices]
        cor_ratios = [diag_te[i]["FP_AMP1"] / (diag_te[i]["CIR_PWR"] + 1e-9) for i in correct_indices]
        print(f"  {error_type}: ratio={np.mean(err_ratios):.4f} ± {np.std(err_ratios):.4f}")
        print(f"  Correct:          ratio={np.mean(cor_ratios):.4f} ± {np.std(cor_ratios):.4f}")

    # Save results
    results = {
        "total_test": int(len(trues)),
        "total_misclassified": int(n_misc),
        "false_negatives": int(fn_mask.sum()),
        "false_positives": int(fp_mask.sum()),
        "threshold": threshold,
        "per_environment": env_stats,
    }
    out_path = config.LOG_DIR / "failure_analysis.json"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
