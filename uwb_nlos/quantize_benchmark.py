"""
CC-6: INT8 Quantization Benchmark for ECA-UWB paper.

Applies PyTorch dynamic quantization to ECA-UWB.
Measures: model size (KB), accuracy on eWINE test set, inference latency.

Usage:
  python quantize_benchmark.py
"""

import sys, time, json, pickle, os, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from preprocess import _extract_cir_window, remove_outliers
from ecauwb_model import ECAUWBNet

DEVICE = "cpu"
SEED = 42

AUX_COLS = [1, 8, 6, 3, 4, 5, 7]
N_CIR = 50
N_AUX = 7

HP = dict(
    cir_len=50, n_aux=7, ch1=16, ch2=16, k1=5, k2=3, eca_k=3,
    aux_hid=32, feat_dim=16, clf_hid=32, dropout=0.3,
)


def load_and_split():
    """Load eWINE, split, scale — identical to train_ecauwb.py."""
    ewine_dir = Path(config.EWINE_DIR)
    csv_files = sorted(ewine_dir.glob("*.csv"))
    fp_col = config.EWINE_DIAG_COLS["FP_IDX"]
    rx_col = config.EWINE_DIAG_COLS["RXPACC"]

    all_X, all_labels = [], []
    for f in csv_files:
        arr = pd.read_csv(f, header=0).values
        labels = arr[:, 0].astype(np.int8)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(np.float32)
        rows = []
        for i in range(len(arr)):
            fp_idx = int(arr[i, fp_col])
            rxpacc = float(arr[i, rx_col]) + 1e-9
            cir_win = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc
            aux = arr[i, AUX_COLS].astype(np.float32)
            rows.append(np.concatenate([cir_win, aux]))
        all_X.append(np.array(rows, dtype=np.float32))
        all_labels.append(labels)

    X = np.concatenate(all_X, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    cir_part = X[:, :N_CIR]
    aux_part = X[:, N_CIR:]
    cir_part, aux_part, labels = remove_outliers(cir_part, aux_part, labels)
    X = np.concatenate([cir_part, aux_part], axis=1)

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, labels, test_size=0.15, stratify=labels, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=SEED)

    scaler = StandardScaler().fit(X_tr)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    return X_val_s, y_val, X_te_s, y_te


def evaluate(model, X_te, y_te, threshold=0.54):
    """Evaluate on test set."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te))
        probs = logits.sigmoid().numpy().flatten()

    preds = (probs > threshold).astype(int)
    trues = y_te.astype(int)

    return {
        "acc": accuracy_score(trues, preds),
        "f1": f1_score(trues, preds),
        "auc": roc_auc_score(trues, probs),
    }


def measure_model_size(model, label="model"):
    """Save model to temp file, measure size in KB."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_bytes = os.path.getsize(f.name)
        os.unlink(f.name)
    size_kb = size_bytes / 1024
    print(f"  {label} size: {size_kb:.1f} KB ({size_bytes} bytes)")
    return size_kb


def measure_latency(model, input_dim=57, n_runs=3000, warmup=100):
    """Measure inference latency."""
    model.eval()
    dummy = torch.zeros(1, input_dim)
    for _ in range(warmup):
        model(dummy)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(dummy)
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed_ms


def main():
    print("=" * 60)
    print("CC-6: INT8 Quantization Benchmark — ECA-UWB")
    print("=" * 60)

    # Load data
    print("\nLoading eWINE data ...")
    X_val, y_val, X_te, y_te = load_and_split()
    print(f"  Val: {len(y_val)}  Test: {len(y_te)}")

    # Load float32 model
    model_fp32 = ECAUWBNet(**HP)
    model_path = config.MODEL_DIR / "ecauwb.pt"
    model_fp32.load_state_dict(torch.load(model_path, map_location="cpu"))
    model_fp32.eval()

    print("\n── Float32 Baseline ──")
    fp32_size = measure_model_size(model_fp32, "Float32")
    fp32_metrics = evaluate(model_fp32, X_te, y_te)
    fp32_latency = measure_latency(model_fp32)
    print(f"  Acc: {fp32_metrics['acc']*100:.2f}%  F1: {fp32_metrics['f1']*100:.2f}%  "
          f"AUC: {fp32_metrics['auc']*100:.2f}%")
    print(f"  Latency: {fp32_latency:.3f} ms")

    # Dynamic INT8 quantization (quantizes Linear layers)
    print("\n── Dynamic INT8 Quantization ──")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    model_int8.eval()

    int8_size = measure_model_size(model_int8, "INT8 dynamic")
    int8_metrics = evaluate(model_int8, X_te, y_te)
    int8_latency = measure_latency(model_int8)
    print(f"  Acc: {int8_metrics['acc']*100:.2f}%  F1: {int8_metrics['f1']*100:.2f}%  "
          f"AUC: {int8_metrics['auc']*100:.2f}%")
    print(f"  Latency: {int8_latency:.3f} ms")

    # Summary
    acc_drop = (fp32_metrics["acc"] - int8_metrics["acc"]) * 100
    size_ratio = fp32_size / int8_size
    speedup = fp32_latency / int8_latency

    print(f"\n{'='*60}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Float32':>10} {'INT8':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Size (KB)':<20} {fp32_size:>10.1f} {int8_size:>10.1f} {size_ratio:>9.1f}×")
    print(f"  {'Accuracy (%)':<20} {fp32_metrics['acc']*100:>10.2f} "
          f"{int8_metrics['acc']*100:>10.2f} {-acc_drop:>+9.2f}")
    print(f"  {'F1-score (%)':<20} {fp32_metrics['f1']*100:>10.2f} "
          f"{int8_metrics['f1']*100:>10.2f}")
    print(f"  {'AUC-ROC (%)':<20} {fp32_metrics['auc']*100:>10.2f} "
          f"{int8_metrics['auc']*100:>10.2f}")
    print(f"  {'Latency (ms)':<20} {fp32_latency:>10.3f} {int8_latency:>10.3f} "
          f"{speedup:>9.1f}×")

    # Save
    results = {
        "float32": {
            "size_kb": round(fp32_size, 1),
            "accuracy": round(fp32_metrics["acc"] * 100, 2),
            "f1": round(fp32_metrics["f1"] * 100, 2),
            "auc": round(fp32_metrics["auc"] * 100, 2),
            "latency_ms": round(fp32_latency, 3),
        },
        "int8_dynamic": {
            "size_kb": round(int8_size, 1),
            "accuracy": round(int8_metrics["acc"] * 100, 2),
            "f1": round(int8_metrics["f1"] * 100, 2),
            "auc": round(int8_metrics["auc"] * 100, 2),
            "latency_ms": round(int8_latency, 3),
        },
        "accuracy_drop_pp": round(acc_drop, 2),
        "size_compression_ratio": round(size_ratio, 1),
    }

    out_path = config.LOG_DIR / "quantization_results.json"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
