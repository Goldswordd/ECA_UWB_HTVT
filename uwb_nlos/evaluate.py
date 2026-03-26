"""
Evaluation & benchmarking.

Usage:
  python evaluate.py --model cnn       # evaluate saved CNN on test set
  python evaluate.py --model rf        # evaluate saved RF on test set
  python evaluate.py --latency         # benchmark inference latency on this machine
  python evaluate.py --model cnn --latency

Outputs:
  logs/eval_cnn.json / eval_rf.json    — metrics JSON
  logs/confusion_matrix_*.png
  logs/roc_curve_*.png
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score,
)

import config
from preprocess import build_dataset


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Shared test data loader
# ─────────────────────────────────────────────────────────────

_SPLITS_CACHE = None

def get_test_data():
    global _SPLITS_CACHE
    if _SPLITS_CACHE is None:
        _SPLITS_CACHE = build_dataset(use_oiud=False)
    s = _SPLITS_CACHE
    return s[6], s[7], s[8]   # cir_te, feat_te, y_te


def make_X_flat(cir, feat):
    """For sklearn models: concatenate (N,50) + (N,11) → (N,61)."""
    return np.concatenate([cir.squeeze(-1), feat], axis=1)


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

def plot_confusion(cm, name, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["LOS", "NLOS"]); ax.set_yticklabels(["LOS", "NLOS"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Confusion matrix → {save_path}")


def plot_roc(fpr, tpr, auc, name, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  ROC curve → {save_path}")


# ─────────────────────────────────────────────────────────────
# Core metric computation
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, name: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-9)    # recall for NLOS
    specificity = tn / (tn + fp + 1e-9)    # recall for LOS

    metrics = {
        "model":       name,
        "n_test":      int(len(y_true)),
        "accuracy":    round(acc,  4),
        "f1_nlos":     round(f1,   4),
        "roc_auc":     round(auc,  4),
        "avg_prec":    round(ap,   4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    print(f"\n{'─'*50}")
    print(f"  Model     : {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (NLOS) : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Sensitivity (NLOS recall): {sensitivity:.4f}")
    print(f"  Specificity (LOS recall) : {specificity:.4f}")
    print(classification_report(y_true, y_pred, target_names=["LOS", "NLOS"]))

    # Plots
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag = name.lower().replace(" ", "_")
    plot_confusion(cm, name, config.LOG_DIR / f"confusion_{tag}.png")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plot_roc(fpr, tpr, auc, name, config.LOG_DIR / f"roc_{tag}.png")

    return metrics


# ─────────────────────────────────────────────────────────────
# CNN evaluator
# ─────────────────────────────────────────────────────────────

def evaluate_cnn():
    import torch
    from model import NLOSClassifier

    cir_te, feat_te, y_te = get_test_data()
    model_path = config.KERAS_MODEL_PATH.with_suffix(".pt")
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model not found: {model_path}\nRun: python train.py --model cnn")

    model = NLOSClassifier(n_feat=config.N_FEATURES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    cir_t  = torch.tensor(cir_te,  dtype=torch.float32)
    feat_t = torch.tensor(feat_te, dtype=torch.float32)

    with torch.no_grad():
        probs = model.predict_proba(cir_t, feat_t).squeeze(1).numpy()

    preds = (probs > 0.5).astype(int)
    metrics = compute_metrics(y_te, preds, probs, "CNN (Dual-Branch)")
    _save_json(metrics, config.LOG_DIR / "eval_cnn.json")
    return metrics


# ─────────────────────────────────────────────────────────────
# RF evaluator
# ─────────────────────────────────────────────────────────────

def evaluate_rf():
    cir_te, feat_te, y_te = get_test_data()
    rf_path = config.MODEL_DIR / "rf_model.pkl"
    if not rf_path.exists():
        raise FileNotFoundError(f"RF model not found: {rf_path}\nRun: python train.py --model rf")

    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    X_te = make_X_flat(cir_te, feat_te)
    probs = rf.predict_proba(X_te)[:, 1]
    preds = rf.predict(X_te)
    metrics = compute_metrics(y_te, preds, probs, "Random Forest")
    _save_json(metrics, config.LOG_DIR / "eval_rf.json")
    return metrics


# ─────────────────────────────────────────────────────────────
# Latency benchmark
# ─────────────────────────────────────────────────────────────

def benchmark_latency(n_warmup: int = 50, n_runs: int = 1000):
    print(f"\n{'='*50}")
    print(f"Latency benchmark ({n_runs} runs after {n_warmup} warmup)")
    print(f"{'='*50}")

    cir_te, feat_te, y_te = get_test_data()
    # Use single-sample inference (realistic for real-time)
    cir1  = cir_te[:1]
    feat1 = feat_te[:1]

    results = {}

    # CNN
    try:
        import torch
        from model import NLOSClassifier
        model_path = config.KERAS_MODEL_PATH.with_suffix(".pt")
        if model_path.exists():
            model = NLOSClassifier(n_feat=config.N_FEATURES)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            cir_t  = torch.tensor(cir1,  dtype=torch.float32)
            feat_t = torch.tensor(feat1, dtype=torch.float32)
            with torch.no_grad():
                for _ in range(n_warmup):
                    model.predict_proba(cir_t, feat_t)
                times = []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    model.predict_proba(cir_t, feat_t)
                    times.append((time.perf_counter() - t0) * 1e3)
            print(f"  CNN     : mean={np.mean(times):.3f} ms  "
                  f"p95={np.percentile(times,95):.3f} ms  "
                  f"max={np.max(times):.3f} ms")
            results["cnn"] = {"mean_ms": round(np.mean(times), 3),
                               "p95_ms":  round(np.percentile(times, 95), 3)}
    except Exception as e:
        print(f"  CNN skipped: {e}")

    # ONNX
    onnx_path = config.MODEL_DIR / "nlos_classifier.onnx"
    try:
        import onnxruntime as ort
        if onnx_path.exists():
            sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])
            for _ in range(n_warmup):
                sess.run(None, {"cir": cir1, "feat": feat1})
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                sess.run(None, {"cir": cir1, "feat": feat1})
                times.append((time.perf_counter() - t0) * 1e3)
            print(f"  ONNX    : mean={np.mean(times):.3f} ms  "
                  f"p95={np.percentile(times,95):.3f} ms  "
                  f"max={np.max(times):.3f} ms")
            results["onnx"] = {"mean_ms": round(np.mean(times), 3),
                                "p95_ms":  round(np.percentile(times, 95), 3)}
    except Exception as e:
        print(f"  ONNX skipped: {e}")

    # RF
    rf_path = config.MODEL_DIR / "rf_model.pkl"
    if rf_path.exists():
        with open(rf_path, "rb") as fh:
            rf = pickle.load(fh)
        X1 = make_X_flat(cir1, feat1)
        for _ in range(n_warmup):
            rf.predict_proba(X1)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            rf.predict_proba(X1)
            times.append((time.perf_counter() - t0) * 1e3)
        print(f"  RF      : mean={np.mean(times):.3f} ms  "
              f"p95={np.percentile(times,95):.3f} ms  "
              f"max={np.max(times):.3f} ms")
        results["rf"] = {"mean_ms": round(np.mean(times), 3),
                          "p95_ms":  round(np.percentile(times, 95), 3)}

    print(f"\n  (DWM1001 delivers ~80 samples/s = 12.5 ms budget per sample)")
    _save_json(results, config.LOG_DIR / "latency_benchmark.json")
    return results


# ─────────────────────────────────────────────────────────────
# Feature importance (RF only)
# ─────────────────────────────────────────────────────────────

def plot_feature_importance():
    rf_path = config.MODEL_DIR / "rf_model.pkl"
    if not rf_path.exists():
        return
    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    # Feature names: 50 CIR bins + 11 global
    names = [f"cir_{i}" for i in range(config.CIR_LEN)] + config.FEATURE_NAMES
    imp   = rf.feature_importances_
    idx   = np.argsort(imp)[-20:]  # top 20

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), imp[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx])
    ax.set_title("Top-20 Feature Importances (Random Forest)")
    ax.set_xlabel("Mean decrease in impurity")
    plt.tight_layout()
    path = config.LOG_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150)
    print(f"  Feature importance → {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["cnn", "rf", "both"])
    parser.add_argument("--latency", action="store_true")
    args = parser.parse_args()

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.model in ("rf", "both"):
        try:
            evaluate_rf()
            plot_feature_importance()
        except FileNotFoundError as e:
            print(e)

    if args.model in ("cnn", "both"):
        try:
            evaluate_cnn()
        except (FileNotFoundError, ImportError) as e:
            print(e)

    if args.latency:
        benchmark_latency()


if __name__ == "__main__":
    main()
