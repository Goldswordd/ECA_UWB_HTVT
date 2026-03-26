"""
Train LightMamba on the eWINE dataset and compare with CNN Dual-Branch.

Usage:
  python train_lightmamba.py            # train + compare
  python train_lightmamba.py --eval     # skip training, just evaluate saved model

Differences from train.py (CNN):
  • Uses full 1016-point CIR  (not 50-point FP window)
  • Uses 14 raw auxiliary features from eWINE columns 1-14
  • Separate StandardScalers for CIR and aux (CIR is also standardized)
  • Hyperparameters match paper: AdamW, BCEWithLogitsLoss, patience=30, batch=128
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from lightmamba_model import LightMamba

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = config.MODEL_DIR / "lightmamba.pt"
SCALER_CIR   = config.MODEL_DIR / "lm_scaler_cir.pkl"
SCALER_AUX   = config.MODEL_DIR / "lm_scaler_aux.pkl"
HISTORY_PNG  = config.LOG_DIR   / "lm_training_history.png"
CM_PNG       = config.LOG_DIR   / "lm_confusion_matrix.png"
RESULTS_JSON = config.LOG_DIR   / "lm_results.json"

config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
config.LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters (paper Table II)
# ─────────────────────────────────────────────────────────────────────────────
HP = dict(
    cir_len    = 1016,
    aux_dim    = 14,
    T          = 8,
    coding_dim = 32,
    num_layers = 1,
    d_state    = 16,
    dropout    = 0.3,
    batch_size = 128,
    lr         = 1e-3,
    weight_decay = 1e-4,
    patience   = 30,
    max_epochs = 300,
    seed       = config.SEED,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading  (full CIR + 14 raw aux features)
# ─────────────────────────────────────────────────────────────────────────────

# eWINE column layout (from config.py comments):
#   col 0     : NLOS label
#   col 1-14  : 14 aux features  (RANGE, FP_IDX, FP_AMP1..3, STDEV_NOISE,
#                                  CIR_PWR, MAX_NOISE, RXPACC, CH,
#                                  FRAME_LEN, PREAM_LEN, BITRATE, PRFR)
#   col 15-1030: CIR[0]..CIR[1015]
AUX_COLS = list(range(1, 15))    # 14 aux features
CIR_COLS = list(range(15, 1031)) # 1016 CIR samples


def load_ewine_full() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all 7 eWINE CSVs.

    Returns
    -------
    cir  : (N, 1016) float32 — raw CIR (NOT normalized here; scaler does it)
    aux  : (N, 14)   float32 — raw aux features
    labels: (N,)     int8
    """
    csv_files = sorted(config.EWINE_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV in {config.EWINE_DIR}")

    all_cir, all_aux, all_labels = [], [], []
    print("=" * 60)
    print("Loading eWINE (full CIR + 14 aux) …")
    for f in csv_files:
        df  = pd.read_csv(f, header=0)
        arr = df.values

        labels = arr[:, 0].astype(np.int8)
        cir    = arr[:, CIR_COLS].astype(np.float32)
        aux    = arr[:, AUX_COLS].astype(np.float32)

        all_cir.append(cir)
        all_aux.append(aux)
        all_labels.append(labels)
        print(f"  {f.name}: {len(labels)} rows  "
              f"LOS={int((labels==0).sum())} NLOS={int((labels==1).sum())}")

    cir    = np.concatenate(all_cir,    axis=0)
    aux    = np.concatenate(all_aux,    axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"\nTotal: {len(labels)} | LOS={int((labels==0).sum())} "
          f"NLOS={int((labels==1).sum())}")
    return cir, aux, labels


def remove_outliers_lm(cir, aux, labels, z_thresh=6.0):
    """
    Remove samples with extreme aux feature values (class-conditional z-score).
    """
    keep = np.ones(len(labels), dtype=bool)
    for cls in [0, 1]:
        idx = labels == cls
        mu  = aux[idx].mean(0)
        sig = aux[idx].std(0) + 1e-9
        z   = np.abs((aux - mu) / sig)
        keep &= ~((labels == cls) & (z.max(1) > z_thresh))
    n_rm = int((~keep).sum())
    print(f"  Removed {n_rm} outliers (aux z > {z_thresh})")
    return cir[keep], aux[keep], labels[keep]


def build_dataset_lm():
    """
    Full pipeline for LightMamba:
      load → outlier removal → split → standardize CIR + aux → tensors
    """
    cir, aux, labels = load_ewine_full()

    print("\nRemoving outliers …")
    cir, aux, labels = remove_outliers_lm(cir, aux, labels)

    print("\nSplitting (70/15/15 stratified) …")
    # First split off test set (15%)
    cir_tv, cir_te, aux_tv, aux_te, y_tv, y_te = train_test_split(
        cir, aux, labels,
        test_size=0.15, stratify=labels, random_state=HP["seed"]
    )
    # Then val from remaining
    val_frac = 0.15 / 0.85
    cir_tr, cir_val, aux_tr, aux_val, y_tr, y_val = train_test_split(
        cir_tv, aux_tv, y_tv,
        test_size=val_frac, stratify=y_tv, random_state=HP["seed"]
    )
    print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_te)}")

    # ── Standardize (fit on train only) ──────────────────────────────
    sc_cir = StandardScaler().fit(cir_tr)
    sc_aux = StandardScaler().fit(aux_tr)

    with open(SCALER_CIR, "wb") as f: pickle.dump(sc_cir, f)
    with open(SCALER_AUX, "wb") as f: pickle.dump(sc_aux, f)
    print(f"  Scalers saved → {SCALER_CIR.name}, {SCALER_AUX.name}")

    def to_tensors(c, a, y):
        c = torch.tensor(sc_cir.transform(c), dtype=torch.float32)
        a = torch.tensor(sc_aux.transform(a), dtype=torch.float32)
        y = torch.tensor(y.astype(np.float32))
        return c, a, y

    tr  = to_tensors(cir_tr,  aux_tr,  y_tr)
    val = to_tensors(cir_val, aux_val, y_val)
    te  = to_tensors(cir_te,  aux_te,  y_te)

    print("=" * 60)
    return tr, val, te


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training
# ─────────────────────────────────────────────────────────────────────────────

def make_loader(tensors, batch_size, shuffle):
    ds = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_lightmamba(tr, val):
    torch.manual_seed(HP["seed"])
    device = torch.device("cpu")

    model = LightMamba(
        cir_len    = HP["cir_len"],
        aux_dim    = HP["aux_dim"],
        T          = HP["T"],
        coding_dim = HP["coding_dim"],
        num_layers = HP["num_layers"],
        d_state    = HP["d_state"],
        dropout    = HP["dropout"],
    ).to(device)

    n_params = model.count_params()
    print(f"\n[LightMamba]  Parameters: {n_params:,}")

    loader_tr  = make_loader(tr,  HP["batch_size"], shuffle=True)
    loader_val = make_loader(val, HP["batch_size"], shuffle=False)

    # Paper: AdamW + BCEWithLogitsLoss + class weights
    n_pos = int(tr[2].sum().item())
    n_neg = len(tr[2]) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HP["lr"], weight_decay=HP["weight_decay"]
    )
    # Cosine LR decay (not in paper but improves stability)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP["max_epochs"], eta_min=1e-5
    )

    best_val_loss = float("inf")
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    print(f"\n  {'Ep':>4}  {'TrainLoss':>10}  {'ValLoss':>10}  "
          f"{'TrainF1':>9}  {'ValF1':>7}  {'LR':>9}")
    print("  " + "─" * 62)

    t0 = time.time()
    for epoch in range(1, HP["max_epochs"] + 1):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_preds, tr_true = 0.0, [], []
        for cir_b, aux_b, y_b in loader_tr:
            cir_b, aux_b, y_b = cir_b.to(device), aux_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(cir_b, aux_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # paper: gradient clipping
            optimizer.step()
            tr_loss += loss.item() * len(y_b)
            tr_preds.extend((logits.detach() > 0).cpu().numpy())
            tr_true.extend(y_b.cpu().numpy())

        tr_loss /= len(tr[2])
        tr_f1    = f1_score(tr_true, tr_preds, zero_division=0)

        # ── validate ────────────────────────────────────────────────────
        model.eval()
        val_loss, val_preds, val_true = 0.0, [], []
        with torch.no_grad():
            for cir_b, aux_b, y_b in loader_val:
                cir_b, aux_b, y_b = cir_b.to(device), aux_b.to(device), y_b.to(device)
                logits   = model(cir_b, aux_b)
                val_loss += criterion(logits, y_b).item() * len(y_b)
                val_preds.extend((logits > 0).cpu().numpy())
                val_true.extend(y_b.cpu().numpy())

        val_loss /= len(val[2])
        val_f1    = f1_score(val_true, val_preds, zero_division=0)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(val_f1)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  {epoch:>4}  {tr_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{tr_f1:>9.4f}  {val_f1:>7.4f}  {lr_now:>9.2e}")

        # ── early stopping ──────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve += 1
            if no_improve >= HP["patience"]:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        scheduler.step()

    elapsed = time.time() - t0
    print(f"\n  Best val loss : {best_val_loss:.4f}")
    print(f"  Total time    : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Model saved   → {MODEL_PATH}")

    # ── plot training curves ─────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history["train_f1"],   label="Train F1")
    ax2.plot(history["val_f1"],     label="Val F1")
    ax2.set_title("F1-score"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle("LightMamba Training History")
    plt.tight_layout()
    plt.savefig(HISTORY_PNG, dpi=150)
    plt.close()
    print(f"  Training curves → {HISTORY_PNG}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Evaluation + comparison
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_lightmamba(te):
    device = torch.device("cpu")
    model  = LightMamba(**{k: HP[k] for k in
                           ["cir_len","aux_dim","T","coding_dim",
                            "num_layers","d_state","dropout"]}).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    loader = make_loader(te, HP["batch_size"], shuffle=False)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for cir_b, aux_b, y_b in loader:
            logits = model(cir_b.to(device), aux_b.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y_b.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(int)
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()
    preds  = (logits > 0).astype(int)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, pos_label=1)
    auc = roc_auc_score(labels, probs)
    cm  = confusion_matrix(labels, preds)

    print("\n" + "─" * 50)
    print(f"  Model     : LightMamba (paper arch)")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (NLOS) : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Sensitivity (NLOS recall): {cm[1,1]/(cm[1,0]+cm[1,1]+1e-9):.4f}")
    print(f"  Specificity (LOS  recall): {cm[0,0]/(cm[0,0]+cm[0,1]+1e-9):.4f}")
    print(classification_report(labels, preds, target_names=["LOS","NLOS"]))

    # ── confusion matrix plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["LOS","NLOS"]); ax.set_yticklabels(["LOS","NLOS"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("LightMamba Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {CM_PNG}")

    return dict(acc=float(acc), f1=float(f1), auc=float(auc),
                params=model.count_params())


def latency_benchmark(n_runs=1000, warmup=50):
    """Measure per-sample inference latency on CPU."""
    device = torch.device("cpu")
    model  = LightMamba(**{k: HP[k] for k in
                           ["cir_len","aux_dim","T","coding_dim",
                            "num_layers","d_state","dropout"]}).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    cir_dummy = torch.randn(1, HP["cir_len"])
    aux_dummy = torch.randn(1, HP["aux_dim"])

    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(cir_dummy, aux_dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(cir_dummy, aux_dummy)
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"\n  Latency ({n_runs} runs, single sample):")
    print(f"    mean={times.mean():.3f} ms  "
          f"p95={np.percentile(times,95):.3f} ms  "
          f"max={times.max():.3f} ms")
    return times.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def load_cnn_results():
    """Try to load CNN evaluation results saved by evaluate.py."""
    path     = config.LOG_DIR / "eval_cnn.json"
    lat_path = config.LOG_DIR / "latency_benchmark.json"
    res = dict(acc=0.8882, f1=0.8837, auc=0.9533, params=7441, latency_ms=0.66)
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        res["acc"]    = d.get("accuracy", res["acc"])
        res["f1"]     = d.get("f1_nlos",  res["f1"])
        res["auc"]    = d.get("roc_auc",  res["auc"])
    if lat_path.exists():
        with open(lat_path) as f:
            d = json.load(f)
        res["latency_ms"] = d.get("cnn", {}).get("mean_ms", res["latency_ms"])
    return res


def print_comparison(lm_res: dict, lm_latency: float):
    cnn = load_cnn_results()

    print("\n" + "=" * 72)
    print("  COMPARISON TABLE — eWINE Dataset")
    print("=" * 72)
    print(f"  {'Model':<28} {'Acc':>7}  {'F1':>7}  {'AUC':>7}  "
          f"{'Params':>8}  {'ms/sample':>10}")
    print("  " + "─" * 68)

    # Paper reference numbers
    paper_refs = [
        ("RF (baseline)",          0.8794, 0.8748, 0.9489, "—",     "152.6"),
        ("CNN Dual-Branch (ours)", cnn["acc"], cnn["f1"], cnn["auc"],
         f"{cnn.get('params',7441):,}", f"{cnn.get('latency_ms',0.66):.2f}"),
        ("LightMamba (ours)",      lm_res["acc"], lm_res["f1"], lm_res["auc"],
         f"{lm_res['params']:,}", f"{lm_latency:.2f}"),
        ("LightMamba (paper)",     0.9238, 0.9226, "—", "25,793", "1.61"),
        ("1D-CLANet (SOTA)",       0.9689, 0.9689, "—", "~1.4M",  "—"),
    ]

    for name, acc, f1, auc, params, lat in paper_refs:
        if isinstance(acc, float):
            print(f"  {name:<28} {acc:>7.4f}  {f1:>7.4f}  "
                  f"{str(auc) if isinstance(auc,str) else f'{auc:.4f}':>7}  "
                  f"{params:>8}  {lat:>10}")
        else:
            print(f"  {name:<28} {acc:>7}  {f1:>7}  {auc:>7}  "
                  f"{params:>8}  {lat:>10}")
    print("=" * 72)

    # Save results
    out = {"lightmamba": lm_res, "lightmamba_latency_ms": lm_latency,
           "cnn": cnn}
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results → {RESULTS_JSON}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Skip training, load saved model and evaluate only")
    args = parser.parse_args()

    tr, val, te = build_dataset_lm()

    if not args.eval:
        train_lightmamba(tr, val)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Run without --eval first.")
        return

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    lm_res     = evaluate_lightmamba(te)
    lm_latency = latency_benchmark()
    print_comparison(lm_res, lm_latency)


if __name__ == "__main__":
    main()
