"""
Train and evaluate Si et al. 2023 (CNN+MLP) on the eWINE dataset,
then print a full comparison table with CNN Dual-Branch and LightMamba.

Usage:
  python train_si2023.py          # train + eval
  python train_si2023.py --eval   # eval only (load saved model)
"""

import sys, time, json, pickle, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from preprocess import _extract_cir_window, _compute_features, remove_outliers
from si2023_model import Si2023Net

# ── Hyperparameters ───────────────────────────────────────────────────────────

HP = dict(
    cir_len      = 1016,
    n_manual     = 11,
    batch_size   = 256,
    lr           = 1e-3,
    weight_decay = 1e-4,
    patience     = 30,
    max_epochs   = 200,
    seed         = 42,
)

MODEL_PATH       = config.MODEL_DIR / "si2023.pt"
FEAT_SCALER_PATH = config.MODEL_DIR / "si2023_scaler_feat.pkl"
CIR_SCALER_PATH  = config.MODEL_DIR / "si2023_scaler_cir.pkl"
RESULTS_PATH     = config.LOG_DIR  / "si2023_results.json"

torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ewine_si2023(ewine_dir=None):
    """
    Load all eWINE CSVs.

    Returns
    -------
    cir_full : (N, 1016) float32 — full CIR normalized by RXPACC
    feat_11  : (N,  11)  float32 — 11 engineered features (unscaled)
    labels   : (N,)      int8
    """
    ewine_dir = Path(ewine_dir or config.EWINE_DIR)
    csv_files = sorted(ewine_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ewine_dir}")

    all_cir, all_feat, all_labels = [], [], []
    diag_idx = config.EWINE_DIAG_COLS

    for f in csv_files:
        print(f"  {f.name} ...", end=" ", flush=True)
        arr     = pd.read_csv(f, header=0).values        # (N, 1031)
        labels  = arr[:, 0].astype(np.int8)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(np.float32)  # (N, 1016)

        cirs, feats = [], []
        for i in range(len(arr)):
            row    = arr[i]
            fp_idx = int(row[diag_idx["FP_IDX"]])
            rxpacc = float(row[diag_idx["RXPACC"]]) + 1e-9

            # Full CIR normalized by RXPACC  →  CNN branch input
            cirs.append(cir_raw[i] / rxpacc)

            # 50-pt window around first path  →  11 engineered features
            cir_win = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc
            diag    = {k: float(row[v]) for k, v in diag_idx.items()}
            feats.append(_compute_features(diag, cir_win))

        all_cir.append(np.array(cirs,  dtype=np.float32))
        all_feat.append(np.array(feats, dtype=np.float32))
        all_labels.append(labels)
        print(f"{len(arr)} rows  "
              f"LOS={int((labels==0).sum())}  NLOS={int((labels==1).sum())}")

    cir_full = np.concatenate(all_cir,    axis=0)
    feat_11  = np.concatenate(all_feat,   axis=0)
    labels   = np.concatenate(all_labels, axis=0)
    print(f"Total: {len(labels)} | LOS={int((labels==0).sum())} "
          f"NLOS={int((labels==1).sum())}\n")
    return cir_full, feat_11, labels


def build_dataset_si2023():
    print("=" * 60)
    print("Loading eWINE (full 1016-pt CIR + 11 features) …")
    cir, feat, labels = load_ewine_si2023()

    print("Removing outliers …")
    cir, feat, labels = remove_outliers(cir, feat, labels)

    # Stratified 70 / 15 / 15 split
    cir_tv, cir_te, feat_tv, feat_te, y_tv, y_te = train_test_split(
        cir, feat, labels,
        test_size=0.15, stratify=labels, random_state=HP["seed"])
    cir_tr, cir_val, feat_tr, feat_val, y_tr, y_val = train_test_split(
        cir_tv, feat_tv, y_tv,
        test_size=0.15 / 0.85, stratify=y_tv, random_state=HP["seed"])

    print(f"Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_te)}\n")

    # Fit scalers on train only
    feat_scaler = StandardScaler().fit(feat_tr)
    cir_scaler  = StandardScaler().fit(cir_tr)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEAT_SCALER_PATH, "wb") as fh: pickle.dump(feat_scaler, fh)
    with open(CIR_SCALER_PATH,  "wb") as fh: pickle.dump(cir_scaler,  fh)

    def scale(c, f):
        return (cir_scaler.transform(c).astype(np.float32),
                feat_scaler.transform(f).astype(np.float32))

    cir_tr,  feat_tr  = scale(cir_tr,  feat_tr)
    cir_val, feat_val = scale(cir_val, feat_val)
    cir_te,  feat_te  = scale(cir_te,  feat_te)

    return (cir_tr, feat_tr, y_tr,
            cir_val, feat_val, y_val,
            cir_te,  feat_te,  y_te)


def _make_loader(cir, feat, labels, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(cir),
        torch.from_numpy(feat),
        torch.from_numpy(labels.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── Training ──────────────────────────────────────────────────────────────────

def train_si2023(data):
    cir_tr, feat_tr, y_tr, cir_val, feat_val, y_val, *_ = data

    model = Si2023Net(HP["cir_len"], HP["n_manual"])
    print(f"Si2023Net parameters: {model.count_params():,}\n")

    pos_w     = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP["max_epochs"], eta_min=1e-5)

    tr_loader  = _make_loader(cir_tr, feat_tr, y_tr, HP["batch_size"])
    val_loader = _make_loader(cir_val, feat_val, y_val, HP["batch_size"],
                              shuffle=False)

    best_val  = float("inf")
    patience  = 0
    t0 = time.time()

    for epoch in range(1, HP["max_epochs"] + 1):
        model.train()
        for cir_b, feat_b, y_b in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(cir_b, feat_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cir_b, feat_b, y_b in val_loader:
                val_loss += criterion(model(cir_b, feat_b), y_b).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1

        if epoch % 20 == 0 or patience == 0:
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | "
                  f"best={best_val:.4f} | patience={patience}")

        if patience >= HP["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")
    print(f"Model saved → {MODEL_PATH}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_si2023(data):
    *_, cir_te, feat_te, y_te = data

    model = Si2023Net(HP["cir_len"], HP["n_manual"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(cir_te), torch.from_numpy(feat_te))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    y_np  = y_te.astype(int)

    acc = float((preds == y_np).mean())
    f1  = float(f1_score(y_np, preds))
    auc = float(roc_auc_score(y_np, probs))

    print(f"\n{'='*50}")
    print(f"Si2023  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"{'='*50}\n")

    # Confusion matrix plot
    cm  = confusion_matrix(y_np, preds)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["LOS", "NLOS"])
    ax.set_yticklabels(["LOS", "NLOS"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Si2023 CNN+MLP — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.LOG_DIR / "si2023_confusion.png", dpi=120)
    plt.close(fig)

    return {"acc": acc, "f1": f1, "auc": auc, "params": model.count_params()}


def latency_benchmark(n_warmup=50, n_runs=1000):
    model = Si2023Net(HP["cir_len"], HP["n_manual"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    cir_s  = torch.zeros(1, HP["cir_len"])
    feat_s = torch.zeros(1, HP["n_manual"])

    with torch.no_grad():
        for _ in range(n_warmup):
            model(cir_s, feat_s)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(cir_s, feat_s)
    ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"Si2023 latency: {ms:.3f} ms/sample ({n_runs} runs)")
    return ms


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(si_res, si_lat):
    lm_data, rf_data = {}, {}

    lm_path = config.LOG_DIR / "lm_results.json"
    if lm_path.exists():
        with open(lm_path) as f:
            lm_data = json.load(f)

    rf_path = config.LOG_DIR / "eval_rf.json"
    if rf_path.exists():
        with open(rf_path) as f:
            rf_data = json.load(f)

    def pct(d, k, fallback="N/A"):
        v = d.get(k, None)
        return f"{v*100:.2f}%" if isinstance(v, float) else fallback

    W = 80
    print("\n" + "=" * W)
    print(f"{'Model':<24} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Params':>8} {'Latency':>9}")
    print("-" * W)

    if rf_data:
        print(f"{'RF':<24} "
              f"{pct(rf_data,'accuracy'):>8} "
              f"{pct(rf_data,'f1_nlos'):>8} "
              f"{pct(rf_data,'roc_auc'):>8} "
              f"{'—':>8} {'~152 ms':>9}")

    cnn = lm_data.get("cnn", {})
    if cnn:
        lat_cnn = cnn.get("latency_ms", 0.66)
        print(f"{'CNN Dual-Branch':<24} "
              f"{pct(cnn,'acc'):>8} "
              f"{pct(cnn,'f1'):>8} "
              f"{pct(cnn,'auc'):>8} "
              f"{cnn.get('params','—'):>8} "
              f"{lat_cnn:.2f} ms".rjust(9))

    print(f"{'Si2023 CNN+MLP':<24} "
          f"{si_res['acc']*100:>7.2f}% "
          f"{si_res['f1']*100:>7.2f}% "
          f"{si_res['auc']*100:>7.2f}% "
          f"{si_res['params']:>8} "
          f"{si_lat:.2f} ms".rjust(9))

    lm = lm_data.get("lightmamba", {})
    if lm:
        lat_lm = lm_data.get("lightmamba_latency_ms", 1.71)
        print(f"{'LightMamba (ours)':<24} "
              f"{pct(lm,'acc'):>8} "
              f"{pct(lm,'f1'):>8} "
              f"{pct(lm,'auc'):>8} "
              f"{lm.get('params','—'):>8} "
              f"{lat_lm:.2f} ms".rjust(9))

    print(f"{'LightMamba (paper)':<24} "
          f"{'92.38%':>8} {'92.26%':>8} {'—':>8} {'25,793':>8} {'1.61 ms':>9}")
    print("=" * W)

    # Save results
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({**si_res, "latency_ms": si_lat}, f, indent=2)
    print(f"\nSi2023 results saved → {RESULTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Skip training, load saved model and evaluate")
    args = parser.parse_args()

    data = build_dataset_si2023()

    if not args.eval:
        train_si2023(data)

    si_res = evaluate_si2023(data)
    si_lat = latency_benchmark()
    print_comparison(si_res, si_lat)


if __name__ == "__main__":
    main()
