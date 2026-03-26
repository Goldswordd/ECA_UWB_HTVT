"""
Train and evaluate Jiang et al. 2026 (multi-scale CNN + self-attention + MLP)
on the eWINE dataset and compare with all baseline models.

Usage:
  python train_jiang2026.py          # train + eval + compare
  python train_jiang2026.py --eval   # eval only (load saved model)
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
from preprocess import remove_outliers
from jiang2026_model import Jiang2026Net, FocalLoss

# ── Constants ─────────────────────────────────────────────────────────────────

# CIR truncation parameters (optimized in paper, Sec. II-B)
CIR_X = 6    # look-back window size
CIR_Y = 63   # total window length → multipath window = 63 - 6 = 57

# 8 auxiliary feature columns (sorted by RF importance, Fig. 5 of paper):
#   RXPACC, RANGE, CIR_PWR, FP_AMP3, MAX_NOISE, FP_AMP2, FP_AMP1, STDEV_NOISE
AUX_COLS = [9, 1, 7, 5, 8, 4, 3, 6]    # 0-based column indices in eWINE CSV
N_AUX    = len(AUX_COLS)               # 8

HP = dict(
    cir_len      = CIR_Y,   # 63
    n_aux        = N_AUX,   # 8
    d_model      = 32,
    n_heads      = 8,
    dropout      = 0.4,
    dropout2     = 0.2,
    batch_size   = 256,
    lr           = 2e-3,
    weight_decay = 1e-4,
    patience     = 30,
    max_epochs   = 200,
    focal_gamma  = 2.0,
    focal_alpha  = 0.8,
    seed         = 42,
)

MODEL_PATH     = config.MODEL_DIR / "jiang2026.pt"
AUX_SCALER_PATH = config.MODEL_DIR / "jiang2026_scaler_aux.pkl"
CIR_SCALER_PATH = config.MODEL_DIR / "jiang2026_scaler_cir.pkl"
RESULTS_PATH   = config.LOG_DIR / "jiang2026_results.json"

torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])


# ── CIR window extraction ────────────────────────────────────────────────────

def _extract_window(cir_full: np.ndarray, fp_idx: int,
                    X: int = CIR_X, Y: int = CIR_Y) -> np.ndarray:
    """
    Extract Y-point CIR window with X samples look-back from FP_IDX.
      window = CIR[fp_idx - X : fp_idx - X + Y]
    Handles boundary by clamping and zero-padding if needed.
    """
    start = fp_idx - X
    end   = start + Y
    # Clamp to valid range
    if start < 0:
        start, end = 0, Y
    if end > len(cir_full):
        end   = len(cir_full)
        start = max(0, end - Y)
    window = cir_full[start:end]
    if len(window) < Y:
        window = np.pad(window, (0, Y - len(window)))
    return window.astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ewine_jiang(ewine_dir=None):
    """
    Load eWINE CSVs.
    Returns:
      cir_63   : (N, 63)  float32 — truncated CIR (X=6, Y=63), RXPACC-normalized
      aux_8    : (N, 8)   float32 — 8 physical diagnostic params (raw values)
      labels   : (N,)     int8
    """
    ewine_dir = Path(ewine_dir or config.EWINE_DIR)
    csv_files = sorted(ewine_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {ewine_dir}")

    all_cir, all_aux, all_labels = [], [], []

    for f in csv_files:
        print(f"  {f.name} ...", end=" ", flush=True)
        arr = pd.read_csv(f, header=0).values          # (N, 1031)

        labels  = arr[:, 0].astype(np.int8)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(np.float32)   # (N, 1016)

        cirs, auxs = [], []
        fp_col = config.EWINE_DIAG_COLS["FP_IDX"]
        rx_col = config.EWINE_DIAG_COLS["RXPACC"]

        for i in range(len(arr)):
            fp_idx = int(arr[i, fp_col])
            rxpacc = float(arr[i, rx_col]) + 1e-9

            # 63-pt window, normalized by RXPACC
            win = _extract_window(cir_raw[i], fp_idx) / rxpacc
            cirs.append(win)

            # 8 aux features (raw, unscaled here)
            auxs.append(arr[i, AUX_COLS].astype(np.float32))

        all_cir.append(np.array(cirs,   dtype=np.float32))
        all_aux.append(np.array(auxs,   dtype=np.float32))
        all_labels.append(labels)
        print(f"{len(arr)} rows")

    cir_63 = np.concatenate(all_cir,    axis=0)
    aux_8  = np.concatenate(all_aux,    axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Total: {len(labels)} | LOS={int((labels==0).sum())} "
          f"NLOS={int((labels==1).sum())}")
    return cir_63, aux_8, labels


def build_dataset_jiang():
    print("=" * 60)
    print("Loading eWINE (63-pt CIR + 8 aux features)...")
    cir, aux, labels = load_ewine_jiang()

    # Outlier removal (z-score on aux features, class-conditional)
    print("Removing outliers ...")
    cir, aux, labels = remove_outliers(cir, aux, labels)

    # 70 / 15 / 15 stratified split
    cir_tv, cir_te, aux_tv, aux_te, y_tv, y_te = train_test_split(
        cir, aux, labels,
        test_size=0.15, stratify=labels, random_state=HP["seed"])
    cir_tr, cir_val, aux_tr, aux_val, y_tr, y_val = train_test_split(
        cir_tv, aux_tv, y_tv,
        test_size=0.15 / 0.85, stratify=y_tv, random_state=HP["seed"])

    print(f"Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_te)}")

    # Fit scalers on train split only
    cir_scaler = StandardScaler().fit(cir_tr)
    aux_scaler = StandardScaler().fit(aux_tr)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CIR_SCALER_PATH, "wb") as fh: pickle.dump(cir_scaler, fh)
    with open(AUX_SCALER_PATH, "wb") as fh: pickle.dump(aux_scaler, fh)

    def scale(c, a):
        return (cir_scaler.transform(c).astype(np.float32),
                aux_scaler.transform(a).astype(np.float32))

    cir_tr,  aux_tr  = scale(cir_tr,  aux_tr)
    cir_val, aux_val = scale(cir_val, aux_val)
    cir_te,  aux_te  = scale(cir_te,  aux_te)

    return (cir_tr, aux_tr, y_tr,
            cir_val, aux_val, y_val,
            cir_te,  aux_te,  y_te)


def _make_loader(cir, aux, labels, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(cir),
        torch.from_numpy(aux),
        torch.from_numpy(labels.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── Training ──────────────────────────────────────────────────────────────────

def train_jiang(data):
    cir_tr, aux_tr, y_tr, cir_val, aux_val, y_val, *_ = data

    model = Jiang2026Net(
        cir_len=HP["cir_len"], n_aux=HP["n_aux"],
        d_model=HP["d_model"], n_heads=HP["n_heads"],
        dropout=HP["dropout"], dropout2=HP["dropout2"],
    )
    print(f"\nJiang2026Net parameters: {model.count_params():,}")

    criterion = FocalLoss(gamma=HP["focal_gamma"], alpha=HP["focal_alpha"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP["max_epochs"], eta_min=1e-5
    )

    tr_loader  = _make_loader(cir_tr,  aux_tr,  y_tr,  HP["batch_size"])
    val_loader = _make_loader(cir_val, aux_val, y_val, HP["batch_size"], shuffle=False)

    best_val_loss = float("inf")
    patience_cnt  = 0
    t0 = time.time()

    for epoch in range(1, HP["max_epochs"] + 1):
        # ── train ──
        model.train()
        for cir_b, aux_b, y_b in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(cir_b, aux_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cir_b, aux_b, y_b in val_loader:
                val_loss += criterion(model(cir_b, aux_b), y_b).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_cnt += 1

        if epoch % 20 == 0 or patience_cnt == 0:
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | "
                  f"best={best_val_loss:.4f} | patience={patience_cnt}")

        if patience_cnt >= HP["patience"]:
            print(f"  Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"Training done in {elapsed / 60:.1f} min")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_jiang(data):
    *_, cir_te, aux_te, y_te = data

    model = Jiang2026Net(
        cir_len=HP["cir_len"], n_aux=HP["n_aux"],
        d_model=HP["d_model"], n_heads=HP["n_heads"],
        dropout=HP["dropout"], dropout2=HP["dropout2"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(cir_te), torch.from_numpy(aux_te))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    y_np  = y_te.astype(int)

    acc    = float((preds == y_np).mean())
    f1     = f1_score(y_np, preds)
    auc    = roc_auc_score(y_np, probs)
    recall = float((preds[y_np == 1] == 1).mean())

    print(f"\n{'='*55}")
    print(f"Jiang2026  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
          f"NLOS-Recall={recall:.4f}")
    print(f"{'='*55}")

    # Confusion matrix
    cm = confusion_matrix(y_np, preds)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["LOS", "NLOS"]); ax.set_yticklabels(["LOS", "NLOS"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Jiang2026 CNN+Attention+MLP")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.LOG_DIR / "jiang2026_confusion.png", dpi=120)
    plt.close(fig)

    return {
        "acc":    acc,
        "f1":     f1,
        "auc":    auc,
        "recall": recall,
        "params": model.count_params(),
    }


def latency_benchmark(n_warmup=50, n_runs=1000):
    model = Jiang2026Net(
        cir_len=HP["cir_len"], n_aux=HP["n_aux"],
        d_model=HP["d_model"], n_heads=HP["n_heads"],
        dropout=HP["dropout"], dropout2=HP["dropout2"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    cir_s = torch.zeros(1, HP["cir_len"])
    aux_s = torch.zeros(1, HP["n_aux"])

    with torch.no_grad():
        for _ in range(n_warmup):
            model(cir_s, aux_s)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(cir_s, aux_s)
    ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"Jiang2026 latency: {ms:.3f} ms / sample ({n_runs} runs)")
    return ms


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(jg_res: dict, jg_lat: float):
    # Load previous results
    def _load(path):
        p = config.LOG_DIR / path
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    lm   = _load("lm_results.json")
    si   = _load("si2023_results.json")
    rf_p = config.LOG_DIR / "eval_rf.json"
    rf   = {}
    if rf_p.exists():
        with open(rf_p) as f:
            rf = json.load(f)

    def pct(d, k, default=None):
        v = d.get(k, default)
        return f"{v*100:.2f}%" if isinstance(v, float) else "—"

    H = f"{'Model':<26} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Params':>8} {'Latency':>9}"
    SEP = "-" * len(H)
    print("\n" + "=" * len(H))
    print(H)
    print(SEP)

    # RF
    if rf:
        print(f"{'RF':<26} {pct(rf,'accuracy'):>8} {pct(rf,'f1_nlos'):>8} "
              f"{pct(rf,'roc_auc'):>8} {'—':>8} {'~152ms':>9}")

    # CNN Dual-Branch
    cnn = lm.get("cnn", {})
    if cnn:
        lat = cnn.get("latency_ms", 0.66)
        print(f"{'CNN Dual-Branch':<26} {pct(cnn,'acc'):>8} {pct(cnn,'f1'):>8} "
              f"{pct(cnn,'auc'):>8} {cnn.get('params',7441):>8} {lat:.2f}ms".rjust(9))

    # Si2023
    if si:
        print(f"{'Si2023 CNN+MLP':<26} {pct(si,'acc'):>8} {pct(si,'f1'):>8} "
              f"{pct(si,'auc'):>8} {si.get('params',1578):>8} "
              f"{si.get('latency_ms',0.37):.2f}ms".rjust(9))

    # Jiang2026 (ours)
    print(f"{'Jiang2026 (ours)':<26} "
          f"{jg_res['acc']*100:.2f}%".rjust(9) +
          f" {jg_res['f1']*100:.2f}%".rjust(9) +
          f" {jg_res['auc']*100:.2f}%".rjust(9) +
          f" {jg_res['params']:>8} {jg_lat:.2f}ms".rjust(9))

    # Jiang2026 paper reference (STM32 quantized, recall metric)
    print(f"{'Jiang2026 (paper,quant)':<26} {'93.10%':>8} {'93.20%':>8} "
          f"{'—':>8} {'~24K':>8} {'29ms*':>9}")

    # LightMamba ours
    lm_m = lm.get("lightmamba", {})
    if lm_m:
        lm_lat = lm.get("lightmamba_latency_ms", 1.71)
        print(f"{'LightMamba (ours)':<26} {pct(lm_m,'acc'):>8} {pct(lm_m,'f1'):>8} "
              f"{pct(lm_m,'auc'):>8} {lm_m.get('params',16225):>8} "
              f"{lm_lat:.2f}ms".rjust(9))

    # LightMamba paper
    print(f"{'LightMamba (paper)':<26} {'92.38%':>8} {'92.26%':>8} "
          f"{'—':>8} {'25,793':>8} {'1.61ms':>9}")

    print("=" * len(H))
    print("* Jiang2026 paper latency on STM32 MCU (INT8 quantized); "
          "our result is on CPU (float32)")

    # Save results
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    res = {**jg_res, "latency_ms": jg_lat}
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Skip training, evaluate saved model only")
    args = parser.parse_args()

    data = build_dataset_jiang()

    if not args.eval:
        train_jiang(data)

    jg_res = evaluate_jiang(data)
    jg_lat = latency_benchmark()
    print_comparison(jg_res, jg_lat)


if __name__ == "__main__":
    main()
