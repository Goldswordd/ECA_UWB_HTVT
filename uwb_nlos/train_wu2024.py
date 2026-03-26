"""
Train and evaluate Wu et al. 2024 (Self-Attention TinyML) on eWINE.
Two-stage training:
  Stage 1 — pretrain MLP on 57-D (50-CIR + 7 channel features)
  Stage 2 — freeze encoder, retrain self-attention + classifier

Usage:
  python train_wu2024.py          # full pipeline: stage1 + stage2 + eval
  python train_wu2024.py --eval   # eval saved Stage-2 model only
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
from preprocess import _extract_cir_window, remove_outliers
from wu2024_model import PretrainedMLP, Wu2024Net

# ── Constants ─────────────────────────────────────────────────────────────────

# 57-D mixed dataset:  50-pt CIR + 7 raw channel characteristics
# Channel chars (from Fig. 3): Distance, MaxNoise, StdNoise, FP_AMP1,
#                               FP_AMP2, FP_AMP3, CIR_PWR
AUX_COLS = [1, 8, 6, 3, 4, 5, 7]   # 0-based column indices in eWINE CSV
N_AUX    = len(AUX_COLS)            # 7
N_CIR    = config.CIR_LEN           # 50
IN_DIM   = N_CIR + N_AUX           # 57

HP = dict(
    in_dim       = IN_DIM,   # 57
    # Stage-1
    s1_lr        = 1e-3,
    s1_epochs    = 350,
    s1_patience  = 30,
    s1_dropout   = 0.2,
    # Stage-2
    s2_lr        = 1e-3,
    s2_epochs    = 350,
    s2_patience  = 30,
    s2_dropout1  = 0.4,
    s2_dropout2  = 0.2,
    # shared
    batch_size   = 256,
    weight_decay = 1e-4,
    seed         = 42,
)

S1_MODEL_PATH  = config.MODEL_DIR / "wu2024_stage1.pt"
S2_MODEL_PATH  = config.MODEL_DIR / "wu2024_stage2.pt"
SCALER_PATH    = config.MODEL_DIR / "wu2024_scaler.pkl"
RESULTS_PATH   = config.LOG_DIR   / "wu2024_results.json"

torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ewine_wu2024(ewine_dir=None):
    """
    Load eWINE CSVs → 57-D mixed dataset.
    Returns:
      X57   : (N, 57) float32 — [50-pt CIR | 7 channel chars], UN-scaled
      labels: (N,) int8
    """
    ewine_dir = Path(ewine_dir or config.EWINE_DIR)
    csv_files = sorted(ewine_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {ewine_dir}")

    all_X, all_labels = [], []
    fp_col = config.EWINE_DIAG_COLS["FP_IDX"]
    rx_col = config.EWINE_DIAG_COLS["RXPACC"]

    for f in csv_files:
        print(f"  {f.name} ...", end=" ", flush=True)
        arr = pd.read_csv(f, header=0).values        # (N, 1031)
        labels  = arr[:, 0].astype(np.int8)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(np.float32)

        rows = []
        for i in range(len(arr)):
            fp_idx = int(arr[i, fp_col])
            rxpacc = float(arr[i, rx_col]) + 1e-9

            # 50-pt CIR window, RXPACC-normalised
            cir_win = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc  # (50,)

            # 7 raw channel characteristics (NOT normalised by RXPACC here;
            # StandardScaler will handle it later)
            aux = arr[i, AUX_COLS].astype(np.float32)                    # (7,)

            rows.append(np.concatenate([cir_win, aux]))                  # (57,)

        all_X.append(np.array(rows, dtype=np.float32))
        all_labels.append(labels)
        print(f"{len(arr)} rows")

    X57    = np.concatenate(all_X,      axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Total: {len(labels)} | LOS={int((labels==0).sum())} "
          f"NLOS={int((labels==1).sum())}")
    return X57, labels


def build_dataset_wu2024():
    print("=" * 60)
    print("Loading eWINE (57-D: 50-CIR + 7 channel features)...")

    X, labels = load_ewine_wu2024()

    # Outlier removal using remove_outliers on CIR part only
    # (pass dummy feat = X[:, :N_CIR] to use CIR, or just apply to full X)
    # Simplest: use the raw aux part for outlier z-score
    cir_part = X[:, :N_CIR]
    aux_part = X[:, N_CIR:]
    print("Removing outliers ...")
    cir_part, aux_part, labels = remove_outliers(cir_part, aux_part, labels)
    X = np.concatenate([cir_part, aux_part], axis=1)

    # 70 / 15 / 15 stratified split
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, labels, test_size=0.15, stratify=labels, random_state=HP["seed"])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=HP["seed"])

    print(f"Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_te)}")

    # StandardScaler on full 57-D (fit on train only)
    scaler = StandardScaler().fit(X_tr)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as fh: pickle.dump(scaler, fh)

    X_tr  = scaler.transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te  = scaler.transform(X_te).astype(np.float32)

    return X_tr, y_tr, X_val, y_val, X_te, y_te


def _make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ── Stage 1: Pretrain MLP ─────────────────────────────────────────────────────

def train_stage1(X_tr, y_tr, X_val, y_val):
    model = PretrainedMLP(HP["in_dim"], HP["s1_dropout"])
    print(f"\n[Stage 1] PretrainedMLP params (total): {model.count_params():,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["s1_lr"], weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP["s1_epochs"], eta_min=1e-5)

    tr_loader  = _make_loader(X_tr,  y_tr,  HP["batch_size"])
    val_loader = _make_loader(X_val, y_val, HP["batch_size"], shuffle=False)

    best_loss, patience_cnt = float("inf"), 0
    t0 = time.time()

    for epoch in range(1, HP["s1_epochs"] + 1):
        model.train()
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            nn.BCEWithLogitsLoss()(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += nn.BCEWithLogitsLoss()(model(xb), yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss, patience_cnt = val_loss, 0
            torch.save(model.state_dict(), S1_MODEL_PATH)
        else:
            patience_cnt += 1

        if epoch % 50 == 0 or patience_cnt == 0:
            print(f"  S1 Epoch {epoch:3d} | val_loss={val_loss:.4f} | "
                  f"best={best_loss:.4f} | patience={patience_cnt}")
        if patience_cnt >= HP["s1_patience"]:
            print(f"  Stage 1 early stop at epoch {epoch}")
            break

    print(f"Stage 1 done in {(time.time()-t0)/60:.1f} min")
    return model


# ── Stage 2: Self-attention retraining ────────────────────────────────────────

def train_stage2(X_tr, y_tr, X_val, y_val):
    # Build Stage-2 model and load frozen encoder from Stage 1
    s2_model = Wu2024Net(
        HP["in_dim"], HP["s2_dropout1"], HP["s2_dropout2"]
    )

    # Load encoder weights from Stage-1 checkpoint
    s1_state = torch.load(S1_MODEL_PATH, map_location="cpu")
    encoder_state = {k.replace("encoder.", ""): v
                     for k, v in s1_state.items() if k.startswith("encoder.")}
    s2_model.encoder.load_state_dict(encoder_state)
    s2_model.freeze_encoder()

    trainable = s2_model.count_params()
    total     = s2_model.count_params_total()
    print(f"\n[Stage 2] Wu2024Net — trainable: {trainable:,} / total: {total:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, s2_model.parameters()),
        lr=HP["s2_lr"], weight_decay=HP["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP["s2_epochs"], eta_min=1e-5)

    tr_loader  = _make_loader(X_tr,  y_tr,  HP["batch_size"])
    val_loader = _make_loader(X_val, y_val, HP["batch_size"], shuffle=False)

    best_loss, patience_cnt = float("inf"), 0
    t0 = time.time()

    for epoch in range(1, HP["s2_epochs"] + 1):
        s2_model.train()
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            loss = criterion(s2_model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s2_model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        s2_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(s2_model(xb), yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss, patience_cnt = val_loss, 0
            torch.save(s2_model.state_dict(), S2_MODEL_PATH)
        else:
            patience_cnt += 1

        if epoch % 50 == 0 or patience_cnt == 0:
            print(f"  S2 Epoch {epoch:3d} | val_loss={val_loss:.4f} | "
                  f"best={best_loss:.4f} | patience={patience_cnt}")
        if patience_cnt >= HP["s2_patience"]:
            print(f"  Stage 2 early stop at epoch {epoch}")
            break

    print(f"Stage 2 done in {(time.time()-t0)/60:.1f} min")
    return s2_model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_wu2024(X_te, y_te):
    model = Wu2024Net(HP["in_dim"], HP["s2_dropout1"], HP["s2_dropout2"])
    model.load_state_dict(torch.load(S2_MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_te))
    probs  = torch.sigmoid(logits).numpy()
    preds  = (probs > 0.5).astype(int)
    y_np   = y_te.astype(int)

    acc    = float((preds == y_np).mean())
    f1     = f1_score(y_np, preds)
    auc    = roc_auc_score(y_np, probs)
    recall = float((preds[y_np == 1] == 1).mean())
    prec   = float((y_np[preds == 1] == 1).mean()) if preds.sum() > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Wu2024  Acc={acc:.4f}  Prec={prec:.4f}  "
          f"Recall={recall:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"{'='*60}")

    # Confusion matrix
    cm = confusion_matrix(y_np, preds)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["LOS", "NLOS"]); ax.set_yticklabels(["LOS", "NLOS"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Wu2024 SA-TinyML")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.LOG_DIR / "wu2024_confusion.png", dpi=120)
    plt.close(fig)

    return {
        "acc": acc, "precision": prec, "recall": recall,
        "f1": f1, "auc": auc,
        "params_trainable": model.count_params(),
        "params_total":     model.count_params_total(),
    }


def latency_benchmark(n_warmup=50, n_runs=1000):
    model = Wu2024Net(HP["in_dim"], HP["s2_dropout1"], HP["s2_dropout2"])
    model.load_state_dict(torch.load(S2_MODEL_PATH, map_location="cpu"))
    model.eval()

    x_s = torch.zeros(1, HP["in_dim"])
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x_s)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x_s)
    ms = (time.perf_counter() - t0) / n_runs * 1000
    print(f"Wu2024 latency: {ms:.3f} ms / sample ({n_runs} runs)")
    return ms


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(wu_res: dict, wu_lat: float):
    def _load(fname):
        p = config.LOG_DIR / fname
        return json.load(open(p)) if p.exists() else {}

    lm  = _load("lm_results.json")
    si  = _load("si2023_results.json")
    jg  = _load("jiang2026_results.json")
    rf  = _load("eval_rf.json")

    def pct(d, k):
        v = d.get(k)
        return f"{v*100:.2f}%" if isinstance(v, float) else "—"

    H   = f"{'Model':<28} {'Acc':>8} {'F1':>8} {'AUC':>8} {'Params':>8} {'Latency':>9}"
    SEP = "-" * len(H)
    print("\n" + "=" * len(H))
    print(H); print(SEP)

    if rf:
        print(f"{'RF':<28} {pct(rf,'accuracy'):>8} {pct(rf,'f1_nlos'):>8} "
              f"{pct(rf,'roc_auc'):>8} {'—':>8} {'~152ms':>9}")

    cnn = lm.get("cnn", {})
    if cnn:
        print(f"{'CNN Dual-Branch':<28} {pct(cnn,'acc'):>8} {pct(cnn,'f1'):>8} "
              f"{pct(cnn,'auc'):>8} {cnn.get('params',7441):>8} "
              f"{cnn.get('latency_ms',0.66):.2f}ms".rjust(9))

    if si:
        print(f"{'Si2023 CNN+MLP':<28} {pct(si,'acc'):>8} {pct(si,'f1'):>8} "
              f"{pct(si,'auc'):>8} {si.get('params',1578):>8} "
              f"{si.get('latency_ms',0.37):.2f}ms".rjust(9))

    if jg:
        print(f"{'Jiang2026 CNN+Attn':<28} {pct(jg,'acc'):>8} {pct(jg,'f1'):>8} "
              f"{pct(jg,'auc'):>8} {jg.get('params',24145):>8} "
              f"{jg.get('latency_ms',1.22):.2f}ms".rjust(9))

    # Wu2024
    pt = wu_res.get("params_trainable", "—")
    print(f"{'Wu2024 SA-TinyML (ours)':<28} "
          f"{wu_res['acc']*100:.2f}%".rjust(9) +
          f" {wu_res['f1']*100:.2f}%".rjust(9) +
          f" {wu_res['auc']*100:.2f}%".rjust(9) +
          f" {pt:>8} {wu_lat:.2f}ms".rjust(9))

    # Wu2024 paper reference
    print(f"{'Wu2024 (paper,unquant)':<28} {'93.10%':>8} {'93.17%':>8} "
          f"{'—':>8} {'~4.8K':>8} {'<1ms†':>9}")

    lm_m = lm.get("lightmamba", {})
    if lm_m:
        print(f"{'LightMamba (ours)':<28} {pct(lm_m,'acc'):>8} {pct(lm_m,'f1'):>8} "
              f"{pct(lm_m,'auc'):>8} {lm_m.get('params',16225):>8} "
              f"{lm.get('lightmamba_latency_ms',1.71):.2f}ms".rjust(9))

    print(f"{'LightMamba (paper)':<28} {'92.38%':>8} {'92.26%':>8} "
          f"{'—':>8} {'25,793':>8} {'1.61ms':>9}")

    print("=" * len(H))
    print("† Wu2024 paper: Arduino Nano MCU, <1 ms after INT8 quantization")

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({**wu_res, "latency_ms": wu_lat}, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",  action="store_true",
                        help="Skip training, evaluate Stage-2 model only")
    parser.add_argument("--stage2", action="store_true",
                        help="Skip Stage 1, only run Stage 2 (Stage 1 must exist)")
    args = parser.parse_args()

    X_tr, y_tr, X_val, y_val, X_te, y_te = build_dataset_wu2024()

    if not args.eval:
        if not args.stage2:
            train_stage1(X_tr, y_tr, X_val, y_val)
        train_stage2(X_tr, y_tr, X_val, y_val)

    wu_res = evaluate_wu2024(X_te, y_te)
    wu_lat = latency_benchmark()
    print_comparison(wu_res, wu_lat)


if __name__ == "__main__":
    main()
