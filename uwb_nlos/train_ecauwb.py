"""
Train and evaluate ECA-UWB on the eWINE dataset.

Usage:
  python train_ecauwb.py              # full train + eval
  python train_ecauwb.py --ablation   # ablation study (4 variants)
  python train_ecauwb.py --eval       # eval saved model only
  python train_ecauwb.py --ablation --eval  # eval all ablation checkpoints
"""

import sys, time, json, pickle, argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from preprocess import _extract_cir_window, remove_outliers
from ecauwb_model import (
    ECAUWBNet,
    ECAUWBNet_NoECA,
    ECAUWBNet_ConcatFusion,
    ECAUWBNet_NoBranch,
)

# ── Constants ──────────────────────────────────────────────────────────────────

# 7 raw diagnostics: RANGE, MAX_NOISE, STDEV_NOISE, FP_AMP1-3, CIR_PWR
# (same as Wu2024; FP_IDX excluded — CIR window is already centred on it)
AUX_COLS = [1, 8, 6, 3, 4, 5, 7]   # 0-based column indices in eWINE CSV
N_AUX    = len(AUX_COLS)            # 7
N_CIR    = config.CIR_LEN           # 50
IN_DIM   = N_CIR + N_AUX           # 57

HP = dict(
    in_dim       = IN_DIM,
    cir_len      = N_CIR,
    n_aux        = N_AUX,
    # CNN
    ch1          = 16,
    ch2          = 16,
    k1           = 5,
    k2           = 3,
    eca_k        = 3,
    # Aux MLP
    aux_hid      = 32,
    feat_dim     = 16,
    # Classifier
    clf_hid      = 32,
    dropout      = 0.3,
    # Training
    lr           = 1e-3,
    weight_decay = 1e-4,
    batch_size   = 256,
    epochs       = 300,
    patience     = 30,
    seed         = 42,
    # Cost-sensitive: penalise NLOS false-negatives more (NLOS weight > 1)
    # 1.5 → NLOS loss contribution weighted 1.5× vs LOS, pushing recall up
    pos_weight   = 1.5,
)

MODEL_PATH   = config.MODEL_DIR / "ecauwb.pt"
SCALER_PATH  = config.MODEL_DIR / "ecauwb_scaler.pkl"
RESULTS_PATH = config.LOG_DIR   / "ecauwb_results.json"
PLOT_PATH    = config.LOG_DIR   / "ecauwb_training.png"

torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ewine_ecauwb(ewine_dir=None):
    """
    Load eWINE CSVs → 57-D mixed dataset (identical layout to Wu2024).
    Returns:
      X57   : (N, 57) float32 — [50-pt CIR | 7 channel diagnostics], UN-scaled
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
        arr    = pd.read_csv(f, header=0).values        # (N, 1031)
        labels = arr[:, 0].astype(np.int8)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(np.float32)

        rows = []
        for i in range(len(arr)):
            fp_idx = int(arr[i, fp_col])
            rxpacc = float(arr[i, rx_col]) + 1e-9

            # 50-pt CIR window, RXPACC-normalised
            cir_win = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc  # (50,)

            # 7 raw channel diagnostics (StandardScaler handles scaling later)
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


def build_dataset():
    print("=" * 60)
    print("Loading eWINE (57-D: 50-CIR + 7 channel diagnostics)...")
    X, labels = load_ewine_ecauwb()

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
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_model(model, tr_loader, val_loader, model_path,
                    lr=None, patience=None, epochs=None):
    lr      = lr      or HP["lr"]
    patience= patience or HP["patience"]
    epochs  = epochs  or HP["epochs"]

    model.to(DEVICE)
    pw = torch.tensor([HP.get("pos_weight", 1.0)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0; tr_preds = []; tr_trues = []
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yb)
            tr_preds.extend((logits.sigmoid() > 0.5).cpu().long().tolist())
            tr_trues.extend(yb.cpu().long().tolist())
        tr_loss /= len(tr_loader.dataset)
        train_f1 = f1_score(tr_trues, tr_preds, zero_division=0)

        # ── Validate ──
        model.eval()
        val_loss = 0.0; preds = []; trues = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                preds.extend((logits.sigmoid() > 0.5).cpu().long().tolist())
                trues.extend(yb.cpu().long().tolist())
        val_loss /= len(val_loader.dataset)
        val_acc   = accuracy_score(trues, preds)
        val_f1    = f1_score(trues, preds, zero_division=0)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {ep}")
                break

        if ep % 20 == 0 or ep == 1:
            print(f"  ep {ep:3d} | tr={tr_loss:.4f} val={val_loss:.4f} "
                  f"val_acc={val_acc:.4f} | best={best_val_loss:.4f}")

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed/60:.1f} min ({ep} epochs)")

    model.load_state_dict(best_state)
    torch.save(best_state, model_path)
    print(f"  Saved → {model_path}")
    return history


def find_optimal_threshold(model, val_loader, grid=None, min_los_recall=0.89):
    """
    Sweep classification threshold on the validation set.
    Selects the LOWEST threshold where LOS recall >= min_los_recall,
    which maximises NLOS recall (catching as many NLOS as possible) while
    keeping LOS recall at an acceptable level.
    Falls back to macro-recall optimum if constraint cannot be met.
    Returns the optimal threshold (float).
    """
    if grid is None:
        grid = np.arange(0.28, 0.68, 0.02)

    model.eval().to(DEVICE)
    all_probs, all_y = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            logits = model(Xb.to(DEVICE))
            all_probs.extend(logits.sigmoid().cpu().tolist())
            all_y.extend(yb.cpu().long().tolist())
    probs = np.array(all_probs)
    trues = np.array(all_y)

    print("\n  Threshold sweep on validation set:")
    print(f"  {'thr':>5}  {'nlos_r':>7}  {'los_r':>7}  {'macro_r':>8}  {'acc':>7}")

    rows = []
    best_macro_thr, best_macro = 0.5, 0.0
    for thr in grid:
        preds = (probs > thr).astype(int)
        cm = confusion_matrix(trues, preds)
        tn, fp, fn, tp = cm.ravel()
        nlos_r = tp / (tp + fn + 1e-9)
        los_r  = tn / (tn + fp + 1e-9)
        macro  = (nlos_r + los_r) / 2
        acc    = (tn + tp) / len(trues)
        rows.append((float(thr), nlos_r, los_r, macro, acc))
        print(f"  {thr:>5.2f}  {nlos_r*100:>6.2f}%  {los_r*100:>6.2f}%  "
              f"{macro*100:>7.2f}%  {acc*100:>6.2f}%")
        if macro > best_macro:
            best_macro = macro
            best_macro_thr = float(thr)

    # Pick lowest threshold where LOS recall >= min_los_recall (=> max NLOS recall)
    constrained = [(thr, nr, lr, m, a) for thr, nr, lr, m, a in rows if lr >= min_los_recall]
    if constrained:
        best_thr, best_nr, best_lr, best_m, best_a = constrained[0]  # lowest thr first
        print(f"\n  → Optimal threshold: {best_thr:.2f}  "
              f"(NLOS R={best_nr*100:.2f}%, LOS R={best_lr*100:.2f}%, "
              f"acc={best_a*100:.2f}%, constraint: LOS≥{min_los_recall*100:.0f}%)")
    else:
        best_thr = best_macro_thr
        print(f"\n  → Fallback to macro-optimal threshold: {best_thr:.2f}")
    return best_thr


def _evaluate(model, te_loader, label="Test", threshold=0.5):
    model.eval().to(DEVICE)
    all_logits, all_y = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            logits = model(Xb.to(DEVICE))
            all_logits.extend(logits.cpu().tolist())
            all_y.extend(yb.cpu().long().tolist())

    probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds  = (probs > threshold).astype(int)
    trues  = np.array(all_y)

    acc    = accuracy_score(trues, preds)
    f1     = f1_score(trues, preds, zero_division=0)
    auc    = roc_auc_score(trues, probs)
    cm     = confusion_matrix(trues, preds)
    tn, fp, fn, tp = cm.ravel()
    recall_nlos  = tp / (tp + fn + 1e-9)
    recall_los   = tn / (tn + fp + 1e-9)

    print(f"\n{'='*60}")
    print(f"{label} Results  (threshold={threshold:.2f})")
    print(f"{'='*60}")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  F1-score      : {f1*100:.2f}%")
    print(f"  AUC-ROC       : {auc*100:.2f}%")
    print(f"  NLOS Recall   : {recall_nlos*100:.2f}%")
    print(f"  LOS  Recall   : {recall_los*100:.2f}%")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")

    return {"acc": acc, "f1": f1, "auc": auc,
            "recall_nlos": recall_nlos, "recall_los": recall_los,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": threshold}


def _plot_history(history, title, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss"); ax1.legend()
    if "train_f1" in history:
        ax2.plot(history["train_f1"], label="Train F1")
        ax2.plot(history["val_f1"],   label="Val F1")
        ax2.legend()
    else:
        ax2.plot(history["val_f1"], label="Val F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
    ax2.set_title(f"{title} — F1 Score")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


# ── Latency measurement ────────────────────────────────────────────────────────

def measure_latency(model, n_runs: int = 1000):
    model.eval().to("cpu")
    dummy = torch.zeros(1, HP["in_dim"])
    # warm-up
    for _ in range(50):
        model(dummy)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(dummy)
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000
    return elapsed_ms


# ── Main training pipeline ────────────────────────────────────────────────────

def run_main(args):
    X_tr, y_tr, X_val, y_val, X_te, y_te = build_dataset()
    tr_loader  = _make_loader(X_tr,  y_tr,  HP["batch_size"], shuffle=True)
    val_loader = _make_loader(X_val, y_val, HP["batch_size"], shuffle=False)
    te_loader  = _make_loader(X_te,  y_te,  HP["batch_size"], shuffle=False)

    model = ECAUWBNet(
        cir_len  = HP["cir_len"],
        n_aux    = HP["n_aux"],
        ch1      = HP["ch1"],
        ch2      = HP["ch2"],
        k1       = HP["k1"],
        k2       = HP["k2"],
        eca_k    = HP["eca_k"],
        aux_hid  = HP["aux_hid"],
        feat_dim = HP["feat_dim"],
        clf_hid  = HP["clf_hid"],
        dropout  = HP["dropout"],
    )

    n_params = model.count_params()
    branch   = model.branch_params()
    print(f"\nECA-UWB model: {n_params} trainable params")
    for k, v in branch.items():
        print(f"  {k:15s}: {v} params")

    if args.eval and MODEL_PATH.exists():
        print(f"\nLoading saved model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        print(f"\n{'='*60}\nTraining ECA-UWB\n{'='*60}")
        hist = train_one_model(model, tr_loader, val_loader, MODEL_PATH)
        _plot_history(hist, "ECA-UWB", PLOT_PATH)
        HIST_JSON = config.LOG_DIR / "ecauwb_train_history.json"
        with open(HIST_JSON, "w") as fh:
            json.dump(hist, fh, indent=2)
        print(f"  History saved → {HIST_JSON}")

    # Evaluate at default threshold=0.5
    results_05 = _evaluate(model, te_loader, label="Test @thr=0.50", threshold=0.5)

    # Find optimal threshold on validation set, then evaluate on test
    # Constraint: LOS recall ≥ 91% on val → yields balanced operating point
    opt_thr = find_optimal_threshold(model, val_loader, min_los_recall=0.91)
    results  = _evaluate(model, te_loader, label="Test @optimal thr", threshold=opt_thr)
    results["results_at_0.5"] = results_05
    results["n_params"] = n_params
    results["branch_params"] = branch

    lat_ms = measure_latency(model)
    results["latency_ms"] = round(lat_ms, 3)
    print(f"\n  Latency (CPU, single sample): {lat_ms:.3f} ms")

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")
    return results, X_tr, y_tr, X_val, y_val, X_te, y_te, tr_loader, val_loader, te_loader


# ── Ablation study ────────────────────────────────────────────────────────────

ABLATION_VARIANTS = {
    "full":        (ECAUWBNet,             "ecauwb_full.pt"),
    "no_eca":      (ECAUWBNet_NoECA,       "ecauwb_no_eca.pt"),
    "concat_fuse": (ECAUWBNet_ConcatFusion,"ecauwb_concat.pt"),
    "no_aux":      (ECAUWBNet_NoBranch,    "ecauwb_no_aux.pt"),
}

def run_ablation(args, X_tr=None, y_tr=None, X_val=None, y_val=None,
                 X_te=None,  y_te=None,  tr_loader=None,
                 val_loader=None, te_loader=None):

    if X_tr is None:
        X_tr, y_tr, X_val, y_val, X_te, y_te = build_dataset()
        tr_loader  = _make_loader(X_tr,  y_tr,  HP["batch_size"], shuffle=True)
        val_loader = _make_loader(X_val, y_val, HP["batch_size"], shuffle=False)
        te_loader  = _make_loader(X_te,  y_te,  HP["batch_size"], shuffle=False)

    model_hp = dict(
        cir_len=HP["cir_len"], n_aux=HP["n_aux"], ch1=HP["ch1"], ch2=HP["ch2"],
        k1=HP["k1"], k2=HP["k2"], eca_k=HP["eca_k"], aux_hid=HP["aux_hid"],
        feat_dim=HP["feat_dim"], clf_hid=HP["clf_hid"], dropout=HP["dropout"],
    )

    ablation_results = {}
    print(f"\n{'='*60}\nABLATION STUDY\n{'='*60}")

    for name, (ModelClass, fname) in ABLATION_VARIANTS.items():
        mpath = config.MODEL_DIR / fname
        print(f"\n── Variant: {name} ──")
        model = ModelClass(**model_hp)
        print(f"  Params: {model.count_params()}")

        if args.eval and mpath.exists():
            model.load_state_dict(torch.load(mpath, map_location="cpu"))
        else:
            train_one_model(model, tr_loader, val_loader, mpath)

        res = _evaluate(model, te_loader, label=name, threshold=0.5)
        res["n_params"] = model.count_params()
        ablation_results[name] = res

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Variant':<18} {'Acc%':>7} {'F1%':>7} {'AUC%':>7} {'Params':>8}")
    print("-" * 60)
    for name, r in ablation_results.items():
        print(f"{name:<18} {r['acc']*100:>7.2f} {r['f1']*100:>7.2f} "
              f"{r['auc']*100:>7.2f} {r['n_params']:>8d}")

    abl_path = config.LOG_DIR / "ecauwb_ablation.json"
    with open(abl_path, "w") as fh:
        json.dump(ablation_results, fh, indent=2)
    print(f"\nAblation results → {abl_path}")
    return ablation_results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ECA-UWB")
    parser.add_argument("--eval",     action="store_true",
                        help="Skip training, eval saved model only")
    parser.add_argument("--ablation", action="store_true",
                        help="Run 4-variant ablation study")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"ECA-UWB training pipeline — eWINE dataset")

    results, X_tr, y_tr, X_val, y_val, X_te, y_te, tr_loader, val_loader, te_loader = \
        run_main(args)

    if args.ablation:
        run_ablation(args, X_tr, y_tr, X_val, y_val, X_te, y_te,
                     tr_loader, val_loader, te_loader)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"ECA-UWB FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Accuracy  : {results['acc']*100:.2f}%")
    print(f"  AUC-ROC   : {results['auc']*100:.2f}%")
    print(f"  F1-score  : {results['f1']*100:.2f}%")
    print(f"  Params    : {results['n_params']}")
    print(f"  Latency   : {results['latency_ms']:.3f} ms (CPU, single sample)")


if __name__ == "__main__":
    main()
