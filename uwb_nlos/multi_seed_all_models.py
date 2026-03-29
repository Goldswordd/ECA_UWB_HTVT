"""
Multi-seed validation for ALL reproduced models under standard BCE (w+=1.0, τ=0.50).
Seeds: {0, 1, 2, 3, 4}   Models: CNN, CIR-CNN+MLP, SA-TinyML, ECA-UWB
Output: logs/multi_seed_all_bce10.json

Run:  python multi_seed_all_models.py
      python multi_seed_all_models.py --models ecauwb wu2024   # subset
"""

import sys, time, json, argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                              confusion_matrix)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from preprocess import (load_ewine, remove_outliers, split_data,
                        _extract_cir_window, _compute_features)
from ecauwb_model import ECAUWBNet
from wu2024_model import Wu2024Net
from model import NLOSClassifier
from si2023_model import Si2023Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [0, 1, 2, 3, 4]

# ── Shared common HPs (standard BCE: pos_weight=1.0) ─────────────────────────
COMMON = dict(pos_weight=1.0, lr=1e-3, weight_decay=1e-4,
              batch_size=256, epochs=300, patience=30)

HP_ECA = dict(**COMMON, cir_len=50, n_aux=7, ch1=16, ch2=16, k1=5, k2=3,
              eca_k=3, aux_hid=32, feat_dim=16, clf_hid=32, dropout=0.3)
HP_WU  = dict(**COMMON)
HP_CNN = dict(**COMMON)
HP_SI  = {**COMMON, "epochs": 200}   # Si2023 used 200 epochs

AUX_COLS = [1, 8, 6, 3, 4, 5, 7]   # 7 raw diagnostics for ECA-UWB / Wu2024


# ── Data loading ──────────────────────────────────────────────────────────────

def load_57d():
    """57-D: 50-pt CIR + 7 raw diagnostics. For ECA-UWB and SA-TinyML."""
    ewine_dir = Path(config.EWINE_DIR)
    fp_col = config.EWINE_DIAG_COLS["FP_IDX"]
    rx_col = config.EWINE_DIAG_COLS["RXPACC"]
    all_X, all_y = [], []
    for f in sorted(ewine_dir.glob("*.csv")):
        arr = pd.read_csv(f, header=0).values
        labels = arr[:, 0].astype(np.int8)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(np.float32)
        rows = []
        for i in range(len(arr)):
            fp_idx = int(arr[i, fp_col])
            rxpacc = float(arr[i, rx_col]) + 1e-9
            cir_w  = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc
            aux    = arr[i, AUX_COLS].astype(np.float32)
            rows.append(np.concatenate([cir_w, aux]))
        all_X.append(np.array(rows, np.float32)); all_y.append(labels)
    X = np.concatenate(all_X); y = np.concatenate(all_y)
    cir, aux, y = remove_outliers(X[:, :50], X[:, 50:], y)
    return np.concatenate([cir, aux], 1), y


def load_si2023_data():
    """1016-pt full CIR + 11 engineered features. For CIR-CNN+MLP."""
    ewine_dir = Path(config.EWINE_DIR)
    diag_idx  = config.EWINE_DIAG_COLS
    all_cir, all_feat, all_y = [], [], []
    for f in sorted(ewine_dir.glob("*.csv")):
        arr = pd.read_csv(f, header=0).values
        labels = arr[:, 0].astype(np.int8)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(np.float32)
        cirs, feats = [], []
        for i in range(len(arr)):
            fp_idx = int(arr[i, diag_idx["FP_IDX"]])
            rxpacc = float(arr[i, diag_idx["RXPACC"]]) + 1e-9
            cirs.append(cir_raw[i] / rxpacc)
            cir_win = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc
            diag    = {k: float(arr[i, v]) for k, v in diag_idx.items()}
            feats.append(_compute_features(diag, cir_win))
        all_cir.append(np.array(cirs,  np.float32))
        all_feat.append(np.array(feats, np.float32))
        all_y.append(labels)
    cir  = np.concatenate(all_cir)
    feat = np.concatenate(all_feat)
    y    = np.concatenate(all_y)
    cir, feat, y = remove_outliers(cir, feat, y)
    return cir, feat, y


# ── Training utilities ────────────────────────────────────────────────────────

def _make_loader(tensors, labels, bs=256, shuffle=True):
    all_t = [torch.from_numpy(t) for t in tensors]
    all_t.append(torch.from_numpy(labels.astype(np.float32)))
    return DataLoader(TensorDataset(*all_t), batch_size=bs,
                      shuffle=shuffle, num_workers=0)


def _train(model, tr_loader, val_loader, fwd_fn, hp):
    model.to(DEVICE)
    pw   = torch.tensor([hp.get("pos_weight", 1.0)], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=hp["epochs"], eta_min=1e-5)
    best, best_state, no_imp = float("inf"), None, 0
    for ep in range(1, hp["epochs"] + 1):
        model.train()
        for batch in tr_loader:
            *Xb, yb = [t.to(DEVICE) for t in batch]
            opt.zero_grad(); crit(fwd_fn(model, *Xb), yb).backward(); opt.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                *Xb, yb = [t.to(DEVICE) for t in batch]
                vl += crit(fwd_fn(model, *Xb), yb).item() * len(yb)
        vl /= len(val_loader.dataset); sched.step()
        if vl < best - 1e-5:
            best, best_state, no_imp = vl, deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= hp["patience"]: break
    model.load_state_dict(best_state)
    return model


def _eval_metrics(model, te_loader, fwd_fn):
    model.eval().to(DEVICE)
    logits_all, y_all = [], []
    with torch.no_grad():
        for batch in te_loader:
            *Xb, yb = [t.to(DEVICE) for t in batch]
            out = fwd_fn(model, *Xb).cpu().squeeze(-1)
            logits_all.extend(out.tolist())
            y_all.extend(yb.cpu().long().tolist())
    probs = torch.sigmoid(torch.tensor(logits_all)).numpy()
    preds = (probs > 0.5).astype(int)
    trues = np.array(y_all)
    cm = confusion_matrix(trues, preds)
    tn, fp, fn, tp = cm.ravel()
    return {"acc": float(accuracy_score(trues, preds)),
            "f1":  float(f1_score(trues, preds, zero_division=0)),
            "auc": float(roc_auc_score(trues, probs)),
            "recall_nlos": float(tp / (tp + fn + 1e-9)),
            "recall_los":  float(tn / (tn + fp + 1e-9))}


def _split_1d(X, y, seed):
    """Single array (N, D) → 70/15/15 split + StandardScaler."""
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=seed)
    sc = StandardScaler().fit(X_tr)
    return (sc.transform(X_tr).astype(np.float32), y_tr,
            sc.transform(X_va).astype(np.float32), y_va,
            sc.transform(X_te).astype(np.float32), y_te)


def _split_2arr(cir, feat, y, seed):
    """Two arrays (cir, feat) → 70/15/15 with separate scalers."""
    idx = np.arange(len(y))
    idx_tv, idx_te = train_test_split(idx, test_size=0.15,
                                      stratify=y, random_state=seed)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.15/0.85,
                                      stratify=y[idx_tv], random_state=seed)
    sc_c = StandardScaler().fit(cir[idx_tr])
    sc_f = StandardScaler().fit(feat[idx_tr])
    def s(sc, a, idx): return sc.transform(a[idx]).astype(np.float32)
    return (s(sc_c, cir, idx_tr),  s(sc_f, feat, idx_tr),  y[idx_tr],
            s(sc_c, cir, idx_va),  s(sc_f, feat, idx_va),  y[idx_va],
            s(sc_c, cir, idx_te),  s(sc_f, feat, idx_te),  y[idx_te])


def _summarize(name, per_seed):
    keys = ["acc", "f1", "auc", "recall_nlos", "recall_los"]
    out  = {"model": name, "seeds": SEEDS, "n_runs": len(per_seed)}
    for k in keys:
        vals = [r[k] * 100 for r in per_seed]
        out[f"{k}_mean"] = round(float(np.mean(vals)), 2)
        out[f"{k}_std"]  = round(float(np.std(vals, ddof=0)), 2)
    out["per_seed"] = per_seed
    print(f"\n{'='*50}\n{name} — 5-seed summary (w+=1.0, τ=0.50)\n{'='*50}")
    print(f"  Acc        : {out['acc_mean']:.2f} ± {out['acc_std']:.2f}%")
    print(f"  F1         : {out['f1_mean']:.2f} ± {out['f1_std']:.2f}%")
    print(f"  AUC        : {out['auc_mean']:.2f} ± {out['auc_std']:.2f}%")
    print(f"  NLOS Recall: {out['recall_nlos_mean']:.2f} ± {out['recall_nlos_std']:.2f}%")
    print(f"  LOS  Recall: {out['recall_los_mean']:.2f} ± {out['recall_los_std']:.2f}%")
    return out


# ── Per-model runners ─────────────────────────────────────────────────────────

def run_ecauwb(X57, y):
    fwd = lambda m, x: m(x)
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr,ytr, Xva,yva, Xte,yte = _split_1d(X57, y, seed)
        tr = _make_loader([Xtr], ytr); va = _make_loader([Xva], yva, shuffle=False)
        te = _make_loader([Xte], yte, shuffle=False)
        model = ECAUWBNet(**{k: HP_ECA[k] for k in [
            "cir_len","n_aux","ch1","ch2","k1","k2","eca_k",
            "aux_hid","feat_dim","clf_hid","dropout"]})
        _train(model, tr, va, fwd, HP_ECA)
        res = _eval_metrics(model, te, fwd)
        res["seed"] = seed; per_seed.append(res)
        print(f"  ECA-UWB seed={seed}: Acc={res['acc']*100:.2f}%  "
              f"F1={res['f1']*100:.2f}%  ({time.time()-t0:.0f}s)")
    return _summarize("ECA-UWB", per_seed)


def run_wu2024(X57, y):
    fwd = lambda m, x: m(x)
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr,ytr, Xva,yva, Xte,yte = _split_1d(X57, y, seed)
        tr = _make_loader([Xtr], ytr); va = _make_loader([Xva], yva, shuffle=False)
        te = _make_loader([Xte], yte, shuffle=False)
        model = Wu2024Net(in_dim=57)
        _train(model, tr, va, fwd, HP_WU)
        res = _eval_metrics(model, te, fwd)
        res["seed"] = seed; per_seed.append(res)
        print(f"  SA-TinyML seed={seed}: Acc={res['acc']*100:.2f}%  "
              f"({time.time()-t0:.0f}s)")
    return _summarize("SA-TinyML", per_seed)


def run_cnn(cir_raw, feat_raw, y):
    """CNN uses preprocess.load_ewine (11 computed features, cir shape (N,50,1))."""
    def fwd(m, c, f): return m(c, f).squeeze(-1)   # c:(B,50,1) f:(B,11) → (B,)
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        (cir_tr, feat_tr, ytr, cir_va, feat_va, yva,
         cir_te, feat_te, yte) = _split_2arr(cir_raw, feat_raw, y, seed)
        # Add channel dim: (N,50) → (N,50,1)
        cir_tr = cir_tr[..., np.newaxis]
        cir_va = cir_va[..., np.newaxis]
        cir_te = cir_te[..., np.newaxis]
        tr = _make_loader([cir_tr, feat_tr], ytr)
        va = _make_loader([cir_va, feat_va], yva, shuffle=False)
        te = _make_loader([cir_te, feat_te], yte, shuffle=False)
        model = NLOSClassifier()
        _train(model, tr, va, fwd, HP_CNN)
        res = _eval_metrics(model, te, fwd)
        res["seed"] = seed; per_seed.append(res)
        print(f"  CNN seed={seed}: Acc={res['acc']*100:.2f}%  ({time.time()-t0:.0f}s)")
    return _summarize("CNN", per_seed)


def run_si2023(cir_raw, feat_raw, y):
    """Si2023: full 1016-pt CIR + 11 engineered features."""
    def fwd(m, c, f): return m(c, f).squeeze(-1)   # ensure (B,)
    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        (cir_tr, feat_tr, ytr, cir_va, feat_va, yva,
         cir_te, feat_te, yte) = _split_2arr(cir_raw, feat_raw, y, seed)
        tr = _make_loader([cir_tr, feat_tr], ytr)
        va = _make_loader([cir_va, feat_va], yva, shuffle=False)
        te = _make_loader([cir_te, feat_te], yte, shuffle=False)
        model = Si2023Net(cir_len=1016, n_manual=11)
        _train(model, tr, va, fwd, HP_SI)
        res = _eval_metrics(model, te, fwd)
        res["seed"] = seed; per_seed.append(res)
        print(f"  Si2023 seed={seed}: Acc={res['acc']*100:.2f}%  ({time.time()-t0:.0f}s)")
    return _summarize("CIR-CNN+MLP", per_seed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["ecauwb", "wu2024", "cnn", "si2023"])
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-seed validation  (w+=1.0, τ=0.50, seeds 0-4)")
    print("=" * 60)

    all_results = {}

    # Load shared 57-D dataset once for ECA-UWB + SA-TinyML
    if "ecauwb" in args.models or "wu2024" in args.models:
        print("\nLoading 57-D (50-CIR + 7 raw diag) ...")
        X57, y57 = load_57d()
        print(f"  {X57.shape[0]} samples after outlier removal")

    # Load CNN / Si2023 dataset (11 computed features)
    if "cnn" in args.models or "si2023" in args.models:
        print("\nLoading CNN/Si2023 data (50-CIR + 11 computed features + 1016-CIR) ...")
        # load_ewine from preprocess gives (cir_50, feat_11, labels)
        cir50, feat11, y_cnn = load_ewine()
        cir50, feat11, y_cnn = remove_outliers(cir50, feat11, y_cnn)
        print(f"  CNN 50-CIR: {cir50.shape[0]} samples")

    if "si2023" in args.models:
        print("\nLoading Si2023 full-CIR data ...")
        cir1016, feat11_si, y_si = load_si2023_data()
        print(f"  Si2023 1016-CIR: {cir1016.shape[0]} samples")

    # ── Run models ──────────────────────────────────────────────────────────
    if "ecauwb" in args.models:
        print("\n" + "="*60 + "\nECA-UWB\n" + "="*60)
        all_results["ECA-UWB"] = run_ecauwb(X57, y57)

    if "wu2024" in args.models:
        print("\n" + "="*60 + "\nSA-TinyML\n" + "="*60)
        all_results["SA-TinyML"] = run_wu2024(X57, y57)

    if "cnn" in args.models:
        print("\n" + "="*60 + "\nCNN\n" + "="*60)
        all_results["CNN"] = run_cnn(cir50, feat11, y_cnn)

    if "si2023" in args.models:
        print("\n" + "="*60 + "\nCIR-CNN+MLP\n" + "="*60)
        all_results["CIR-CNN+MLP"] = run_si2023(cir1016, feat11_si, y_si)

    # ── Save & print table ───────────────────────────────────────────────────
    out_path = config.LOG_DIR / "multi_seed_all_bce10.json"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved → {out_path}")

    param_map = {"ECA-UWB": 2374, "SA-TinyML": 4627, "CNN": 7441, "CIR-CNN+MLP": 1578}
    print(f"\n{'='*72}")
    print("FINAL TABLE  (w+=1.0, τ=0.50, seeds 0-4)")
    print(f"{'='*72}")
    hdr = f"{'Model':<16} {'Acc (mean±std)':>18} {'F1':>14} {'NLOS-R':>14} {'Params':>8}"
    print(hdr); print("-"*len(hdr))
    for m, r in all_results.items():
        p = param_map.get(m, "?")
        print(f"{m:<16} "
              f"{r['acc_mean']:>6.2f}±{r['acc_std']:<5.2f}   "
              f"{r['f1_mean']:>6.2f}±{r['f1_std']:<5.2f}   "
              f"{r['recall_nlos_mean']:>6.2f}±{r['recall_nlos_std']:<5.2f}   "
              f"{p:>8}")


if __name__ == "__main__":
    main()
