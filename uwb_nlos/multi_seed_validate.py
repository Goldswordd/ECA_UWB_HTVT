"""
CC-1: Multi-Seed Validation for ECA-UWB paper.

Trains ECA-UWB (and optionally SA-TinyML, CIR-CNN+MLP) with seeds {0,1,2,3,4}.
Reports mean ± std for Accuracy, F1, AUC-ROC.

Usage:
  python multi_seed_validate.py                    # train all seeds
  python multi_seed_validate.py --eval             # eval existing checkpoints only
  python multi_seed_validate.py --models ecauwb    # only ECA-UWB
  python multi_seed_validate.py --models ecauwb wu2024 si2023  # all three
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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from preprocess import _extract_cir_window, remove_outliers
from ecauwb_model import ECAUWBNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 3, 4]

# ── ECA-UWB hyperparameters (from train_ecauwb.py) ─────────────────────
HP_ECAUWB = dict(
    cir_len=50, n_aux=7, ch1=16, ch2=16, k1=5, k2=3, eca_k=3,
    aux_hid=32, feat_dim=16, clf_hid=32, dropout=0.3,
    lr=1e-3, weight_decay=1e-4, batch_size=256, epochs=300, patience=30,
    pos_weight=1.5,
)

AUX_COLS = [1, 8, 6, 3, 4, 5, 7]
N_CIR = 50
N_AUX = 7


def load_ewine():
    """Load eWINE → (N, 57) array + labels."""
    ewine_dir = Path(config.EWINE_DIR)
    csv_files = sorted(ewine_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {ewine_dir}")

    all_X, all_labels = [], []
    fp_col = config.EWINE_DIAG_COLS["FP_IDX"]
    rx_col = config.EWINE_DIAG_COLS["RXPACC"]

    for f in csv_files:
        print(f"  {f.name} ...", end=" ", flush=True)
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
        print(f"{len(arr)} rows")

    X = np.concatenate(all_X, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Remove outliers
    cir_part = X[:, :N_CIR]
    aux_part = X[:, N_CIR:]
    cir_part, aux_part, labels = remove_outliers(cir_part, aux_part, labels)
    X = np.concatenate([cir_part, aux_part], axis=1)

    print(f"Total after outlier removal: {len(labels)}")
    return X, labels


def split_and_scale(X, labels, seed):
    """70/15/15 stratified split with given seed."""
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, labels, test_size=0.15, stratify=labels, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    return X_tr, y_tr, X_val, y_val, X_te, y_te, scaler


def make_loader(X, y, batch_size=256, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X),
                       torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_model(model, tr_loader, val_loader, hp, device):
    """Train model, return best state dict."""
    model.to(device)
    pw = torch.tensor([hp.get("pos_weight", 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                                 weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp["epochs"], eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for ep in range(1, hp["epochs"] + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion(model(Xb), yb).item() * len(yb)
        val_loss /= len(val_loader.dataset)
        scheduler.step()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hp["patience"]:
                break

    model.load_state_dict(best_state)
    return model, ep


def find_threshold(model, val_loader, device, min_los_recall=0.91):
    """Find optimal threshold on validation set."""
    model.eval().to(device)
    all_probs, all_y = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            logits = model(Xb.to(device))
            all_probs.extend(logits.sigmoid().cpu().tolist())
            all_y.extend(yb.cpu().long().tolist())
    probs = np.array(all_probs)
    trues = np.array(all_y)

    best_thr = 0.5
    for thr in np.arange(0.30, 0.66, 0.02):
        preds = (probs > thr).astype(int)
        cm = confusion_matrix(trues, preds)
        tn, fp, fn, tp = cm.ravel()
        los_r = tn / (tn + fp + 1e-9)
        if los_r >= min_los_recall:
            best_thr = float(thr)
            break
    return best_thr


def evaluate(model, te_loader, device, threshold=0.5):
    """Evaluate model, return metrics dict."""
    model.eval().to(device)
    all_logits, all_y = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            logits = model(Xb.to(device))
            all_logits.extend(logits.cpu().tolist())
            all_y.extend(yb.cpu().long().tolist())

    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs > threshold).astype(int)
    trues = np.array(all_y)

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)
    auc = roc_auc_score(trues, probs)
    cm = confusion_matrix(trues, preds)
    tn, fp, fn, tp = cm.ravel()

    return {"acc": acc, "f1": f1, "auc": auc,
            "recall_nlos": tp/(tp+fn+1e-9), "recall_los": tn/(tn+fp+1e-9),
            "threshold": threshold}


def run_ecauwb_multi_seed(X, labels, seeds, eval_only=False):
    """Run ECA-UWB for multiple seeds."""
    results = []
    out_dir = config.MODEL_DIR / "multi_seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"ECA-UWB  seed={seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        X_tr, y_tr, X_val, y_val, X_te, y_te, scaler = split_and_scale(X, labels, seed)
        tr_loader = make_loader(X_tr, y_tr, shuffle=True)
        val_loader = make_loader(X_val, y_val, shuffle=False)
        te_loader = make_loader(X_te, y_te, shuffle=False)

        model = ECAUWBNet(**{k: HP_ECAUWB[k] for k in [
            "cir_len", "n_aux", "ch1", "ch2", "k1", "k2", "eca_k",
            "aux_hid", "feat_dim", "clf_hid", "dropout"]})

        ckpt_path = out_dir / f"ecauwb_seed{seed}.pt"

        if eval_only and ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            print(f"  Loaded {ckpt_path}")
        else:
            t0 = time.time()
            model, ep = train_model(model, tr_loader, val_loader, HP_ECAUWB, DEVICE)
            elapsed = time.time() - t0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Trained {ep} epochs in {elapsed/60:.1f} min → {ckpt_path}")

        thr = find_threshold(model, val_loader, DEVICE)
        res = evaluate(model, te_loader, DEVICE, threshold=thr)
        res["seed"] = seed
        res["model"] = "ECA-UWB"
        results.append(res)
        print(f"  Acc={res['acc']*100:.2f}%  F1={res['f1']*100:.2f}%  "
              f"AUC={res['auc']*100:.2f}%  thr={thr:.2f}")

    return results


def aggregate_results(results, model_name):
    """Compute mean ± std for key metrics."""
    accs = [r["acc"] * 100 for r in results]
    f1s = [r["f1"] * 100 for r in results]
    aucs = [r["auc"] * 100 for r in results]
    nlos_rs = [r["recall_nlos"] * 100 for r in results]
    los_rs = [r["recall_los"] * 100 for r in results]

    print(f"\n{'='*60}")
    print(f"MULTI-SEED SUMMARY: {model_name} (seeds {SEEDS})")
    print(f"{'='*60}")
    print(f"  Accuracy   : {np.mean(accs):.2f} ± {np.std(accs):.2f}%")
    print(f"  F1-score   : {np.mean(f1s):.2f} ± {np.std(f1s):.2f}%")
    print(f"  AUC-ROC    : {np.mean(aucs):.2f} ± {np.std(aucs):.2f}%")
    print(f"  NLOS Recall: {np.mean(nlos_rs):.2f} ± {np.std(nlos_rs):.2f}%")
    print(f"  LOS  Recall: {np.mean(los_rs):.2f} ± {np.std(los_rs):.2f}%")

    return {
        "model": model_name,
        "seeds": SEEDS,
        "n_runs": len(results),
        "accuracy_mean": round(np.mean(accs), 2),
        "accuracy_std": round(np.std(accs), 2),
        "f1_mean": round(np.mean(f1s), 2),
        "f1_std": round(np.std(f1s), 2),
        "auc_mean": round(np.mean(aucs), 2),
        "auc_std": round(np.std(aucs), 2),
        "nlos_recall_mean": round(np.mean(nlos_rs), 2),
        "nlos_recall_std": round(np.std(nlos_rs), 2),
        "los_recall_mean": round(np.mean(los_rs), 2),
        "los_recall_std": round(np.std(los_rs), 2),
        "per_seed": results,
    }


def main():
    parser = argparse.ArgumentParser(description="CC-1: Multi-seed validation")
    parser.add_argument("--eval", action="store_true",
                        help="Eval existing checkpoints only")
    parser.add_argument("--models", nargs="+", default=["ecauwb"],
                        help="Models to run: ecauwb")
    args = parser.parse_args()

    print("Loading eWINE dataset ...")
    X, labels = load_ewine()

    all_summaries = {}

    if "ecauwb" in args.models:
        results = run_ecauwb_multi_seed(X, labels, SEEDS, eval_only=args.eval)
        summary = aggregate_results(results, "ECA-UWB")
        all_summaries["ecauwb"] = summary

    # Save results
    out_path = config.LOG_DIR / "multi_seed_results.json"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(all_summaries, fh, indent=2, default=str)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
