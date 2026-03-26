"""
Training script — supports both CNN (PyTorch) and RF baseline (sklearn).

Usage:
  python train.py --model cnn       # dual-branch CNN  (requires torch)
  python train.py --model rf        # Random Forest    (sklearn only)
  python train.py --model both      # train both, compare

Outputs saved to models/:
  best_model.pt          — best CNN checkpoint (state_dict)
  rf_model.pkl           — Random Forest pickle
  scaler.pkl             — StandardScaler for features
  training_history.png   — loss/accuracy curves
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from preprocess import build_dataset

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_history(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Training curves → {save_path}")


# ─────────────────────────────────────────────────────────────
# CNN training
# ─────────────────────────────────────────────────────────────

def train_cnn(splits):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from model import NLOSClassifier, count_params

    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CNN] Device: {device}")

    (cir_tr, feat_tr, y_tr,
     cir_val, feat_val, y_val,
     cir_te, feat_te, y_te, _) = splits

    def to_tensor(arr, dtype=torch.float32):
        return torch.tensor(arr, dtype=dtype)

    # DataLoaders
    train_ds = TensorDataset(to_tensor(cir_tr), to_tensor(feat_tr),
                              to_tensor(y_tr, torch.float32))
    val_ds   = TensorDataset(to_tensor(cir_val), to_tensor(feat_val),
                              to_tensor(y_val, torch.float32))
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE * 2,
                          shuffle=False, num_workers=2)

    model     = NLOSClassifier(n_feat=config.N_FEATURES,
                                dropout=config.DROPOUT).to(device)
    print(f"  Parameters: {count_params(model):,}")

    # Class weights for imbalanced datasets
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=config.LR_MIN
    )

    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val  = float("inf")
    patience  = 0
    best_path = config.KERAS_MODEL_PATH.with_suffix(".pt")   # save as .pt
    best_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Ep':>4}  {'TrainLoss':>10}  {'ValLoss':>10}  "
          f"{'TrainAcc':>9}  {'ValAcc':>9}  {'LR':>8}")
    print("─" * 62)

    for epoch in range(1, config.EPOCHS + 1):
        # ── train ──────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for cir_b, feat_b, y_b in train_dl:
            cir_b, feat_b, y_b = cir_b.to(device), feat_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logit = model(cir_b, feat_b).squeeze(1)
            loss  = criterion(logit, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_b)
            correct    += ((logit.sigmoid() > 0.5).float() == y_b).sum().item()
            total      += len(y_b)

        tr_loss = total_loss / total
        tr_acc  = correct   / total

        # ── validate ───────────────────────────────────────
        model.eval()
        vl_loss, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for cir_b, feat_b, y_b in val_dl:
                cir_b, feat_b, y_b = cir_b.to(device), feat_b.to(device), y_b.to(device)
                logit = model(cir_b, feat_b).squeeze(1)
                loss  = criterion(logit, y_b)
                vl_loss += loss.item() * len(y_b)
                vc += ((logit.sigmoid() > 0.5).float() == y_b).sum().item()
                vt += len(y_b)

        vl_loss /= vt
        vl_acc   = vc / vt
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"]  .append(vl_loss)
        history["train_acc"] .append(tr_acc)
        history["val_acc"]   .append(vl_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>4}  {tr_loss:>10.4f}  {vl_loss:>10.4f}  "
              f"{tr_acc:>8.4f}  {vl_acc:>8.4f}  {lr_now:>8.2e}")

        if vl_loss < best_val:
            best_val = vl_loss
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    print(f"\n  Best val loss: {best_val:.4f}  → {best_path}")
    plot_history(history, config.LOG_DIR / "training_history.png")
    _save_json(history, config.LOG_DIR / "history.json")

    # Load best weights for test evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, device


# ─────────────────────────────────────────────────────────────
# Random Forest training
# ─────────────────────────────────────────────────────────────

def train_rf(splits):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from model import build_rf_baseline

    (cir_tr, feat_tr, y_tr,
     cir_val, feat_val, y_val,
     cir_te, feat_te, y_te, _) = splits

    # Flatten CIR + concatenate with features for RF input
    # RF input: (N, 50 + 11) = (N, 61)
    def make_X(cir, feat):
        return np.concatenate([cir.squeeze(-1), feat], axis=1)

    X_tr  = make_X(cir_tr,  feat_tr)
    X_val = make_X(cir_val, feat_val)
    X_te  = make_X(cir_te,  feat_te)

    print(f"\n[RF] Training on {len(y_tr)} samples, X shape: {X_tr.shape}")
    rf = build_rf_baseline()
    t0 = time.perf_counter()
    rf.fit(X_tr, y_tr)
    print(f"  Training time: {time.perf_counter()-t0:.1f}s")

    for name, X, y in [("Val", X_val, y_val), ("Test", X_te, y_te)]:
        pred = rf.predict(X)
        acc  = accuracy_score(y, pred)
        f1   = f1_score(y, pred)
        print(f"\n  {name}: acc={acc:.4f}  F1={f1:.4f}")
        print(classification_report(y, pred, target_names=["LOS", "NLOS"]))

    rf_path = config.MODEL_DIR / "rf_model.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  RF model saved → {rf_path}")
    return rf


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both",
                        choices=["cnn", "rf", "both"],
                        help="Which model(s) to train")
    parser.add_argument("--oiud", action="store_true",
                        help="Also load OIUD dataset if available")
    args = parser.parse_args()

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Building dataset …")
    splits = build_dataset(use_oiud=args.oiud)

    if args.model in ("rf", "both"):
        train_rf(splits)

    if args.model in ("cnn", "both"):
        try:
            import torch
            train_cnn(splits)
        except ImportError:
            print("\n[CNN] PyTorch not installed — skipping CNN training.")
            print("  Install: pip install torch --index-url "
                  "https://download.pytorch.org/whl/cpu")


if __name__ == "__main__":
    main()
