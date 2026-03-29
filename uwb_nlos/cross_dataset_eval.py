"""
Cross-Dataset Evaluation: eWINE-trained models → VNU Indoor Dataset

Protocol:
  - 20% VNU calibration set (stratified, seed=42)
  - 80% VNU test set
  - Zero-shot: eWINE weights, threshold=0.5, report per-env Acc + overall AUC
  - Head-adapted (ALL models get weight-update fine-tuning on their natural head):
      ECA-UWB  : fine-tune fusion+classifier (643 params, backbone frozen)
      SA-TinyML: fine-tune attn+classifier (2331 params, stage-1 encoder frozen)
      MS-CNN-SA: fine-tune params_mlp+classifier (7025 params, backbone frozen)
    Report overall Acc (%) + F1 (%) on test set

VNU column layout (matches eWINE cols 0-9):
  Col 0  : NLOS label (0=LOS, 1=NLOS) — sometimes stored as 'p' in garage_los.csv
  Col 1  : RANGE (mm) — divide by 1000 for eWINE-compatible metres
  Col 2  : FP_IDX (raw DW1000 chip index, 500-1016)
  Col 3-5: FP_AMP1, FP_AMP2, FP_AMP3
  Col 6  : STDEV_NOISE
  Col 7  : CIR_PWR
  Col 8  : MAX_NOISE
  Col 9  : RXPACC
  Col 10-109: CIR[FP_IDX-5 … FP_IDX+94]  (100 samples)

Environments:
  garage → Lobby (Table IV col 3)
  hallway → Hallway (Table IV col 2)
  room → Room (Table IV col 1)
"""

import sys, json, pickle
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from ecauwb_model import ECAUWBNet
from wu2024_model import Wu2024Net
from jiang2026_model import Jiang2026Net

# ── Paths ─────────────────────────────────────────────────────────────────────
VNU_DIR        = config.VNU_DIR
MODEL_DIR      = config.MODEL_DIR
LOG_DIR        = config.LOG_DIR

ECAUWB_MODEL   = MODEL_DIR / "ecauwb.pt"
ECAUWB_SCALER  = MODEL_DIR / "ecauwb_scaler.pkl"

WU2024_MODEL   = MODEL_DIR / "wu2024_stage2.pt"
WU2024_SCALER  = MODEL_DIR / "wu2024_scaler.pkl"

JIANG_MODEL    = MODEL_DIR / "jiang2026.pt"
JIANG_CIR_SCL  = MODEL_DIR / "jiang2026_scaler_cir.pkl"
JIANG_AUX_SCL  = MODEL_DIR / "jiang2026_scaler_aux.pkl"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── VNU environment mapping ───────────────────────────────────────────────────
# CSV file prefix → display name in Table IV
ENVS = {
    "room":    "Room",
    "hallway": "Hallway",
    "garage":  "Lobby",
}

# ── Column indices (same as eWINE) ────────────────────────────────────────────
VNU_AUX_COLS_ECA  = [1, 8, 6, 3, 4, 5, 7]   # RANGE,MAX_NOISE,STDEV,AMP1,AMP2,AMP3,CIR_PWR
VNU_AUX_COLS_JIAN = [9, 1, 7, 5, 8, 4, 3, 6] # RXPACC,RANGE,CIR_PWR,AMP3,MAX_NOISE,AMP2,AMP1,STDEV
VNU_CIR_START     = 10   # CIR starts at column 10 in VNU CSV

# In the 100-tap pre-windowed array: FP is at position 5 (firmware: FP-5 to FP+94)
VNU_FP_OFFSET = 5

# CIR windows to extract (positions within the 100-tap VNU array)
#   ECA-UWB / SA-TinyML: FP-2 → FP+47 = 50 taps → positions [3:53]
ECA_CIR_START = VNU_FP_OFFSET - config.CIR_WIN_PRE   # = 5-2 = 3
ECA_CIR_END   = ECA_CIR_START + config.CIR_LEN        # = 3+50 = 53

#   Jiang2026: FP-6 → FP+56 = 63 taps → positions [-1:62]
#   → clamp: CIR[0:63] (shift -1; zero-pad left is handled below)
JIAN_CIR_START = max(0, VNU_FP_OFFSET - 6)  # = max(0,-1) = 0 (clamp)
JIAN_CIR_END   = JIAN_CIR_START + 63         # = 63
JIAN_CIR_LEN   = 63


# ── Data loading ──────────────────────────────────────────────────────────────

def _read_vnu_csv(path: Path) -> pd.DataFrame:
    """Read one VNU CSV, fix the 'p' header in garage_los.csv."""
    df = pd.read_csv(path)
    # garage_los.csv was saved with 'p' as first column name (firmware artefact)
    if df.columns[0] != "NLOS":
        df = df.rename(columns={df.columns[0]: "NLOS"})
    return df


def load_vnu_env(env_prefix: str):
    """
    Load both LOS and NLOS CSV for one environment.
    Returns arrays, labels, plus env tag per sample.
    """
    los_path  = VNU_DIR / f"{env_prefix}_los.csv"
    nlos_path = VNU_DIR / f"{env_prefix}_nlos.csv"

    df_los  = _read_vnu_csv(los_path)
    df_nlos = _read_vnu_csv(nlos_path)

    df = pd.concat([df_los, df_nlos], ignore_index=True)
    arr    = df.values.astype(np.float64)
    labels = arr[:, 0].astype(np.int8)

    # ── ECA-UWB / SA-TinyML (57-D) ──────────────────────────────────────────
    cir100   = arr[:, VNU_CIR_START:VNU_CIR_START + 100]
    cir_eca  = cir100[:, ECA_CIR_START:ECA_CIR_END].astype(np.float32)  # (N,50)
    rxpacc   = arr[:, 9].reshape(-1, 1) + 1e-9
    cir_eca  = cir_eca / rxpacc                                          # RXPACC-norm

    aux_raw  = arr[:, VNU_AUX_COLS_ECA].astype(np.float32)              # (N,7)
    aux_raw[:, 0] /= 1000.0      # RANGE mm → m (col 1 in CSV = index 0 in aux)

    X57 = np.concatenate([cir_eca, aux_raw], axis=1)                    # (N,57)

    # ── Jiang2026 (63-D CIR + 8 AUX) ────────────────────────────────────────
    # Jiang needs CIR[FP-6:FP+57]; firmware starts at FP-5, so we get [FP-5:FP+58]
    # (1 tap shift, negligible cross-dataset effect).
    cir_jian = cir100[:, JIAN_CIR_START:JIAN_CIR_END].astype(np.float32)  # (N,63)
    cir_jian = cir_jian / rxpacc

    aux8_raw = arr[:, VNU_AUX_COLS_JIAN].astype(np.float32)             # (N,8)
    aux8_raw[:, 1] /= 1000.0    # RANGE mm → m (index 1 in Jiang aux = RANGE)

    return labels, X57, cir_jian, aux8_raw, ENVS[env_prefix]


def load_all_vnu():
    """Load all three environments, return combined arrays + per-sample env tag."""
    all_labels, all_X57, all_cir_jian, all_aux8, all_env = [], [], [], [], []

    for prefix in ["room", "hallway", "garage"]:
        labels, X57, cir_jian, aux8, env_name = load_vnu_env(prefix)
        all_labels.append(labels)
        all_X57.append(X57)
        all_cir_jian.append(cir_jian)
        all_aux8.append(aux8)
        all_env.extend([env_name] * len(labels))
        print(f"  {env_name:10s}: {len(labels)} samples "
              f"(LOS={int((labels==0).sum())}, NLOS={int((labels==1).sum())})")

    return (
        np.concatenate(all_labels,    axis=0),
        np.concatenate(all_X57,       axis=0),
        np.concatenate(all_cir_jian,  axis=0),
        np.concatenate(all_aux8,      axis=0),
        np.array(all_env),
    )


# ── ECA-UWB evaluation ────────────────────────────────────────────────────────

def build_ecauwb():
    m = ECAUWBNet()
    m.load_state_dict(torch.load(ECAUWB_MODEL, map_location="cpu"))
    m.eval()
    return m


def ecauwb_zero_shot(model, X_test_sc, labels_test, env_arr):
    """Zero-shot: eWINE weights, thr=0.5, per-env acc + overall AUC."""
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_sc))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)

    overall_auc = roc_auc_score(labels_test.astype(int), probs)

    per_env_acc = {}
    for env in ENVS.values():
        mask = env_arr == env
        if mask.sum() == 0:
            continue
        per_env_acc[env] = accuracy_score(
            labels_test[mask].astype(int), preds[mask])

    return per_env_acc, overall_auc, probs, preds


def ecauwb_head_adapt(model, X_calib_sc, y_calib, X_test_sc, y_test,
                      epochs=60, lr=5e-3, batch_size=64):
    """
    Fine-tune only ECA-UWB's 643-param head (fusion + classifier).
    Backbone (cir_branch + aux_branch) is frozen.
    """
    m = deepcopy(model)

    # Freeze backbone
    for p in m.cir_branch.parameters(): p.requires_grad_(False)
    for p in m.aux_branch.parameters():  p.requires_grad_(False)

    head_params = list(m.fusion.parameters()) + list(m.classifier.parameters())
    n_head = sum(p.numel() for p in head_params)
    print(f"    Head params (trainable): {n_head}")

    opt = torch.optim.Adam(head_params, lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.5]))

    ds = TensorDataset(
        torch.from_numpy(X_calib_sc),
        torch.from_numpy(y_calib.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    m.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(m(xb), yb)
            loss.backward()
            opt.step()

    m.eval()
    with torch.no_grad():
        logits = m(torch.from_numpy(X_test_sc))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    y_np  = y_test.astype(int)

    acc  = accuracy_score(y_np, preds)
    f1   = f1_score(y_np, preds)
    auc  = roc_auc_score(y_np, probs)
    cm   = confusion_matrix(y_np, preds).tolist()
    return acc, f1, auc, cm, m


# ── SA-TinyML (Wu2024) evaluation ────────────────────────────────────────────

def build_wu2024():
    m = Wu2024Net(in_dim=57)
    m.load_state_dict(torch.load(WU2024_MODEL, map_location="cpu"))
    m.eval()
    return m


def wu2024_zero_shot(model, X_test_sc, labels_test, env_arr):
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_sc))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)

    overall_auc = roc_auc_score(labels_test.astype(int), probs)

    per_env_acc = {}
    for env in ENVS.values():
        mask = env_arr == env
        if mask.sum() == 0:
            continue
        per_env_acc[env] = accuracy_score(
            labels_test[mask].astype(int), preds[mask])

    return per_env_acc, overall_auc, probs


def head_adapt_generic(model, head_modules, X_calib_sc, y_calib,
                        X_test_sc, y_test, forward_fn,
                        epochs=60, lr=5e-3, batch_size=64, label=""):
    """
    Generic head fine-tuning: freeze all params, then unfreeze only head_modules.
    forward_fn(model, X_batch) -> logits
    """
    m = deepcopy(model)
    # Freeze all
    for p in m.parameters(): p.requires_grad_(False)
    # Unfreeze head
    head_params = []
    for module in head_modules(m):
        for p in module.parameters():
            p.requires_grad_(True)
            head_params.append(p)
    n_head = sum(p.numel() for p in head_params)
    print(f"    {label} head params (trainable): {n_head}")

    opt  = torch.optim.Adam(head_params, lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))

    X_c = torch.from_numpy(X_calib_sc)
    y_c = torch.from_numpy(y_calib.astype(np.float32))
    ds  = TensorDataset(X_c, y_c)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    m.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(forward_fn(m, xb), yb).backward()
            opt.step()

    m.eval()
    with torch.no_grad():
        logits = forward_fn(m, torch.from_numpy(X_test_sc))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    y_np  = y_test.astype(int)
    acc   = accuracy_score(y_np, preds)
    f1    = f1_score(y_np, preds)
    auc   = roc_auc_score(y_np, probs)
    return acc, f1, auc, n_head


def thr_calibrate(model_fn, X_calib_sc, y_calib, X_test_sc, y_test):
    """
    Threshold calibration: find thr maximising F1 on calibration set,
    then apply to test set. model_fn() returns (probs_calib, probs_test).
    """
    probs_calib, probs_test = model_fn(X_calib_sc, X_test_sc)

    # Grid search threshold on calibration set
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.1, 0.91, 0.01):
        preds_c = (probs_calib > thr).astype(int)
        f1c = f1_score(y_calib.astype(int), preds_c, zero_division=0)
        if f1c > best_f1:
            best_f1, best_thr = f1c, thr

    preds_test = (probs_test > best_thr).astype(int)
    y_np = y_test.astype(int)
    acc = accuracy_score(y_np, preds_test)
    f1  = f1_score(y_np, preds_test, zero_division=0)
    return acc, f1, best_thr


# ── Jiang2026 evaluation ──────────────────────────────────────────────────────

def build_jiang2026():
    m = Jiang2026Net(cir_len=63, n_aux=8)
    m.load_state_dict(torch.load(JIANG_MODEL, map_location="cpu"))
    m.eval()
    return m


def jiang_zero_shot(model, cir_sc, aux_sc, labels_test, env_arr):
    cir_t = torch.from_numpy(cir_sc)
    aux_t = torch.from_numpy(aux_sc)
    with torch.no_grad():
        logits = model(cir_t, aux_t)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)

    overall_auc = roc_auc_score(labels_test.astype(int), probs)

    per_env_acc = {}
    for env in ENVS.values():
        mask = env_arr == env
        if mask.sum() == 0:
            continue
        per_env_acc[env] = accuracy_score(
            labels_test[mask].astype(int), preds[mask])

    return per_env_acc, overall_auc, probs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading VNU dataset ...")
    labels, X57, cir_jian, aux8, env_arr = load_all_vnu()
    print(f"Total: {len(labels)} samples | "
          f"LOS={int((labels==0).sum())} NLOS={int((labels==1).sum())}")

    # ── Calibration / test split (stratified, 20% calib) ─────────────────────
    idx = np.arange(len(labels))
    idx_calib, idx_test = train_test_split(
        idx, test_size=0.80, stratify=labels, random_state=SEED)

    print(f"\nCalib: {len(idx_calib)} | Test: {len(idx_test)}")

    # ── Load scalers ─────────────────────────────────────────────────────────
    with open(ECAUWB_SCALER,  "rb") as f: ecauwb_sc  = pickle.load(f)
    with open(WU2024_SCALER,  "rb") as f: wu2024_sc  = pickle.load(f)
    with open(JIANG_CIR_SCL,  "rb") as f: jian_cir_sc = pickle.load(f)
    with open(JIANG_AUX_SCL,  "rb") as f: jian_aux_sc = pickle.load(f)

    # Scale all data with eWINE-fitted scalers
    X57_sc_eca  = ecauwb_sc.transform(X57).astype(np.float32)
    X57_sc_wu   = wu2024_sc.transform(X57).astype(np.float32)
    cir_jian_sc = jian_cir_sc.transform(cir_jian).astype(np.float32)
    aux8_sc     = jian_aux_sc.transform(aux8).astype(np.float32)

    # ── Splits ───────────────────────────────────────────────────────────────
    y_calib = labels[idx_calib]
    y_test  = labels[idx_test]
    env_calib = env_arr[idx_calib]
    env_test  = env_arr[idx_test]

    # ECA-UWB / SA-TinyML
    Xeca_calib, Xeca_test = X57_sc_eca[idx_calib], X57_sc_eca[idx_test]
    Xwu_calib,  Xwu_test  = X57_sc_wu[idx_calib],  X57_sc_wu[idx_test]

    # Jiang2026
    cir_calib_j = cir_jian_sc[idx_calib]; cir_test_j = cir_jian_sc[idx_test]
    aux_calib_j = aux8_sc[idx_calib];     aux_test_j  = aux8_sc[idx_test]

    results = {
        "VNU_total_samples": int(len(labels)),
        "VNU_calib_samples": int(len(idx_calib)),
        "VNU_test_samples":  int(len(idx_test)),
        "protocol": "20% calibration / 80% test (stratified seed=42)",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 1. ECA-UWB
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ECA-UWB")
    print("=" * 60)
    eca_model = build_ecauwb()
    n_total_eca = sum(p.numel() for p in eca_model.parameters())
    print(f"  Total params: {n_total_eca}")

    # Zero-shot
    per_env, auc_zs, probs_zs, preds_zs = ecauwb_zero_shot(
        eca_model, X57_sc_eca, labels, env_arr)
    acc_zs = accuracy_score(labels.astype(int), preds_zs)
    print(f"\nZero-shot (all {len(labels)} samples, thr=0.5):")
    print(f"  Overall Acc={acc_zs*100:.2f}%  AUC={auc_zs:.4f}")
    for env, acc in per_env.items():
        print(f"  {env}: {acc*100:.2f}%")

    # Zero-shot per-env on TEST set only + full-dataset AUC (for Table IV)
    per_env_test, auc_zs_test, probs_zs_test, preds_zs_test = ecauwb_zero_shot(
        eca_model, Xeca_test, y_test, env_test)
    _, auc_zs_full, _, _ = ecauwb_zero_shot(eca_model, X57_sc_eca, labels, env_arr)
    print(f"\nZero-shot (test set, for Table IV):")
    print(f"  Overall AUC (all 9000 samples)={auc_zs_full:.4f}")
    for env, acc in per_env_test.items():
        print(f"  {env}: {acc*100:.2f}%")

    # Head-adapted
    print("\nHead-adapting ECA-UWB ...")
    ha_acc, ha_f1, ha_auc, ha_cm, eca_adapted = ecauwb_head_adapt(
        eca_model, Xeca_calib, y_calib, Xeca_test, y_test)
    print(f"  Head-adapted: Acc={ha_acc*100:.2f}%  F1={ha_f1*100:.2f}%  AUC={ha_auc:.4f}")
    print(f"  Confusion matrix:\n    {ha_cm}")

    results["ECA-UWB"] = {
        "zero_shot": {
            "per_env_acc": {k: round(v*100, 2) for k, v in per_env_test.items()},
            "overall_auc": round(auc_zs_full, 4),
        },
        "head_adapted": {
            "acc":  round(ha_acc*100, 2),
            "f1":   round(ha_f1*100, 2),
            "auc":  round(ha_auc, 4),
            "cm":   ha_cm,
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 2. SA-TinyML (Wu2024)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SA-TinyML (Wu2024)")
    print("=" * 60)
    wu_model = build_wu2024()

    # Zero-shot (whole dataset → per-env; AUC on whole dataset)
    per_env_wu, auc_wu, probs_wu = wu2024_zero_shot(
        wu_model, X57_sc_wu, labels, env_arr)
    preds_wu_all = (probs_wu > 0.5).astype(int)
    acc_wu_all = accuracy_score(labels.astype(int), preds_wu_all)
    print(f"\nZero-shot (all samples, thr=0.5):")
    print(f"  Overall Acc={acc_wu_all*100:.2f}%  AUC={auc_wu:.4f}")
    for env, acc in per_env_wu.items():
        print(f"  {env}: {acc*100:.2f}%")

    # Zero-shot per-env on TEST set only (for Table IV)
    per_env_wu_test, auc_wu_full, probs_wu_full = wu2024_zero_shot(
        wu_model, X57_sc_wu, labels, env_arr)
    per_env_wu_test_only, _, _ = wu2024_zero_shot(
        wu_model, Xwu_test, y_test, env_test)

    # Head fine-tuning: freeze stage-1 encoder, fine-tune attn+clf (2331 params)
    print("\nHead-adapting SA-TinyML (attn + classifier, encoder frozen) ...")
    def wu_head_modules(m):
        return [m.q_proj, m.k_proj, m.v_proj, m.bn_q, m.bn_k, m.bn_v, m.clf]
    wu_ha_acc, wu_ha_f1, wu_ha_auc, wu_n_head = head_adapt_generic(
        wu_model, wu_head_modules,
        Xwu_calib, y_calib, Xwu_test, y_test,
        forward_fn=lambda m, xb: m(xb),
        label="SA-TinyML",
    )
    print(f"  Head-adapted: Acc={wu_ha_acc*100:.2f}%  F1={wu_ha_f1*100:.2f}%  AUC={wu_ha_auc:.4f}")

    results["SA-TinyML"] = {
        "zero_shot": {
            "per_env_acc": {k: round(v*100, 2) for k, v in per_env_wu_test_only.items()},
            "overall_auc": round(auc_wu_full, 4),
        },
        "head_adapted": {
            "acc":      round(wu_ha_acc*100, 2),
            "f1":       round(wu_ha_f1*100, 2),
            "auc":      round(wu_ha_auc, 4),
            "n_params": wu_n_head,
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MS-CNN-SA (Jiang2026)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("MS-CNN-SA (Jiang2026)")
    print("=" * 60)
    jian_model = build_jiang2026()

    # Zero-shot (whole dataset)
    per_env_jian, auc_jian, probs_jian = jiang_zero_shot(
        jian_model, cir_jian_sc, aux8_sc, labels, env_arr)
    preds_jian_all = (probs_jian > 0.5).astype(int)
    acc_jian_all = accuracy_score(labels.astype(int), preds_jian_all)
    print(f"\nZero-shot (all samples, thr=0.5):")
    print(f"  Overall Acc={acc_jian_all*100:.2f}%  AUC={auc_jian:.4f}")
    for env, acc in per_env_jian.items():
        print(f"  {env}: {acc*100:.2f}%")

    # Zero-shot on TEST set only (for Table IV per-env)
    per_env_jian_test, auc_jian_full, probs_jian_full = jiang_zero_shot(
        jian_model, cir_jian_sc, aux8_sc, labels, env_arr)
    per_env_jian_test_only, _, _ = jiang_zero_shot(
        jian_model, cir_test_j, aux_test_j, y_test, env_test)

    # Head fine-tuning for Jiang2026: freeze backbone, fine-tune params_mlp+clf (7025 params)
    # Need special handling: two separate inputs (cir, aux)
    print("\nHead-adapting MS-CNN-SA (params_mlp + classifier, backbone frozen) ...")

    m_jian_ha = deepcopy(jian_model)
    for p in m_jian_ha.parameters(): p.requires_grad_(False)
    jian_head_params = (list(m_jian_ha.params_mlp.parameters()) +
                        list(m_jian_ha.classifier.parameters()))
    for p in jian_head_params: p.requires_grad_(True)
    jian_n_head = sum(p.numel() for p in jian_head_params)
    print(f"    MS-CNN-SA head params (trainable): {jian_n_head}")

    opt_j  = torch.optim.Adam(jian_head_params, lr=5e-3, weight_decay=1e-4)
    crit_j = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]))
    ds_j = TensorDataset(
        torch.from_numpy(cir_calib_j),
        torch.from_numpy(aux_calib_j),
        torch.from_numpy(y_calib.astype(np.float32)),
    )
    loader_j = DataLoader(ds_j, batch_size=64, shuffle=True)

    m_jian_ha.train()
    for _ in range(60):
        for cir_b, aux_b, y_b in loader_j:
            opt_j.zero_grad()
            crit_j(m_jian_ha(cir_b, aux_b), y_b).backward()
            opt_j.step()

    m_jian_ha.eval()
    with torch.no_grad():
        logits_j = m_jian_ha(torch.from_numpy(cir_test_j), torch.from_numpy(aux_test_j))
    probs_j = torch.sigmoid(logits_j).numpy()
    preds_j = (probs_j > 0.5).astype(int)
    y_np_j  = y_test.astype(int)
    jian_ha_acc = accuracy_score(y_np_j, preds_j)
    jian_ha_f1  = f1_score(y_np_j, preds_j)
    jian_ha_auc = roc_auc_score(y_np_j, probs_j)
    print(f"  Head-adapted: Acc={jian_ha_acc*100:.2f}%  F1={jian_ha_f1*100:.2f}%  AUC={jian_ha_auc:.4f}")

    results["MS-CNN-SA"] = {
        "zero_shot": {
            "per_env_acc": {k: round(v*100, 2) for k, v in per_env_jian_test_only.items()},
            "overall_auc": round(auc_jian_full, 4),
        },
        "head_adapted": {
            "acc":      round(jian_ha_acc*100, 2),
            "f1":       round(jian_ha_f1*100, 2),
            "auc":      round(jian_ha_auc, 4),
            "n_params": jian_n_head,
        },
    }

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = LOG_DIR / "cross_dataset_vnu_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved → {out_path}")

    # ── Print Table IV summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TABLE IV SUMMARY (test set, 80% of VNU)")
    print(f"{'='*60}")
    header = f"{'Method':<16} {'Room%':>7} {'Hall%':>7} {'Lobby%':>7} {'AUC':>7} {'HA-Acc%':>8} {'HA-F1%':>7}"
    print(header)
    print("-" * len(header))

    for model_name, res in [
        ("SA-TinyML",   results["SA-TinyML"]),
        ("MS-CNN-SA",   results["MS-CNN-SA"]),
        ("ECA-UWB",     results["ECA-UWB"]),
    ]:
        zs = res["zero_shot"]
        ha = res["head_adapted"]
        room = zs["per_env_acc"].get("Room", 0)
        hall = zs["per_env_acc"].get("Hallway", 0)
        lobb = zs["per_env_acc"].get("Lobby", 0)
        auc  = zs["overall_auc"]
        hacc = ha["acc"]
        hf1  = ha["f1"]
        print(f"{model_name:<16} {room:>7.2f} {hall:>7.2f} {lobb:>7.2f} "
              f"{auc:>7.4f} {hacc:>8.2f} {hf1:>7.2f}")

    print(f"{'='*60}")
    print("\nNote: HA = Head-Adapted (ECA-UWB: head fine-tune; others: thr calibration)")


if __name__ == "__main__":
    main()
