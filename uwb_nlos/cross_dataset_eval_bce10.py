"""
Cross-Dataset Re-evaluation: ECA-UWB backbone retrained to w+=1.0
Uses ecauwb_full.pt (ablation "full" variant, pos_weight=1.0, τ=0.50)
All head fine-tuning also uses standard BCE (pos_weight=1.0).
Saves to logs/cross_dataset_vnu_bce10.json.
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

# ── Paths ──────────────────────────────────────────────────────────────────────
VNU_DIR   = config.VNU_DIR
MODEL_DIR = config.MODEL_DIR
LOG_DIR   = config.LOG_DIR

# ECA-UWB: use w+=1.0 backbone (ablation "full" checkpoint)
ECAUWB_MODEL  = MODEL_DIR / "ecauwb_full.pt"   # ← w+=1.0
ECAUWB_SCALER = MODEL_DIR / "ecauwb_scaler.pkl"

WU2024_MODEL  = MODEL_DIR / "wu2024_stage2.pt"
WU2024_SCALER = MODEL_DIR / "wu2024_scaler.pkl"

JIANG_MODEL   = MODEL_DIR / "jiang2026.pt"
JIANG_CIR_SCL = MODEL_DIR / "jiang2026_scaler_cir.pkl"
JIANG_AUX_SCL = MODEL_DIR / "jiang2026_scaler_aux.pkl"

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

ENVS = {"room": "Room", "hallway": "Hallway", "garage": "Lobby"}

VNU_AUX_COLS_ECA  = [1, 8, 6, 3, 4, 5, 7]
VNU_AUX_COLS_JIAN = [9, 1, 7, 5, 8, 4, 3, 6]
VNU_CIR_START     = 10
VNU_FP_OFFSET     = 5
ECA_CIR_START     = VNU_FP_OFFSET - config.CIR_WIN_PRE
ECA_CIR_END       = ECA_CIR_START + config.CIR_LEN
JIAN_CIR_START    = max(0, VNU_FP_OFFSET - 6)
JIAN_CIR_END      = JIAN_CIR_START + 63

# Head fine-tune: standard BCE for all models (pos_weight=1.0, consistent protocol)
HEAD_POS_WEIGHT = 1.0


def _read_vnu_csv(path):
    df = pd.read_csv(path)
    if df.columns[0] != "NLOS":
        df = df.rename(columns={df.columns[0]: "NLOS"})
    return df


def load_vnu_env(env_prefix):
    df_los  = _read_vnu_csv(VNU_DIR / f"{env_prefix}_los.csv")
    df_nlos = _read_vnu_csv(VNU_DIR / f"{env_prefix}_nlos.csv")
    df = pd.concat([df_los, df_nlos], ignore_index=True)
    arr    = df.values.astype(np.float64)
    labels = arr[:, 0].astype(np.int8)

    cir100  = arr[:, VNU_CIR_START:VNU_CIR_START + 100]
    rxpacc  = arr[:, 9].reshape(-1, 1) + 1e-9
    cir_eca = (cir100[:, ECA_CIR_START:ECA_CIR_END] / rxpacc).astype(np.float32)
    aux_raw = arr[:, VNU_AUX_COLS_ECA].astype(np.float32)
    aux_raw[:, 0] /= 1000.0
    X57 = np.concatenate([cir_eca, aux_raw], axis=1)

    cir_jian = (cir100[:, JIAN_CIR_START:JIAN_CIR_END] / rxpacc).astype(np.float32)
    aux8_raw = arr[:, VNU_AUX_COLS_JIAN].astype(np.float32)
    aux8_raw[:, 1] /= 1000.0

    return labels, X57, cir_jian, aux8_raw, ENVS[env_prefix]


def load_all_vnu():
    all_labels, all_X57, all_cjian, all_a8, all_env = [], [], [], [], []
    for prefix in ["room", "hallway", "garage"]:
        labels, X57, cjian, a8, name = load_vnu_env(prefix)
        all_labels.append(labels); all_X57.append(X57)
        all_cjian.append(cjian); all_a8.append(a8)
        all_env.extend([name] * len(labels))
        print(f"  {name:10s}: {len(labels)} (LOS={int((labels==0).sum())}, "
              f"NLOS={int((labels==1).sum())})")
    return (np.concatenate(all_labels), np.concatenate(all_X57),
            np.concatenate(all_cjian), np.concatenate(all_a8),
            np.array(all_env))


def _per_env_acc_auc(logits_or_probs, labels, env_arr, is_logits=False):
    probs = torch.sigmoid(torch.tensor(logits_or_probs)).numpy() if is_logits \
            else np.array(logits_or_probs)
    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(labels.astype(int), probs)
    per_env = {}
    for env in ENVS.values():
        mask = env_arr == env
        if mask.sum():
            per_env[env] = accuracy_score(labels[mask].astype(int), preds[mask])
    return per_env, auc, probs, preds


def head_adapt(model, freeze_fn, head_fn, X_calib, y_calib, X_test, y_test,
               forward_fn=None, epochs=60, lr=5e-3, bs=64, label=""):
    """Generic head fine-tuning with standard BCE (pos_weight=1.0)."""
    m = deepcopy(model)
    freeze_fn(m)
    head_params = head_fn(m)
    n_head = sum(p.numel() for p in head_params)
    print(f"    {label} head params: {n_head}")

    opt  = torch.optim.Adam(head_params, lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([HEAD_POS_WEIGHT]))

    ds = TensorDataset(torch.from_numpy(X_calib),
                       torch.from_numpy(y_calib.astype(np.float32)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    fwd = forward_fn or (lambda mod, xb: mod(xb))

    m.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(fwd(m, xb), yb).backward()
            opt.step()

    m.eval()
    with torch.no_grad():
        logits = fwd(m, torch.from_numpy(X_test))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    y_np  = y_test.astype(int)
    return (accuracy_score(y_np, preds), f1_score(y_np, preds),
            roc_auc_score(y_np, probs), confusion_matrix(y_np, preds).tolist(),
            n_head, m)


def head_adapt_jiang(m_orig, cir_calib, aux_calib, y_calib,
                     cir_test, aux_test, y_test, epochs=60, lr=5e-3, bs=64):
    m = deepcopy(m_orig)
    for p in m.parameters(): p.requires_grad_(False)
    head_params = (list(m.params_mlp.parameters()) +
                   list(m.classifier.parameters()))
    for p in head_params: p.requires_grad_(True)
    n_head = sum(p.numel() for p in head_params)
    print(f"    MS-CNN-SA head params: {n_head}")

    opt  = torch.optim.Adam(head_params, lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([HEAD_POS_WEIGHT]))
    ds   = TensorDataset(torch.from_numpy(cir_calib),
                         torch.from_numpy(aux_calib),
                         torch.from_numpy(y_calib.astype(np.float32)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    m.train()
    for _ in range(epochs):
        for cb, ab, yb in loader:
            opt.zero_grad()
            crit(m(cb, ab), yb).backward()
            opt.step()

    m.eval()
    with torch.no_grad():
        logits = m(torch.from_numpy(cir_test), torch.from_numpy(aux_test))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    y_np  = y_test.astype(int)
    return (accuracy_score(y_np, preds), f1_score(y_np, preds),
            roc_auc_score(y_np, probs), n_head)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cross-Dataset Eval — ECA-UWB backbone: w+=1.0 (ecauwb_full.pt)")
    print("Head fine-tuning: standard BCE (pos_weight=1.0) for all models")
    print("=" * 60)

    # ── Load VNU ──────────────────────────────────────────────────────────────
    labels, X57, cir_jian, aux8, env_arr = load_all_vnu()
    print(f"Total: {len(labels)} | LOS={int((labels==0).sum())} "
          f"NLOS={int((labels==1).sum())}")

    idx = np.arange(len(labels))
    idx_calib, idx_test = train_test_split(
        idx, test_size=0.80, stratify=labels, random_state=SEED)
    y_calib, y_test = labels[idx_calib], labels[idx_test]
    env_test = env_arr[idx_test]
    print(f"Calib: {len(idx_calib)} | Test: {len(idx_test)}")

    # ── Scalers ───────────────────────────────────────────────────────────────
    with open(ECAUWB_SCALER,  "rb") as f: eca_sc   = pickle.load(f)
    with open(WU2024_SCALER,  "rb") as f: wu_sc    = pickle.load(f)
    with open(JIANG_CIR_SCL,  "rb") as f: j_cir_sc = pickle.load(f)
    with open(JIANG_AUX_SCL,  "rb") as f: j_aux_sc = pickle.load(f)

    Xeca = eca_sc.transform(X57).astype(np.float32)
    Xwu  = wu_sc.transform(X57).astype(np.float32)
    Xcj  = j_cir_sc.transform(cir_jian).astype(np.float32)
    Xaj  = j_aux_sc.transform(aux8).astype(np.float32)

    Xeca_c, Xeca_t = Xeca[idx_calib], Xeca[idx_test]
    Xwu_c,  Xwu_t  = Xwu[idx_calib],  Xwu[idx_test]
    Xcj_c,  Xcj_t  = Xcj[idx_calib],  Xcj[idx_test]
    Xaj_c,  Xaj_t  = Xaj[idx_calib],  Xaj[idx_test]

    results = {
        "VNU_total_samples": int(len(labels)),
        "VNU_calib_samples": int(len(idx_calib)),
        "VNU_test_samples":  int(len(idx_test)),
        "protocol": "20% calibration / 80% test (stratified seed=42)",
        "backbone_objective": "ECA-UWB: w+=1.0 (ecauwb_full.pt); SA-TinyML: w+=1.0; MS-CNN-SA: w+=1.0",
        "head_objective": f"All models: standard BCE (pos_weight={HEAD_POS_WEIGHT})",
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 1. ECA-UWB  (w+=1.0 backbone)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60 + "\nECA-UWB (w+=1.0 backbone)\n" + "=" * 60)
    eca_model = ECAUWBNet()
    eca_model.load_state_dict(torch.load(ECAUWB_MODEL, map_location="cpu"))
    eca_model.eval()
    print(f"  Loaded: {ECAUWB_MODEL}")

    # Direct transfer (whole dataset → AUC; test set → per-env acc)
    with torch.no_grad():
        logits_full = eca_model(torch.from_numpy(Xeca)).numpy()
    pe_full, auc_full, _, _ = _per_env_acc_auc(logits_full, labels, env_arr, is_logits=True)

    with torch.no_grad():
        logits_test = eca_model(torch.from_numpy(Xeca_t)).numpy()
    pe_test, _, _, _ = _per_env_acc_auc(logits_test, y_test, env_test, is_logits=True)

    print(f"Direct transfer AUC (all 9000): {auc_full:.4f}")
    for e, a in pe_test.items():
        print(f"  {e}: {a*100:.2f}%")

    # Head adaptation
    print("\nHead-adapting ECA-UWB ...")
    def eca_freeze(m):
        for p in m.cir_branch.parameters(): p.requires_grad_(False)
        for p in m.aux_branch.parameters():  p.requires_grad_(False)
    def eca_head(m):
        return list(m.fusion.parameters()) + list(m.classifier.parameters())

    ha_acc, ha_f1, ha_auc, ha_cm, _, _ = head_adapt(
        eca_model, eca_freeze, eca_head,
        Xeca_c, y_calib, Xeca_t, y_test, label="ECA-UWB")
    print(f"  Adapted: Acc={ha_acc*100:.2f}%  F1={ha_f1*100:.2f}%  AUC={ha_auc:.4f}")
    print(f"  CM: {ha_cm}")

    results["ECA-UWB"] = {
        "zero_shot": {
            "per_env_acc": {k: round(v*100, 2) for k, v in pe_test.items()},
            "overall_auc": round(auc_full, 4),
        },
        "head_adapted": {
            "acc": round(ha_acc*100, 2),
            "f1":  round(ha_f1*100, 2),
            "auc": round(ha_auc, 4),
            "cm":  ha_cm,
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 2. SA-TinyML
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60 + "\nSA-TinyML\n" + "=" * 60)
    wu_model = Wu2024Net(in_dim=57)
    wu_model.load_state_dict(torch.load(WU2024_MODEL, map_location="cpu"))
    wu_model.eval()

    with torch.no_grad():
        logits_wu = wu_model(torch.from_numpy(Xwu)).numpy()
    pe_wu_full, auc_wu, _, _ = _per_env_acc_auc(logits_wu, labels, env_arr, is_logits=True)

    with torch.no_grad():
        logits_wu_t = wu_model(torch.from_numpy(Xwu_t)).numpy()
    pe_wu_test, _, _, _ = _per_env_acc_auc(logits_wu_t, y_test, env_test, is_logits=True)

    print(f"Direct transfer AUC: {auc_wu:.4f}")
    for e, a in pe_wu_test.items():
        print(f"  {e}: {a*100:.2f}%")

    print("\nHead-adapting SA-TinyML ...")
    def wu_freeze(m):
        for p in m.parameters(): p.requires_grad_(False)
    def wu_head(m):
        heads = [m.q_proj, m.k_proj, m.v_proj, m.bn_q, m.bn_k, m.bn_v, m.clf]
        params = []
        for h in heads:
            for p in h.parameters():
                p.requires_grad_(True); params.append(p)
        return params

    wu_acc, wu_f1, wu_auc, wu_cm, wu_n_head, _ = head_adapt(
        wu_model, wu_freeze, wu_head,
        Xwu_c, y_calib, Xwu_t, y_test, label="SA-TinyML")
    print(f"  Adapted: Acc={wu_acc*100:.2f}%  F1={wu_f1*100:.2f}%  AUC={wu_auc:.4f}")

    results["SA-TinyML"] = {
        "zero_shot": {
            "per_env_acc": {k: round(v*100, 2) for k, v in pe_wu_test.items()},
            "overall_auc": round(auc_wu, 4),
        },
        "head_adapted": {
            "acc": round(wu_acc*100, 2),
            "f1":  round(wu_f1*100, 2),
            "auc": round(wu_auc, 4),
            "n_params": wu_n_head,
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MS-CNN-SA
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60 + "\nMS-CNN-SA\n" + "=" * 60)
    jian_model = Jiang2026Net(cir_len=63, n_aux=8)
    jian_model.load_state_dict(torch.load(JIANG_MODEL, map_location="cpu"))
    jian_model.eval()

    with torch.no_grad():
        logits_j = jian_model(torch.from_numpy(Xcj), torch.from_numpy(Xaj)).numpy()
    pe_j_full, auc_j, _, _ = _per_env_acc_auc(logits_j, labels, env_arr, is_logits=True)

    with torch.no_grad():
        logits_jt = jian_model(torch.from_numpy(Xcj_t), torch.from_numpy(Xaj_t)).numpy()
    pe_j_test, _, _, _ = _per_env_acc_auc(logits_jt, y_test, env_test, is_logits=True)

    print(f"Direct transfer AUC: {auc_j:.4f}")
    for e, a in pe_j_test.items():
        print(f"  {e}: {a*100:.2f}%")

    print("\nHead-adapting MS-CNN-SA ...")
    j_acc, j_f1, j_auc, j_n_head = head_adapt_jiang(
        jian_model, Xcj_c, Xaj_c, y_calib, Xcj_t, Xaj_t, y_test)
    print(f"  Adapted: Acc={j_acc*100:.2f}%  F1={j_f1*100:.2f}%  AUC={j_auc:.4f}")

    results["MS-CNN-SA"] = {
        "zero_shot": {
            "per_env_acc": {k: round(v*100, 2) for k, v in pe_j_test.items()},
            "overall_auc": round(auc_j, 4),
        },
        "head_adapted": {
            "acc": round(j_acc*100, 2),
            "f1":  round(j_f1*100, 2),
            "auc": round(j_auc, 4),
            "n_params": j_n_head,
        },
    }

    # ── Save & print ─────────────────────────────────────────────────────────
    out_path = LOG_DIR / "cross_dataset_vnu_bce10.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print(f"\n{'='*60}")
    print("TABLE IV (standardised — w+=1.0 backbone + head)")
    print(f"{'='*60}")
    hdr = f"{'Method':<16} {'Room%':>7} {'Hall%':>7} {'Lobby%':>7} {'AUC':>7} "
    hdr += f"{'HeadPar':>8} {'HA-Acc%':>8} {'HA-F1%':>7}"
    print(hdr); print("-" * len(hdr))
    for mname, res in [("SA-TinyML", results["SA-TinyML"]),
                       ("MS-CNN-SA", results["MS-CNN-SA"]),
                       ("ECA-UWB",   results["ECA-UWB"])]:
        zs = res["zero_shot"]; ha = res["head_adapted"]
        print(f"{mname:<16} "
              f"{zs['per_env_acc'].get('Room',0):>7.2f} "
              f"{zs['per_env_acc'].get('Hallway',0):>7.2f} "
              f"{zs['per_env_acc'].get('Lobby',0):>7.2f} "
              f"{zs['overall_auc']:>7.4f} "
              f"{ha.get('n_params',643):>8} "
              f"{ha['acc']:>8.2f} "
              f"{ha['f1']:>7.2f}")


if __name__ == "__main__":
    main()
