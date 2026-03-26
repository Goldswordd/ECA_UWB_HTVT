"""
Data loading, CIR window extraction, feature engineering, and normalization.

Supports:
  - Ewine dataset (7 CSV files, 42 000 samples, 1016-sample CIR)
  - OIUD dataset  (add loader below when format is known)

The output of build_dataset() is identical in shape to what
the DWM1001 tag sends over serial, so the same scaler and model
can be used for both offline training and live inference.
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1.  Low-level helpers
# ─────────────────────────────────────────────────────────────

def _extract_cir_window(cir_full: np.ndarray, fp_idx: int) -> np.ndarray:
    """
    Extract a 50-sample window centred around first-path index.
      window = cir_full[fp_idx - CIR_WIN_PRE : fp_idx + CIR_WIN_POST]
    Zero-pad if the window exceeds array bounds.
    """
    start = int(fp_idx) - config.CIR_WIN_PRE
    end   = int(fp_idx) + config.CIR_WIN_POST
    n     = len(cir_full)

    padded = np.zeros(config.CIR_LEN, dtype=np.float32)
    src_s = max(start, 0)
    src_e = min(end,   n)
    dst_s = src_s - start
    dst_e = dst_s + (src_e - src_s)
    padded[dst_s:dst_e] = cir_full[src_s:src_e]
    return padded


def _compute_features(row_diag: dict, cir_win: np.ndarray) -> np.ndarray:
    """
    Compute the 11 global features that match what DWM1001 sends.

    row_diag keys: FP_IDX, FP_AMP1, FP_AMP2, FP_AMP3,
                   STDEV_NOISE, CIR_PWR, MAX_NOISE, RXPACC
    """
    eps = 1e-9
    rxpacc = float(row_diag["RXPACC"]) + eps

    fp_pwr = (row_diag["FP_AMP1"]**2 +
              row_diag["FP_AMP2"]**2 +
              row_diag["FP_AMP3"]**2) / rxpacc**2

    rss    = float(row_diag["CIR_PWR"]) / rxpacc**2
    fp_rss = fp_pwr / (rss + eps)

    noise_std = float(row_diag["STDEV_NOISE"]) / rxpacc
    max_noise = float(row_diag["MAX_NOISE"])    / rxpacc

    snr_linear = rss / (noise_std + eps)
    snr_db     = 10.0 * np.log10(snr_linear + eps)

    # Features derived from the 50-sample CIR window
    # cir_win must already be the extracted 50-sample window
    assert len(cir_win) == config.CIR_LEN, \
        f"Expected {config.CIR_LEN}-sample window, got {len(cir_win)}"
    cir_norm     = cir_win / rxpacc              # normalize by accumulations
    cir_peak     = float(np.max(cir_norm))
    cir_kurt     = float(kurtosis(cir_norm, fisher=True))  # excess kurtosis

    # Delay spread (assume 1 sample = 1 ns for DW1000 at ~1 GHz)
    power_prof   = cir_norm ** 2
    total_pwr    = float(np.sum(power_prof)) + eps
    time_axis    = np.arange(config.CIR_LEN, dtype=np.float32)
    mean_del     = float(np.sum(time_axis * power_prof) / total_pwr)
    rms_del      = float(np.sqrt(np.sum((time_axis - mean_del)**2 * power_prof) / total_pwr))

    # Rise time = position of peak relative to window start
    peak_pos = float(np.argmax(cir_norm))

    return np.array([
        fp_pwr,       # 0
        rss,          # 1
        fp_rss,       # 2  ← strongest single feature
        noise_std,    # 3
        max_noise,    # 4
        snr_db,       # 5
        cir_peak,     # 6
        cir_kurt,     # 7
        rms_del,      # 8
        mean_del,     # 9
        peak_pos,     # 10
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# 2.  Ewine dataset loader
# ─────────────────────────────────────────────────────────────

def load_ewine(ewine_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all CSVs in ewine_dir and return:
      cir_windows : (N, 50)   float32 — CIR window / RXPACC
      features    : (N, 11)   float32 — engineered global features
      labels      : (N,)      int8    — 0=LOS, 1=NLOS

    The CSV column layout is documented in config.py.
    """
    ewine_dir = Path(ewine_dir or config.EWINE_DIR)
    csv_files = sorted(ewine_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {ewine_dir}.\n"
            "Download from: https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set"
        )

    all_cir, all_feat, all_labels = [], [], []
    diag_col_idx = config.EWINE_DIAG_COLS
    cir_start    = config.EWINE_CIR_START

    for f in csv_files:
        print(f"  Loading {f.name} ...", end=" ", flush=True)
        df = pd.read_csv(f, header=0)
        arr = df.values  # shape (N, 1031)

        labels  = arr[:, 0].astype(np.int8)
        cir_all = arr[:, cir_start:].astype(np.float32)   # (N, 1016)

        cir_wins, feats = [], []
        for i in range(len(arr)):
            row    = arr[i]
            fp_idx = int(row[diag_col_idx["FP_IDX"]])
            rxpacc = float(row[diag_col_idx["RXPACC"]])

            cir_win = _extract_cir_window(cir_all[i], fp_idx)
            # Normalize window by RXPACC (as per Ewine paper recommendation)
            cir_win = cir_win / (rxpacc + 1e-9)

            diag = {k: float(row[v]) for k, v in diag_col_idx.items()}
            feat = _compute_features(diag, cir_win)   # uses normalized 50-sample window

            cir_wins.append(cir_win)
            feats.append(feat)

        all_cir.append(np.array(cir_wins,  dtype=np.float32))
        all_feat.append(np.array(feats,    dtype=np.float32))
        all_labels.append(labels)
        print(f"  {len(arr)} rows, LOS={int((labels==0).sum())}, NLOS={int((labels==1).sum())}")

    cir_windows = np.concatenate(all_cir,    axis=0)
    features    = np.concatenate(all_feat,   axis=0)
    labels      = np.concatenate(all_labels, axis=0)

    print(f"\nEwine total: {len(labels)} samples | "
          f"LOS={int((labels==0).sum())} NLOS={int((labels==1).sum())}")
    return cir_windows, features, labels


# ─────────────────────────────────────────────────────────────
# 3.  OIUD dataset loader  (stub — fill in when format known)
# ─────────────────────────────────────────────────────────────

def load_oiud(oiud_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Placeholder for OIUD dataset.
    Return same format as load_ewine(): (cir_windows, features, labels)
    """
    oiud_dir = Path(oiud_dir or config.OIUD_DIR)
    files = sorted(oiud_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No OIUD CSV files found in {oiud_dir}.\n"
            "Please provide the dataset and update this loader."
        )
    # TODO: implement when OIUD format is confirmed
    raise NotImplementedError("OIUD loader not yet implemented — see comment above.")


# ─────────────────────────────────────────────────────────────
# 4.  Clean outliers
# ─────────────────────────────────────────────────────────────

def remove_outliers(cir: np.ndarray, feat: np.ndarray,
                    labels: np.ndarray, z_thresh: float = 6.0):
    """
    Drop samples where ANY feature is more than z_thresh std devs
    from its class mean (catches sensor glitches in Ewine).
    """
    keep = np.ones(len(labels), dtype=bool)
    for cls in [0, 1]:
        idx  = labels == cls
        mu   = feat[idx].mean(0)
        sig  = feat[idx].std(0) + 1e-9
        z    = np.abs((feat - mu) / sig)
        keep &= ~((labels == cls) & (z.max(1) > z_thresh))

    n_removed = int((~keep).sum())
    if n_removed:
        print(f"  Removed {n_removed} outlier samples (z>{z_thresh})")
    return cir[keep], feat[keep], labels[keep]


# ─────────────────────────────────────────────────────────────
# 5.  Normalization & splitting
# ─────────────────────────────────────────────────────────────

def fit_scaler(feat_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(feat_train)
    return scaler


def save_scaler(scaler: StandardScaler, path: Path | None = None):
    path = Path(path or config.SCALER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved → {path}")


def load_scaler(path: Path | None = None) -> StandardScaler:
    path = Path(path or config.SCALER_PATH)
    with open(path, "rb") as f:
        return pickle.load(f)


def split_data(cir: np.ndarray, feat: np.ndarray, labels: np.ndarray,
               val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT,
               seed=config.SEED):
    """Stratified train / val / test split."""
    test_ratio = test_split
    val_ratio  = val_split / (1.0 - test_ratio)

    cir_tv,  cir_test,  feat_tv,  feat_test,  y_tv,  y_test = \
        train_test_split(cir, feat, labels,
                         test_size=test_ratio, stratify=labels,
                         random_state=seed)
    cir_tr, cir_val, feat_tr, feat_val, y_tr, y_val = \
        train_test_split(cir_tv, feat_tv, y_tv,
                         test_size=val_ratio, stratify=y_tv,
                         random_state=seed)

    print(f"  Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")
    return (cir_tr, feat_tr, y_tr,
            cir_val, feat_val, y_val,
            cir_test, feat_test, y_test)


# ─────────────────────────────────────────────────────────────
# 6.  Master pipeline
# ─────────────────────────────────────────────────────────────

def build_dataset(use_oiud: bool = False):
    """
    Full preprocessing pipeline.
    Returns train/val/test splits with CIR windows (raw, NOT scaled —
    model normalizes internally) and feature arrays (StandardScaler fitted
    on train, applied to val/test).

    Also saves the fitted scaler to disk for inference.
    """
    print("=" * 60)
    print("Loading datasets …")
    cir_e, feat_e, lab_e = load_ewine()

    if use_oiud:
        try:
            cir_o, feat_o, lab_o = load_oiud()
            cir_all  = np.concatenate([cir_e, cir_o],  axis=0)
            feat_all = np.concatenate([feat_e, feat_o], axis=0)
            lab_all  = np.concatenate([lab_e, lab_o],  axis=0)
            print(f"\nCombined total: {len(lab_all)} samples")
        except (FileNotFoundError, NotImplementedError) as ex:
            print(f"  OIUD skipped: {ex}")
            cir_all, feat_all, lab_all = cir_e, feat_e, lab_e
    else:
        cir_all, feat_all, lab_all = cir_e, feat_e, lab_e

    print("\nRemoving outliers …")
    cir_all, feat_all, lab_all = remove_outliers(cir_all, feat_all, lab_all)

    print("\nSplitting …")
    splits = split_data(cir_all, feat_all, lab_all)
    (cir_tr, feat_tr, y_tr,
     cir_val, feat_val, y_val,
     cir_te, feat_te, y_te) = splits

    print("\nFitting StandardScaler on train features …")
    scaler = fit_scaler(feat_tr)
    save_scaler(scaler)

    feat_tr  = scaler.transform(feat_tr).astype(np.float32)
    feat_val = scaler.transform(feat_val).astype(np.float32)
    feat_te  = scaler.transform(feat_te).astype(np.float32)

    # Add channel dim for 1D CNN: (N, 50) → (N, 50, 1)
    cir_tr  = cir_tr[..., np.newaxis]
    cir_val = cir_val[..., np.newaxis]
    cir_te  = cir_te[..., np.newaxis]

    print("=" * 60)
    return (cir_tr, feat_tr, y_tr,
            cir_val, feat_val, y_val,
            cir_te, feat_te, y_te,
            scaler)


# ─────────────────────────────────────────────────────────────
# 7.  Single-sample preparation  (used by inference_pi.py)
# ─────────────────────────────────────────────────────────────

def prepare_single(cir_win_raw: np.ndarray,
                   feat_raw: np.ndarray,
                   scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare one sample from live DWM1001 serial data.

    cir_win_raw : (50,) float32 — CIR window already divided by RXPACC
    feat_raw    : (11,) float32 — 11 global features (raw, unscaled)
    scaler      : fitted StandardScaler

    Returns:
      cir_input  : (1, 50, 1)
      feat_input : (1, 11)
    """
    cir_in  = cir_win_raw.astype(np.float32)[np.newaxis, :, np.newaxis]
    feat_sc = scaler.transform(feat_raw.reshape(1, -1)).astype(np.float32)
    return cir_in, feat_sc


if __name__ == "__main__":
    # Quick test — run from project root:  python preprocess.py
    splits = build_dataset(use_oiud=False)
    cir_tr, feat_tr, y_tr = splits[0], splits[1], splits[2]
    print(f"\nCIR shape : {cir_tr.shape}")
    print(f"Feat shape: {feat_tr.shape}")
    print(f"Labels    : {np.bincount(y_tr.astype(int))}")
