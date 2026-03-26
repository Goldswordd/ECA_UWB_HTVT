"""
Centralized configuration for UWB NLOS Classifier.
Modify paths and hyperparameters here — nowhere else.
"""
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EWINE_DIR = Path.home() / "Documents/UWB-NLOS/LightMamba/1030/dataset"
OIUD_DIR  = DATA_DIR / "oiud"
MODEL_DIR = ROOT / "models"
LOG_DIR   = ROOT / "logs"

SCALER_PATH     = MODEL_DIR / "scaler.pkl"
KERAS_MODEL_PATH = MODEL_DIR / "best_model.keras"
TFLITE_PATH     = MODEL_DIR / "nlos_classifier.tflite"
TFLITE_INT8_PATH = MODEL_DIR / "nlos_classifier_int8.tflite"

# ──────────────────────────────────────────────
# CIR Window (must match DWM1001 firmware output)
#   DW1000 CIR window: FP_IDX - 2  →  FP_IDX + 47
#   = 50 samples total
# ──────────────────────────────────────────────
CIR_WIN_PRE  = 2    # samples before FP_IDX
CIR_WIN_POST = 48   # samples after  FP_IDX
CIR_LEN      = CIR_WIN_PRE + CIR_WIN_POST  # = 50

# ──────────────────────────────────────────────
# Ewine dataset column layout
#   Col  0       : NLOS label (1=NLOS, 0=LOS)
#   Col  1       : range (m)
#   Col  2       : FP_IDX
#   Col  3-5     : FP_AMP1, FP_AMP2, FP_AMP3
#   Col  6       : STDEV_NOISE
#   Col  7       : CIR_PWR
#   Col  8       : MAX_NOISE
#   Col  9       : RXPACC
#   Col  10      : CH
#   Col  11      : FRAME_LEN
#   Col  12      : PREAM_LEN
#   Col  13      : BITRATE
#   Col  14      : PRFR
#   Col  15-1030 : CIR[0]..CIR[1015]
# ──────────────────────────────────────────────
EWINE_LABEL_COL    = "NLOS"
EWINE_CIR_START    = 15         # first CIR column index (0-based)
EWINE_CIR_LEN      = 1016
EWINE_DIAG_COLS    = {
    "FP_IDX":      2,
    "FP_AMP1":     3,
    "FP_AMP2":     4,
    "FP_AMP3":     5,
    "STDEV_NOISE": 6,
    "CIR_PWR":     7,
    "MAX_NOISE":   8,
    "RXPACC":      9,
}

# ──────────────────────────────────────────────
# VNU Indoor Dataset (your own collection)
# Same diagnostic column indices as eWINE (cols 0–9 identical).
# Only difference: no 5 hardware-config padding cols → CIR starts at 10.
#
# Firmware output format (ss_initiator_datacollect.c):
#   Col 0       : NLOS label (0=LOS, 1=NLOS)  ← set g_label in firmware
#   Col 1       : RANGE   (mm, from TWR)
#   Col 2       : FP_IDX
#   Col 3–5     : FP_AMP1, FP_AMP2, FP_AMP3
#   Col 6       : STDEV_NOISE
#   Col 7       : CIR_PWR  (raw register, same as eWINE col 7)
#   Col 8       : MAX_NOISE
#   Col 9       : RXPACC
#   Col 10–109  : CIR magnitude [FP_IDX-5 … FP_IDX+94]  (100 samples)
# ──────────────────────────────────────────────
VNU_DIR          = Path.home() / "Documents/UWB-NLOS/VNU_dataset"
VNU_CIR_START    = 10       # only difference from eWINE (15)
VNU_CIR_LEN      = 100      # firmware collects 100; ECA-UWB uses first 50
VNU_DIAG_COLS    = EWINE_DIAG_COLS   # cols 1–9 identical → zero code change

# ──────────────────────────────────────────────
# 11 Global Features (computed from diagnostics + CIR window)
# These are the SAME features the DWM1001 should send
# in columns 51-61 of each serial line.
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    "fp_pwr",            # (FP_AMP1²+FP_AMP2²+FP_AMP3²) / RXPACC²
    "rss",               # CIR_PWR / RXPACC²
    "fp_rss_ratio",      # fp_pwr / rss  ← #1 NLOS discriminator
    "noise_std",         # STDEV_NOISE / RXPACC
    "max_noise",         # MAX_NOISE / RXPACC
    "snr_db",            # 10*log10(rss / noise_std)  (dB)
    "cir_peak_norm",     # max(CIR_window) / RXPACC
    "cir_kurtosis",      # Fisher kurtosis of CIR window
    "rms_delay",         # RMS delay spread (ns, assuming 1 sample=1 ns)
    "mean_excess_delay", # mean excess delay (ns)
    "rise_time",         # index of peak - 0  (relative to window start)
]
N_FEATURES = len(FEATURE_NAMES)  # 11

# ──────────────────────────────────────────────
# Live serial format from DWM1001 tag
#   Each line = one anchor measurement, CSV:
#   [anchor_id, cir_0..cir_49, feat_0..feat_10]
#   total columns = 1 + 50 + 11 = 62
#   Set SERIAL_HAS_ANCHOR_ID = False if no anchor_id column
# ──────────────────────────────────────────────
SERIAL_PORT          = "/dev/ttyACM0"
SERIAL_BAUD          = 115200
SERIAL_HAS_ANCHOR_ID = True   # first column is anchor ID string
SERIAL_CIR_START_COL = 1      # if SERIAL_HAS_ANCHOR_ID else 0
SERIAL_FEAT_START_COL = 51    # = SERIAL_CIR_START_COL + CIR_LEN

# ──────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────
SEED         = 42
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 1e-3
LR_MIN       = 1e-5
PATIENCE     = 15          # early stopping
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
DROPOUT      = 0.3

# CNN branch channels
CNN_FILTERS  = [16, 32, 32]
CNN_KERNELS  = [5,   3,  3]
CNN_POOL     = [2,   2,  0]   # 0 = no pool on last conv

# MLP branch hidden dims
MLP_HIDDEN   = [32, 16]

# Fusion head
FUSION_HIDDEN = 32
