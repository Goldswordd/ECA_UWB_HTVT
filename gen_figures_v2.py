"""
ECA-UWB Paper — Publication-Quality Figures (v2)
Follows NeurIPS 2025 aesthetic from PaperBanana style guide:
  • White backgrounds, high contrast
  • Soft-Tech pastel palette, rounded-rect process nodes
  • Open-spine axes, light-grey dashed grid behind data
  • Sans-serif throughout, 300 DPI output
"""

import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

ROOT   = Path(__file__).parent
FIG    = ROOT / "figures"
FIG.mkdir(exist_ok=True)

UWBNLOS = Path(__file__).parent.parent / "uwb_nlos"
sys.path.insert(0, str(UWBNLOS))
import config

# ── Global RC ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.grid":          False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# ── Palette (NeurIPS "Soft Tech") ─────────────────────────────────────────────
# CIR branch: Ice-blue family
CIR_DARK   = "#1a6fa8"   # deep blue
CIR_MID    = "#4fa3d1"   # mid blue
CIR_LIGHT  = "#d6eaf8"   # pale-blue zone fill

# Aux branch: Teal/sage family
AUX_DARK   = "#1a7a6b"   # deep teal
AUX_MID    = "#3dbea8"   # mid teal
AUX_LIGHT  = "#d0f0eb"   # pale-teal zone fill

# ECA highlight: warm coral (trainable special module)
ECA_COLOR  = "#e07b54"   # warm orange-coral
ECA_DARK   = "#c05a35"

# Fusion: soft purple
FUS_COLOR  = "#7c5cbf"   # medium purple
FUS_LIGHT  = "#ede7ff"

# Classifier: slate
CLF_COLOR  = "#4a5568"
CLF_LIGHT  = "#edf2f7"

# Input/Output: dark navy header
HDR_COLOR  = "#2d3748"
HDR_LIGHT  = "#e2e8f0"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rounded_box(ax, cx, cy, w, h, label,
                fc="#4fa3d1", ec="white", lw=1.0,
                txt_color="white", fontsize=8.0, bold=False,
                radius=0.012, alpha=1.0, zorder=3):
    """Draw a rounded-rectangle process node."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color=txt_color, zorder=zorder + 1,
            fontweight="bold" if bold else "normal",
            linespacing=1.35)


def zone_rect(ax, x0, y0, x1, y1, fc, ec, lw=1.0, ls="--", radius=0.02, alpha=0.25, zorder=0):
    """Draw a background zone with dashed border."""
    w, h = x1 - x0, y1 - y0
    patch = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder
    )
    ax.add_patch(patch)


def arrow_v(ax, x, y0, y1, color="#555", lw=1.1, ms=7):
    """Straight vertical arrow."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms), zorder=4)


def arrow_diag(ax, x0, y0, x1, y1, color="#555", lw=1.1, ms=7):
    """Diagonal arrow."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms), zorder=4)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Architecture  (wide, for figure* two-column)
# ─────────────────────────────────────────────────────────────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_xlim(0, 20)
    ax.set_ylim(-0.4, 7.2)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    bw, bh  = 5.4, 0.50   # block width / height
    cx_cir  = 4.2          # CIR branch X centre
    cx_aux  = 15.8         # Aux branch X centre
    cx_mid  = 10.0         # Centre of figure

    # ── Zone backgrounds (start below input, include branch label row) ────────
    zone_rect(ax, 0.6, 0.55, 7.0, 6.90, fc=CIR_LIGHT, ec=CIR_MID, lw=1.3,
              alpha=0.30, zorder=0)
    zone_rect(ax, 13.0, 0.55, 19.4, 6.90, fc=AUX_LIGHT, ec=AUX_MID, lw=1.3,
              alpha=0.30, zorder=0)

    # ── Input (sits above zones) ───────────────────────────────────────────────
    rounded_box(ax, cx_mid, 7.00, 16.0, 0.52,
                "Input  x ∈ ℝ⁵⁷  =  [ CIR window (50 samples)  ⊕  Channel diagnostics (7) ]",
                fc=HDR_COLOR, ec="white", txt_color="white", fontsize=9.0, bold=True,
                lw=1.2, radius=0.015)

    # Vertical arrows from input to top of each zone
    arrow_v(ax, cx_cir, 6.74, 6.68, color="#888")
    arrow_v(ax, cx_aux, 6.74, 6.68, color="#888")

    # ── Branch labels (inside zone, clearly separated from input) ──────────────
    ax.text(cx_cir, 6.52, "CIR Branch", ha="center", fontsize=9.5,
            fontweight="bold", color=CIR_DARK)
    ax.text(cx_aux, 6.52, "Auxiliary Branch", ha="center", fontsize=9.5,
            fontweight="bold", color=AUX_DARK)

    # ── CIR Branch blocks ────────────────────────────────────────────────────
    cir_blocks = [
        (6.10, "Reshape  →  (B, 1, 50)",                       CIR_DARK),
        (5.47, "Conv1d(1→16, k=5) + BN + ReLU → (B, 16, 50)", CIR_MID),
        (4.84, "MaxPool(2)  →  (B, 16, 25)",                   CIR_MID),
    ]
    for y, lbl, c in cir_blocks:
        rounded_box(ax, cx_cir, y, bw, bh, lbl, fc=c, ec="white",
                    txt_color="white", fontsize=7.5, lw=0.9, radius=0.012)

    # ECA module — highlighted warm-coral
    eca_y = 4.21
    eca_box = FancyBboxPatch(
        (cx_cir - bw / 2, eca_y - bh / 2), bw, bh,
        boxstyle="round,pad=0.014",
        facecolor=ECA_COLOR, edgecolor=ECA_DARK,
        linewidth=1.6, zorder=3
    )
    ax.add_patch(eca_box)
    ax.text(cx_cir, eca_y,
            "⚡ ECA Module  (k=3, no bias)  —  only 3 params",
            ha="center", va="center", fontsize=8.0,
            color="white", fontweight="bold", zorder=4)

    cir_blocks2 = [
        (3.58, "Conv1d(16→16, k=3) + BN + ReLU → (B, 16, 25)", CIR_MID),
        (2.95, "Global Average Pool  →  f_cir ∈ ℝ¹⁶",           CIR_DARK),
    ]
    for y, lbl, c in cir_blocks2:
        rounded_box(ax, cx_cir, y, bw, bh, lbl, fc=c, ec="white",
                    txt_color="white", fontsize=7.5, lw=0.9, radius=0.012)

    # CIR arrows
    arrow_v(ax, cx_cir, 6.42, 6.10+bh/2)
    for y0, y1 in [(6.10-bh/2, 5.47+bh/2),
                   (5.47-bh/2, 4.84+bh/2),
                   (4.84-bh/2, eca_y+bh/2),
                   (eca_y-bh/2, 3.58+bh/2),
                   (3.58-bh/2, 2.95+bh/2)]:
        arrow_v(ax, cx_cir, y0, y1, color="#555")

    # ── Auxiliary Branch blocks ───────────────────────────────────────────────
    aux_blocks = [
        (6.10, "Linear(7→32) + ReLU  →  (B, 32)",  AUX_MID),
        (5.47, "Linear(32→16) + ReLU  →  (B, 16)", AUX_MID),
        (4.84, "f_aux  ∈  ℝ¹⁶",                    AUX_DARK),
    ]
    for y, lbl, c in aux_blocks:
        rounded_box(ax, cx_aux, y, bw, bh, lbl, fc=c, ec="white",
                    txt_color="white", fontsize=7.5, lw=0.9, radius=0.012)

    arrow_v(ax, cx_aux, 6.42, 6.10+bh/2)
    for y0, y1 in [(6.10-bh/2, 5.47+bh/2),
                   (5.47-bh/2, 4.84+bh/2)]:
        arrow_v(ax, cx_aux, y0, y1, color="#555")

    # ── Gated Fusion ─────────────────────────────────────────────────────────
    fus_y = 1.82
    rounded_box(ax, cx_mid, fus_y, 15.0, 0.60,
                "Gated Fusion :   g = σ( Wg · [f_cir ; f_aux] )     "
                "f_fused = g₁ · f_cir  +  g₂ · f_aux",
                fc=FUS_COLOR, ec="white", txt_color="white",
                fontsize=8.5, bold=True, lw=1.2, radius=0.015)

    # branches → fusion
    arrow_diag(ax, cx_cir,  2.95-bh/2, cx_mid - 6.4, fus_y+0.30, color="#555")
    arrow_diag(ax, cx_aux,  4.84-bh/2, cx_mid + 6.4, fus_y+0.30, color="#555")

    # ── Classifier ───────────────────────────────────────────────────────────
    clf_y = 1.06
    rounded_box(ax, cx_mid, clf_y, 15.0, 0.60,
                "Linear(16→32)  +  ReLU  +  Dropout(0.3)   →   Linear(32→1)",
                fc=CLF_COLOR, ec="white", txt_color="white",
                fontsize=8.5, lw=1.2, radius=0.015)
    arrow_v(ax, cx_mid, fus_y - 0.30, clf_y + 0.30, color="#555")

    # ── Output ───────────────────────────────────────────────────────────────
    out_y = 0.28
    rounded_box(ax, cx_mid, out_y, 3.0, 0.40,
                "ŷ  (logit)", fc=HDR_COLOR, ec="white",
                txt_color="white", fontsize=9.0, bold=True, lw=1.2)
    arrow_v(ax, cx_mid, clf_y - 0.30, out_y + 0.20, color="#555")

    # ── Parameter summary — small inset box, right side outside zones ─────────
    px, py = 19.6, 4.80   # anchor (top-right corner of inset)
    iw, ih = 2.8, 3.4
    param_bg = FancyBboxPatch(
        (px - iw, py - ih), iw, ih,
        boxstyle="round,pad=0.06",
        facecolor="#fafafa", edgecolor="#bbbbbb",
        linewidth=0.9, zorder=2, clip_on=False
    )
    ax.add_patch(param_bg)

    param_lines = [
        ("─ Params ─",           True),
        ("Conv1(1→16)+BN   160", False),
        ("ECA (k=3)           3  ←", True),
        ("Conv2(16→16)+BN  784", False),
        ("Aux MLP          784", False),
        ("Gated fusion      66", False),
        ("Classifier       577", False),
        ("─" * 19,                False),
        ("Total          2,374", True),
    ]
    for i, (line, bold) in enumerate(param_lines):
        ax.text(px - 0.15, py - 0.22 - i * 0.36, line,
                ha="right", va="top", fontsize=6.6,
                color=ECA_DARK if bold else "#3d3d3d",
                fontweight="bold" if bold else "normal",
                family="monospace", zorder=5, clip_on=False)

    ax.set_title("ECA-UWB Architecture  (Total: 2,374 parameters)",
                 fontsize=10.5, pad=8, fontweight="bold", color="#2d3748")

    plt.tight_layout(pad=0.5)
    plt.savefig(FIG / "fig1_architecture.png", bbox_inches="tight", dpi=300)
    plt.savefig(FIG / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close()
    print("  ✓ fig1_architecture")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Confusion Matrix (publication quality)
# ─────────────────────────────────────────────────────────────────────────────

def fig_confusion_matrix():
    TN, FP, FN, TP = 2974, 175, 320, 2812
    cm       = np.array([[TN, FP], [FN, TP]])
    row_tot  = cm.sum(axis=1, keepdims=True)
    cm_pct   = cm / row_tot * 100

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Blues but with a custom light minimum so 0% still has visible fill
    cmap = plt.cm.Blues
    im   = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100,
                     aspect="equal")

    # Cell annotations
    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            pct   = cm_pct[i, j]
            count = cm[i, j]
            label = cell_labels[i][j]
            txt_color = "white" if pct > 55 else "#1a2744"
            ax.text(j, i,
                    f"{label}\n{count:,}\n{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=12, color=txt_color,
                    fontweight="bold", linespacing=1.5)

    # Thin white cell borders
    for i in range(3):
        ax.axhline(i - 0.5, color="white", lw=2.0)
        ax.axvline(i - 0.5, color="white", lw=2.0)

    # Axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted LOS", "Predicted NLOS"], fontsize=10)
    ax.set_yticklabels(["Actual LOS", "Actual NLOS"], fontsize=10)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Colorbar — minimal
    cbar = plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label("Row %", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    ax.set_title(
        "Confusion Matrix — eWINE Test Set\n"
        "Acc = 92.12 %    AUC = 97.60 %",
        fontsize=10, fontweight="bold", pad=14, color="#2d3748"
    )

    plt.tight_layout()
    plt.savefig(FIG / "fig8_confusion.png", bbox_inches="tight", dpi=300)
    plt.savefig(FIG / "fig8_confusion.pdf", bbox_inches="tight")
    plt.close()
    print("  ✓ fig8_confusion")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Per-Environment Accuracy (eWINE 7 envs)
# ─────────────────────────────────────────────────────────────────────────────

def fig_per_environment():
    import pandas as pd
    import pickle
    import torch
    from sklearn.metrics import accuracy_score, roc_auc_score
    from ecauwb_model import ECAUWBNet
    from preprocess import _extract_cir_window

    AUX_COLS = [1, 8, 6, 3, 4, 5, 7]
    MDL      = UWBNLOS / "models"

    HP = dict(cir_len=50, n_aux=7, ch1=16, ch2=16, k1=5, k2=3,
              eca_k=3, aux_hid=32, feat_dim=16, clf_hid=32, dropout=0.3)
    model = ECAUWBNet(**HP)
    model.load_state_dict(torch.load(MDL / "ecauwb_full.pt", map_location="cpu"))
    model.eval()

    with open(MDL / "ecauwb_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    fp_col = config.EWINE_DIAG_COLS["FP_IDX"]
    rx_col = config.EWINE_DIAG_COLS["RXPACC"]

    csv_files = sorted(config.EWINE_DIR.glob("*.csv"))
    env_names = [
        "Office 1", "Office 2", "Apartment",
        "Workshop", "Kitchen/\nLiving", "Bedroom", "Boiler\nRoom"
    ]
    if len(csv_files) != 7:
        env_names = [f.stem for f in csv_files]

    accs, los_rec, nlos_rec = [], [], []

    for f, name in zip(csv_files, env_names):
        arr     = pd.read_csv(f, header=0).values
        labels  = arr[:, 0].astype(int)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(float)

        rows = []
        for i in range(len(arr)):
            fp_idx  = int(arr[i, fp_col])
            rxpacc  = float(arr[i, rx_col]) + 1e-9
            cir_win = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc
            aux     = arr[i, AUX_COLS].astype(float)
            rows.append(np.concatenate([cir_win, aux]))

        X = np.array(rows, dtype=np.float32)
        X = scaler.transform(X).astype(np.float32)

        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()

        preds = (probs > 0.5).astype(int)
        acc   = accuracy_score(labels, preds)
        lr    = accuracy_score(labels[labels == 0], preds[labels == 0])
        nr    = accuracy_score(labels[labels == 1], preds[labels == 1])

        accs.append(acc * 100)
        los_rec.append(lr * 100)
        nlos_rec.append(nr * 100)
        print(f"  {name:22s}: Acc={acc*100:.2f}%  "
              f"LOS={lr*100:.2f}%  NLOS={nr*100:.2f}%")

    _draw_per_env(env_names, accs, los_rec, nlos_rec)


def _draw_per_env(env_names, accs, los_rec, nlos_rec):
    """Draw the per-environment grouped bar chart."""
    n   = len(env_names)
    x   = np.arange(n)
    w   = 0.24

    # NeurIPS pastel palette (colorblind-friendly)
    C_ACC  = "#4a90c4"   # steel blue  — Accuracy
    C_LOS  = "#3ab5a0"   # teal        — LOS Recall
    C_NLOS = "#e07b54"   # warm coral  — NLOS Recall

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Light grey dashed horizontal grid — drawn BEFORE bars (low zorder)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#c8c8c8", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    def draw_bars(offset, vals, color, label):
        bars = ax.bar(x + offset, vals, w,
                      label=label,
                      color=color, alpha=0.90,
                      edgecolor="#333333", linewidth=0.6,
                      zorder=3)
        return bars

    b1 = draw_bars(-w,  accs,     C_ACC,  "Accuracy")
    b2 = draw_bars(0,   los_rec,  C_LOS,  "LOS Recall")
    b3 = draw_bars(+w,  nlos_rec, C_NLOS, "NLOS Recall")

    # Value labels — small, above each bar
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=5.8, color="#333333", rotation=90)

    # Mean accuracy dashed reference line
    mean_acc = np.mean(accs)
    ax.axhline(mean_acc, color=C_ACC, lw=1.3, ls="--", alpha=0.70, zorder=2,
               label=f"Mean Acc ({mean_acc:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=8.5, ha="center")
    ax.set_ylabel("(%)", fontsize=10)
    ax.set_ylim(75, 108)

    # Open spine: remove top and right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.legend(fontsize=8.5, loc="lower right", ncol=2,
              frameon=True, framealpha=0.9, edgecolor="#cccccc")

    ax.set_title(
        "Per-Environment Performance on eWINE Test Set\n"
        "(ECA-UWB, trained on full training split)",
        fontsize=10, fontweight="bold", color="#2d3748", pad=8
    )

    plt.tight_layout()
    plt.savefig(FIG / "fig9_per_env.png", bbox_inches="tight", dpi=300)
    plt.savefig(FIG / "fig9_per_env.pdf", bbox_inches="tight")
    plt.close()
    print("  ✓ fig9_per_env")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating ECA-UWB figures (v2) …\n")
    fig_architecture()
    fig_confusion_matrix()
    print("\nPer-environment breakdown:")
    fig_per_environment()
    print(f"\n✓ All figures saved to {FIG}")
