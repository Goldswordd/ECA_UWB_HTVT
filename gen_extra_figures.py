"""
Generate extra figures:
  fig1  — architecture (wide version for figure* two-column)
  fig8  — confusion matrix heatmap (exact numbers from ablation full)
  fig9  — per-environment accuracy breakdown (eWINE 7 CSV files)
"""

import sys, json, pickle
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent
FIG  = ROOT / "figures"
FIG.mkdir(exist_ok=True)

UWBNLOS = Path(__file__).parent.parent / "uwb_nlos"
sys.path.insert(0, str(UWBNLOS))
import config

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── helpers ────────────────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, text, color="#4a90d9", text_color="white",
             fontsize=7.5, alpha=0.92, radius=0.015):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor="white",
                         linewidth=0.8, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=4,
            fontweight="bold" if color not in ("#eceff1", "#f8f9fa") else "normal")

def arrow(ax, x0, y0, x1, y1, color="#666"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=0.9, mutation_scale=7), zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Architecture (wide, for figure* two-column layout)
# ─────────────────────────────────────────────────────────────────────────────

def fig_architecture_wide():
    """
    Wide version: ~170 mm (7.16 in) for figure* in two-column IEEE paper.
    Layout: CIR branch (left) | central flow | Aux branch (right)
    """
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 18); ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")

    C1, C2 = "#1d3557", "#457b9d"   # CIR branch colors
    A1, A2 = "#1d6b5e", "#2a9d8f"   # Aux branch colors
    ECA_C  = "#e63946"              # ECA highlight
    FUS_C  = "#6c3483"              # fusion
    CLF_C  = "#343a40"              # classifier
    HDR_C  = "#495057"              # header boxes

    # ── Input ──────────────────────────────────────────────
    draw_box(ax,  9.0, 5.85, 8.0, 0.52,
             r"Input  x ∈ ℝ⁵⁷  =  [ CIR window (50 samples) | Channel diagnostics (7) ]",
             color=HDR_C, fontsize=8.5)

    # ── CIR branch label ───────────────────────────────────
    ax.text(3.8, 5.30, "CIR Branch", ha="center", fontsize=9,
            fontweight="bold", color=C2)

    # CIR blocks
    bw, bh = 4.6, 0.46
    cx = 3.8
    draw_box(ax, cx, 4.78, bw, bh, "Reshape  →  (B, 1, 50)", color=C1, fontsize=7.8)
    draw_box(ax, cx, 4.20, bw, bh, "Conv1d(1→16, k=5) + BN + ReLU  →  (B, 16, 50)", color=C2, fontsize=7.8)
    draw_box(ax, cx, 3.62, bw, bh, "MaxPool(2)  →  (B, 16, 25)", color=C2, fontsize=7.8)

    # ECA block (highlighted)
    eca = FancyBboxPatch((cx - bw/2, 3.10), bw, 0.46,
                          boxstyle="round,pad=0.015",
                          facecolor=ECA_C, edgecolor="white",
                          linewidth=1.2, alpha=0.93, zorder=3)
    ax.add_patch(eca)
    ax.text(cx, 3.33, "ECA Module  ( Conv1d, k = 3,  bias = False )  →  3 params",
            ha="center", va="center", fontsize=7.8, color="white",
            fontweight="bold", zorder=4)

    draw_box(ax, cx, 2.74, bw, bh, "Conv1d(16→16, k=3) + BN + ReLU  →  (B, 16, 25)", color=C2, fontsize=7.8)
    draw_box(ax, cx, 2.16, bw, bh, "Global Average Pool  →  f_cir ∈ ℝ¹⁶", color=C1, fontsize=7.8)

    # ── Aux branch label ───────────────────────────────────
    ax.text(14.2, 5.30, "Auxiliary Branch", ha="center", fontsize=9,
            fontweight="bold", color=A2)

    aw = 4.6
    ax2 = 14.2
    draw_box(ax, ax2, 4.78, aw, bh, "Linear(7→32) + ReLU  →  (B, 32)", color=A2, fontsize=7.8)
    draw_box(ax, ax2, 4.20, aw, bh, "Linear(32→16) + ReLU  →  (B, 16)", color=A2, fontsize=7.8)
    draw_box(ax, ax2, 3.62, aw, bh, "f_aux ∈ ℝ¹⁶", color=A1, fontsize=7.8)

    # ── Gated Fusion ───────────────────────────────────────
    draw_box(ax, 9.0, 1.52, 10.0, 0.52,
             "Gated Fusion:   g = σ( W_g · [f_cir ; f_aux] )     f_fused = g₁ · f_cir + g₂ · f_aux",
             color=FUS_C, fontsize=8.0)

    # ── Classifier ─────────────────────────────────────────
    draw_box(ax, 9.0, 0.88, 10.0, 0.52,
             "Linear(16→32) + ReLU + Dropout(0.3)   →   Linear(32→1)",
             color=CLF_C, fontsize=8.0)

    # ── Output ─────────────────────────────────────────────
    draw_box(ax, 9.0, 0.26, 2.4, 0.40, "ŷ  (logit)", color=HDR_C, fontsize=8.5)

    # ── Arrows: input → branches ───────────────────────────
    arrow(ax, 5.4, 5.59, cx,  5.02)
    arrow(ax, 12.6, 5.59, ax2, 5.02)

    # CIR internal
    for y0, y1 in [(4.55, 4.44), (3.97, 3.86), (3.39, 3.33),
                   (3.10, 2.97), (2.51, 2.40)]:
        arrow(ax, cx, y0, cx, y1)

    # Aux internal
    for y0, y1 in [(4.55, 4.44), (3.97, 3.86)]:
        arrow(ax, ax2, y0, ax2, y1)

    # Branches → fusion
    arrow(ax, cx,  1.93, 5.2, 1.77)
    arrow(ax, ax2, 3.39, 12.8, 1.77)

    # Fusion → classifier → output
    arrow(ax, 9.0, 1.26, 9.0, 1.14)
    arrow(ax, 9.0, 0.62, 9.0, 0.48)

    # ── Param summary (right side annotation) ──────────────
    lines = [
        "Parameter summary:",
        "  Conv1 + BN       : 160",
        "  ECA (k=3)        :   3  ←",
        "  Conv2 + BN       : 784",
        "  Aux MLP          : 784",
        "  Gated fusion     :  66",
        "  Classifier       : 577",
        "  ─────────────────────",
        "  Total            : 2,374",
    ]
    for i, l in enumerate(lines):
        ax.text(17.5, 4.8 - i * 0.38, l, ha="right", va="center",
                fontsize=6.8, color="#495057",
                fontweight="bold" if "Total" in l or "ECA" in l else "normal",
                family="monospace")

    ax.set_title("Fig. 1.  ECA-UWB Architecture",
                 fontsize=9.5, pad=4, fontweight="bold")
    plt.tight_layout(pad=0.4)
    plt.savefig(FIG / "fig1_architecture.png", bbox_inches="tight", dpi=220)
    plt.savefig(FIG / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close()
    print("  fig1_architecture (wide) regenerated")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Confusion Matrix Heatmap (exact numbers)
# ─────────────────────────────────────────────────────────────────────────────

def fig_confusion_matrix():
    # Exact values from ablation["full"]
    TN, FP, FN, TP = 2974, 175, 320, 2812

    cm = np.array([[TN, FP], [FN, TP]])
    labels = [["TN", "FP"], ["FN", "TP"]]
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_totals * 100          # row-normalized %

    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row %")

    classes = ["LOS", "NLOS"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted LOS", "Predicted NLOS"])
    ax.set_yticklabels(["Actual LOS", "Actual NLOS"])
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = cm_pct[i, j]
            color = "white" if pct > 60 else "#1a1a1a"
            ax.text(j, i,
                    f"{labels[i][j]}\n{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold")

    ax.set_title("Fig. 8. Confusion Matrix — eWINE Test Set\n"
                 f"Accuracy: 92.12%  |  AUC: 97.60%",
                 fontsize=8.5, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(FIG / "fig8_confusion.png", bbox_inches="tight", dpi=200)
    plt.savefig(FIG / "fig8_confusion.pdf", bbox_inches="tight")
    plt.close()
    print("  fig8_confusion done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Per-Environment Accuracy (eWINE 7 CSV files)
# ─────────────────────────────────────────────────────────────────────────────

def fig_per_environment():
    """
    Run eWINE-trained ECA-UWB on each of the 7 CSV files separately.
    Each CSV corresponds to one indoor environment (per eWINE paper).
    Uses the scaler fitted on the full training set.
    """
    import pandas as pd
    import pickle
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from ecauwb_model import ECAUWBNet
    from preprocess import _extract_cir_window

    AUX_COLS = [1, 8, 6, 3, 4, 5, 7]
    MDL = UWBNLOS / "models"

    # Load model + scaler
    HP = dict(cir_len=50, n_aux=7, ch1=16, ch2=16, k1=5, k2=3,
              eca_k=3, aux_hid=32, feat_dim=16, clf_hid=32, dropout=0.3)
    model = ECAUWBNet(**HP)
    model.load_state_dict(torch.load(MDL / "ecauwb_full.pt", map_location="cpu"))
    model.eval()

    with open(MDL / "ecauwb_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    fp_col = config.EWINE_DIAG_COLS["FP_IDX"]
    rx_col = config.EWINE_DIAG_COLS["RXPACC"]

    csv_files = sorted((config.EWINE_DIR).glob("*.csv"))

    # eWINE environment names (from paper: 7 indoor locations)
    env_names = [
        "Office 1", "Office 2", "Apartment",
        "Workshop", "Kitchen/Living", "Bedroom", "Boiler Room"
    ]
    if len(csv_files) != 7:
        env_names = [f.stem for f in csv_files]

    accs, aucs, los_rec, nlos_rec, counts = [], [], [], [], []

    for f, name in zip(csv_files, env_names):
        arr    = pd.read_csv(f, header=0).values
        labels = arr[:, 0].astype(int)
        cir_raw = arr[:, config.EWINE_CIR_START:].astype(float)

        rows = []
        for i in range(len(arr)):
            fp_idx = int(arr[i, fp_col])
            rxpacc = float(arr[i, rx_col]) + 1e-9
            cir_win = _extract_cir_window(cir_raw[i], fp_idx) / rxpacc
            aux = arr[i, AUX_COLS].astype(float)
            rows.append(np.concatenate([cir_win, aux]))

        X = np.array(rows, dtype=np.float32)
        X = scaler.transform(X).astype(np.float32)

        with torch.no_grad():
            logits = model(torch.from_numpy(X))
            probs  = torch.sigmoid(logits).numpy()

        preds = (probs > 0.5).astype(int)
        acc   = accuracy_score(labels, preds)
        auc   = roc_auc_score(labels, probs)

        # per-class recall
        mask_los  = labels == 0
        mask_nlos = labels == 1
        lr = accuracy_score(labels[mask_los],  preds[mask_los])
        nr = accuracy_score(labels[mask_nlos], preds[mask_nlos])

        accs.append(acc * 100); aucs.append(auc * 100)
        los_rec.append(lr * 100); nlos_rec.append(nr * 100)
        counts.append(len(labels))
        print(f"  {name:20s}: Acc={acc*100:.2f}%  AUC={auc*100:.2f}%  "
              f"LOS-Rec={lr*100:.2f}%  NLOS-Rec={nr*100:.2f}%  N={len(labels)}")

    # ── Plot ──
    x = np.arange(len(env_names))
    w = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    b1 = ax.bar(x - w,   accs,     w, label="Accuracy",    color="#457b9d", alpha=0.88, edgecolor="white")
    b2 = ax.bar(x,       los_rec,  w, label="LOS Recall",  color="#2a9d8f", alpha=0.88, edgecolor="white")
    b3 = ax.bar(x + w,   nlos_rec, w, label="NLOS Recall", color="#e63946", alpha=0.88, edgecolor="white")

    ax.bar_label(b1, fmt="%.1f", fontsize=5.8, padding=1.5, rotation=90)
    ax.bar_label(b2, fmt="%.1f", fontsize=5.8, padding=1.5, rotation=90)
    ax.bar_label(b3, fmt="%.1f", fontsize=5.8, padding=1.5, rotation=90)

    # Overall average line
    ax.axhline(np.mean(accs), color="#457b9d", lw=1.2, ls="--", alpha=0.6,
               label=f"Mean Acc ({np.mean(accs):.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=7.5, rotation=15, ha="right")
    ax.set_ylabel("(%)")
    ax.set_ylim(70, 105)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax.set_title("Fig. 9. Per-Environment Performance on eWINE"
                 " (model trained on full training split)",
                 fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG / "fig9_per_env.png", bbox_inches="tight", dpi=200)
    plt.savefig(FIG / "fig9_per_env.pdf", bbox_inches="tight")
    plt.close()
    print("  fig9_per_env done")

    return env_names, accs, aucs, los_rec, nlos_rec, counts


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating extra figures ...")
    fig_architecture_wide()
    fig_confusion_matrix()
    print("\nPer-environment breakdown:")
    results = fig_per_environment()
    print("\nDone. All figures in", FIG)
