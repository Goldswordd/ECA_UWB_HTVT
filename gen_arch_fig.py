"""
ECA-UWB Architecture Diagram — IEEE two-column style.
Clean horizontal left-to-right flow, 7.16 × 3.2 inches.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ────────────────────────────────────────────────────────────────────
C_INPUT = "#e8eaf6"   # lavender – input
C_DIAG  = "#bbdefb"  # light blue – diagnostic (Branch A)
C_CIR   = "#b3e5fc"  # sky blue – CIR (Branch B)
C_ECA   = "#fff9c4"  # yellow – ECA attention
C_FUSE  = "#c8e6c9"  # green – fusion
C_CLS   = "#dcedc8"  # light green – classifier
C_OUT   = "#ffcdd2"  # red – output
EDGE    = "#37474f"
ARROW_C = "#455a64"
BG      = "white"

FW, FH = 7.16, 3.2
DPI = 300

fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH)
ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor(BG)

# ── Primitives ─────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, color, label1, label2=None, fs=6.8, edge=EDGE, lw=0.9, r=0.07):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={r}",
                          fc=color, ec=edge, lw=lw, zorder=3)
    ax.add_patch(rect)
    cx, cy = x + w/2, y + h/2
    if label2:
        ax.text(cx, cy + 0.09, label1, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='#1a237e', zorder=4)
        ax.text(cx, cy - 0.10, label2, ha='center', va='center',
                fontsize=fs - 1.2, color='#37474f', zorder=4)
    else:
        ax.text(cx, cy, label1, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='#1a237e', zorder=4)

def harrow(ax, x0, y, x1, color=ARROW_C, lw=1.1):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=1, shrinkB=1),
                zorder=5)

def varrow(ax, x, y0, y1, color=ARROW_C, lw=1.0):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=1, shrinkB=1),
                zorder=5)

def txt(x, y, s, fs=6.2, color='#546e7a', ha='center', va='center', bold=False):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            color=color, fontweight='bold' if bold else 'normal', zorder=4)

def bracket(x, y, w, label, color):
    """Horizontal brace-like bracket label."""
    ax.plot([x, x+w], [y, y], color=color, lw=1.2, solid_capstyle='round', zorder=2)
    ax.plot([x, x], [y, y+0.07], color=color, lw=1.0, zorder=2)
    ax.plot([x+w, x+w], [y, y+0.07], color=color, lw=1.0, zorder=2)
    ax.text(x + w/2, y - 0.12, label, ha='center', va='top',
            fontsize=6.0, color=color, fontstyle='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# Layout: three horizontal lanes
#   Top lane    (y ≈ 2.35): Branch A
#   Middle lane (y ≈ 1.60): Input / Fusion / Classifier / Output
#   Bottom lane (y ≈ 0.60): Branch B
# ═══════════════════════════════════════════════════════════════════════════════

BH = 0.52   # block height standard
SH = 0.44   # small block height

# ── X grid (left-to-right) ─────────────────────────────────────────────────────
X0   = 0.15   # Input left edge
X1   = 0.92   # Split right / branch start
X_A1 = 1.20   # Branch A: FC
X_B1 = 1.20   # Branch B: Conv1
GAP  = 0.82   # column spacing for branch B blocks
X_B2 = X_B1 + GAP
X_B3 = X_B2 + GAP        # GlobalAvgPool
X_ECA = X_B2 + GAP/2     # ECA (sits between B2 and B3, above)
X_FU = X_B3 + 0.80       # Fusion
X_C1 = X_FU + 0.80       # FC(64→32)
X_C2 = X_C1 + 0.78       # FC(32→1)
X_OUT= X_C2 + 0.78       # Output

BW_B = 0.72   # Branch B block width
BW_A = 0.72   # Branch A FC width
BW_F = 0.72   # Fusion block width
BW_C = 0.68   # Classifier block width
BW_O = 0.52   # Output width

# Y centres
Y_A  = 2.48   # Branch A centre
Y_MID= 1.60   # mid lane (fusion/clf/output) centre
Y_B  = 0.72   # Branch B centre
Y_ECA= 1.22   # ECA box top zone

# ── Draw Input ─────────────────────────────────────────────────────────────────
INW, INH = 0.62, BH
rbox(X0, Y_MID - INH/2, INW, INH, C_INPUT, "Input", "110-dim", fs=7.2)
txt(X0 + INW/2, Y_MID - INH/2 - 0.16, "1×110 vector", fs=5.8)

# ── Draw Split ────────────────────────────────────────────────────────────────
SPW, SPH = 0.52, SH
SPX = X1
rbox(SPX, Y_MID - SPH/2, SPW, SPH, "#eceff1", "Split", fs=7)
harrow(ax, X0 + INW, Y_MID, SPX)

SP_CX = SPX + SPW/2
SP_CY_top = Y_MID + SPH/2
SP_CY_bot = Y_MID - SPH/2

# ── Branch A lane ─────────────────────────────────────────────────────────────
# Split → up to Branch A level
ax.plot([SP_CX, SP_CX], [SP_CY_top, Y_A], color=ARROW_C, lw=1.1, zorder=3)
ax.annotate('', xy=(X_A1, Y_A), xytext=(SP_CX, Y_A),
            arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.1, shrinkA=0, shrinkB=1), zorder=5)

# FC block
rbox(X_A1, Y_A - BH/2, BW_A, BH, C_DIAG, "FC", "10→32, ReLU", fs=7)

# Branch A label
txt(X_A1 + BW_A/2, Y_A + BH/2 + 0.16, "Branch A — Diagnostic", fs=6.5, bold=True, color="#1565c0")

# Arrow: FC(A) → right to Fusion level
X_FA_right = X_A1 + BW_A
# Horizontal to fusion, then down
ax.plot([X_FA_right, X_FU + BW_F/2], [Y_A, Y_A], color=ARROW_C, lw=1.1, zorder=3)
ax.annotate('', xy=(X_FU + BW_F/2, Y_MID + BH/2),
            xytext=(X_FU + BW_F/2, Y_A),
            arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.1, shrinkA=0, shrinkB=1), zorder=5)

# ── Branch B lane ─────────────────────────────────────────────────────────────
# Split → down to Branch B
ax.plot([SP_CX, SP_CX], [SP_CY_bot, Y_B], color=ARROW_C, lw=1.1, zorder=3)
ax.annotate('', xy=(X_B1, Y_B), xytext=(SP_CX, Y_B),
            arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.1, shrinkA=0, shrinkB=1), zorder=5)

# Conv1D block
rbox(X_B1, Y_B - BH/2, BW_B, BH, C_CIR, "Conv1D", "1→16, k=5\nBN+ReLU", fs=6.8)
harrow(ax, X_B1 + BW_B, Y_B, X_B2)

# Conv2 block
rbox(X_B2, Y_B - BH/2, BW_B, BH, C_CIR, "Conv1D", "16→32, k=3\nBN+ReLU", fs=6.8)
harrow(ax, X_B2 + BW_B, Y_B, X_B3)

# GlobalAvgPool block
rbox(X_B3, Y_B - BH/2, BW_B, BH, "#81d4fa", "GAP", "→ 32-dim", fs=7)

# Branch B label
txt((X_B1 + X_B3 + BW_B)/2, Y_B - BH/2 - 0.18, "Branch B — CIR Waveform", fs=6.5, bold=True, color="#01579b")

# ── ECA Attention ─────────────────────────────────────────────────────────────
ECA_W, ECA_H = 0.78, 0.46
ECA_X = X_B3 + BW_B/2 - ECA_W/2
ECA_Y = Y_B + BH/2 + 0.18

rbox(ECA_X, ECA_Y, ECA_W, ECA_H, C_ECA, "ECA", "Conv(k=3)+σ", fs=6.8)

# Arrow: GAP top → ECA bottom
varrow(ax, X_B3 + BW_B/2, Y_B + BH/2, ECA_Y)

# Arrow: ECA top → merges into Fusion (via angled path)
ECA_TOP_X = ECA_X + ECA_W/2
ECA_TOP_Y = ECA_Y + ECA_H
ax.plot([ECA_TOP_X, ECA_TOP_X, X_FU + BW_F/2],
        [ECA_TOP_Y, Y_MID - BH/2 - 0.01, Y_MID - BH/2 - 0.01],
        color=ARROW_C, lw=1.0, zorder=3)
ax.annotate('', xy=(X_FU + BW_F/2, Y_MID - BH/2),
            xytext=(X_FU + BW_F/2, Y_MID - BH/2 - 0.01),
            arrowprops=dict(arrowstyle='->', color=ARROW_C, lw=1.0, shrinkA=0, shrinkB=1), zorder=5)

# ECA annotation
txt(ECA_X + ECA_W/2, ECA_Y + ECA_H + 0.10, "channel\nattention ⊗", fs=5.8, color="#f57f17")

# ── Fusion ────────────────────────────────────────────────────────────────────
rbox(X_FU, Y_MID - BH/2, BW_F, BH, C_FUSE, "Concat", "64-dim", fs=7)

# ── Classifier ────────────────────────────────────────────────────────────────
harrow(ax, X_FU + BW_F, Y_MID, X_C1)
rbox(X_C1, Y_MID - BH/2, BW_C, BH, C_CLS, "FC+Drop", "64→32,p=0.3", fs=6.8)

harrow(ax, X_C1 + BW_C, Y_MID, X_C2)
rbox(X_C2, Y_MID - BH/2, BW_C, BH, C_CLS, "FC", "32→1,σ", fs=7)

# ── Output ────────────────────────────────────────────────────────────────────
harrow(ax, X_C2 + BW_C, Y_MID, X_OUT)
rbox(X_OUT, Y_MID - BH/2, BW_O, BH, C_OUT, "LOS\nNLOS", fs=7.5, edge="#c62828")

# ── Param count ───────────────────────────────────────────────────────────────
txt(FW/2, FH - 0.18, "ECA-UWB  |  2,374 parameters", fs=7.0, bold=True, color="#1a237e")

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)
out = "/home/johnw/Documents/Paper/UWB_journals/paper_ecauwb/figures/fig1_architecture.png"
plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved {out}")
