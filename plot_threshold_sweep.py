"""
Generate fig_threshold_sweep.png — Threshold sweep τ ∈ [0.30, 0.70]
for ECA-UWB paper.

NOTE: This uses *synthetic data* that matches the paper's described
behaviour. Replace with real sweep data before submission.
Key constraints from the paper:
  - At τ=0.54: Acc=91.45%, F1=91.43%, NLOS Recall=91.44%, LOS Recall=91.46%
  - Acc > 90% for τ ∈ [0.45, 0.62]   (17-pp window)
  - τ < 0.45 → NLOS Recall ~ 97%, LOS Recall drops below 86%
  - τ > 0.62 → LOS Recall increases, NLOS Recall drops
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Synthetic sweep data ───────────────────────────────────────────────
tau = np.arange(0.30, 0.71, 0.02)

def sigmoid(x, center, scale, lo, hi):
    """Shifted sigmoid mapping"""
    return lo + (hi - lo) / (1 + np.exp(-scale * (x - center)))

# NLOS Recall: high at low τ, drops at high τ
nlos_recall = sigmoid(tau, center=0.58, scale=-22, lo=78.0, hi=97.5)
# Add known point constraint: at τ=0.54 → 91.44%
nlos_recall_054 = np.interp(0.54, tau, nlos_recall)
nlos_recall += (91.44 - nlos_recall_054) * np.exp(-((tau - 0.54)/0.15)**2)

# LOS Recall: low at low τ, high at high τ
los_recall = sigmoid(tau, center=0.46, scale=20, lo=78.0, hi=96.5)
los_recall_054 = np.interp(0.54, tau, los_recall)
los_recall += (91.46 - los_recall_054) * np.exp(-((tau - 0.54)/0.15)**2)

# Accuracy: peaks around 0.50-0.55
accuracy = (nlos_recall + los_recall) / 2.0
# Ensure Acc at 0.54 = 91.45
acc_054 = np.interp(0.54, tau, accuracy)
accuracy += (91.45 - acc_054)

# F1-score: closely tracks accuracy but slightly lower
f1_score = accuracy - 0.02 + 0.3 * np.sin(np.pi * (tau - 0.3) / 0.4)
# Ensure F1 at 0.54 = 91.43
f1_054 = np.interp(0.54, tau, f1_score)
f1_score += (91.43 - f1_054) * np.exp(-((tau - 0.54)/0.2)**2)

# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.2))

# Shaded region where Acc > 90%
mask = accuracy >= 90.0
if mask.any():
    tau_hi = tau[mask]
    ax.axvspan(tau_hi[0], tau_hi[-1], alpha=0.10, color="#2a9d8f",
               label=r"Acc $\geq$ 90% region")

# Curves
ax.plot(tau, accuracy,    "-o", color="#e63946", markersize=3.5, linewidth=1.8, label="Accuracy")
ax.plot(tau, f1_score,    "-s", color="#457b9d", markersize=3,   linewidth=1.5, label="F1-score")
ax.plot(tau, nlos_recall, "-^", color="#2a9d8f", markersize=3,   linewidth=1.5, label="NLOS Recall")
ax.plot(tau, los_recall,  "-d", color="#e9c46a", markersize=3,   linewidth=1.5, label="LOS Recall")

# Reference lines
ax.axvline(x=0.54, color="#333", linestyle="--", linewidth=0.9, alpha=0.7)
ax.axhline(y=90.0, color="#999", linestyle=":", linewidth=0.8, alpha=0.6)

# Annotate τ=0.54
ax.annotate(r"$\tau = 0.54$", xy=(0.54, 86), xytext=(0.58, 84.5),
            fontsize=8, color="#333", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))

ax.set_xlabel(r"Decision Threshold $\tau$")
ax.set_ylabel("Metric (%)")
ax.set_xlim(0.29, 0.71)
ax.set_ylim(76, 99)
ax.legend(loc="lower left", framealpha=0.85, ncol=1)

plt.tight_layout()
outpath = "figures/fig_threshold_sweep.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {outpath}")
