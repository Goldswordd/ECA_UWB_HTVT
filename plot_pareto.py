"""
Generate fig_pareto.png — Accuracy vs. Parameter Count (log scale)
for ECA-UWB paper. Pareto frontier highlighted.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── IEEE-quality style ─────────────────────────────────────────────────
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

# ── Data from Table II ─────────────────────────────────────────────────
#  (name, params, accuracy, color, marker)
models = [
    ("CNN-LSTM",        7441,   88.82, "#264653", "o"),
    ("FCN-Attention",   200000, 88.24, "#6c757d", "o"),
    ("CIR-CNN+MLP",     1578,   87.96, "#a8dadc", "o"),
    ("SA-TinyML",       4627,   93.65, "#457b9d", "o"),
    ("LightMamba",      25793,  92.38, "#2a9d8f", "o"),
    ("MS-CNN-SA",       24145,  93.10, "#e9c46a", "o"),
    ("ECA-UWB",         2374,   91.45, "#e63946", "*"),
]

# ── Pareto frontier points (sorted by params ascending) ────────────────
pareto_names = ["CIR-CNN+MLP", "ECA-UWB", "SA-TinyML"]
pareto_data = [(m[1], m[2]) for m in models if m[0] in pareto_names]
pareto_data.sort(key=lambda x: x[0])

fig, ax = plt.subplots(figsize=(4.5, 3.2))

# Plot Pareto frontier line
px, py = zip(*pareto_data)
ax.plot(px, py, "--", color="#e63946", alpha=0.45, linewidth=1.5,
        zorder=1, label="Pareto frontier")

# Plot each model
for name, params, acc, color, marker in models:
    ms = 14 if marker == "*" else 7
    zorder = 5 if name == "ECA-UWB" else 3
    edgecolor = "#c62828" if name == "ECA-UWB" else "#333"
    lw = 1.5 if name == "ECA-UWB" else 0.8
    ax.scatter(params, acc, s=ms**2, marker=marker, color=color,
               edgecolors=edgecolor, linewidth=lw, zorder=zorder)

# Labels with offsets
offsets = {
    "CNN-LSTM":      (1.25, -0.9),
    "FCN-Attention": (1.15, +0.4),
    "CIR-CNN+MLP":   (0.65, -0.95),
    "SA-TinyML":     (1.25, +0.4),
    "LightMamba":    (0.75, +0.5),
    "MS-CNN-SA":     (0.75, -0.9),
    "ECA-UWB":       (0.55, +0.55),
}

for name, params, acc, color, marker in models:
    dx_mult, dy = offsets[name]
    fontweight = "bold" if name == "ECA-UWB" else "normal"
    txt_color = "#c62828" if name == "ECA-UWB" else "#333"
    fontsize = 8.5 if name in ("ECA-UWB", "SA-TinyML") else 7.5
    ax.annotate(
        name, xy=(params, acc),
        xytext=(params * dx_mult, acc + dy),
        fontsize=fontsize, color=txt_color, fontweight=fontweight,
        ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color="#999", lw=0.5)
            if name not in ("ECA-UWB", "SA-TinyML") else None,
    )

ax.set_xscale("log")
ax.set_xlabel("Number of Parameters (log scale)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(800, 400000)
ax.set_ylim(86.5, 95.0)

ax.legend(loc="lower right", framealpha=0.85)

plt.tight_layout()
outpath = "figures/fig_pareto.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {outpath}")
