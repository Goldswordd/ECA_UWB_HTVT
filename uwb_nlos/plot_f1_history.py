"""
Generate publication-quality epoch-wise F1-score evolution figure.
Reads logs/ecauwb_train_history.json produced by train_ecauwb.py.

Usage:
  python plot_f1_history.py
  python plot_f1_history.py --out figures/fig_f1_history.png
"""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT   = Path(__file__).parent
LOG    = ROOT / "logs" / "ecauwb_train_history.json"
OUTDIR = ROOT.parent / "paper_ecauwb" / "figures"


def smooth(x, w=5):
    """Simple uniform moving average."""
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    padded = np.pad(x, w // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def plot_f1_history(hist_path: Path, out_path: Path):
    with open(hist_path) as f:
        h = json.load(f)

    train_f1 = np.array(h["train_f1"])
    val_f1   = np.array(h["val_f1"])
    epochs   = np.arange(1, len(train_f1) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Raw (transparent) + smoothed (solid)
    ax.plot(epochs, train_f1, color="#4C72B0", alpha=0.25, linewidth=1)
    ax.plot(epochs, val_f1,   color="#DD8452", alpha=0.25, linewidth=1)
    ax.plot(epochs, smooth(train_f1), color="#4C72B0", linewidth=2,
            label="Train F1")
    ax.plot(epochs, smooth(val_f1),   color="#DD8452", linewidth=2,
            label="Val F1")

    # Mark best val F1
    best_ep  = int(np.argmax(val_f1)) + 1
    best_val = val_f1[best_ep - 1]
    ax.axvline(best_ep, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(f"Best val F1\n{best_val:.4f} @ ep {best_ep}",
                xy=(best_ep, best_val),
                xytext=(best_ep + max(2, len(epochs)//15), best_val - 0.03),
                fontsize=8, color="grey",
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Epoch-wise F1-Score Evolution (ECA-UWB on eWINE)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=max(0, train_f1.min() - 0.05))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist", default=str(LOG))
    parser.add_argument("--out",  default=str(OUTDIR / "fig_f1_history.png"))
    args = parser.parse_args()
    plot_f1_history(Path(args.hist), Path(args.out))
