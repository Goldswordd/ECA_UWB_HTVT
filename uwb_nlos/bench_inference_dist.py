"""
Measure inference time distribution of ECA-UWB on the current device.
Produces a JSON result file and a violin/histogram figure.

Run on each device (x86, Pi 4, Pi 5, ESP32 not applicable):
  python bench_inference_dist.py --device "Raspberry Pi 4" --out logs/latency_pi4.json
  python bench_inference_dist.py --device "Raspberry Pi 5" --out logs/latency_pi5.json
  python bench_inference_dist.py  # defaults to current machine

Then generate the combined figure:
  python bench_inference_dist.py --plot  (reads all latency_*.json files)
"""
import argparse, json, time, platform
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(device_name: str, n_warmup: int = 100, n_runs: int = 3000,
                  model_path: Path = ROOT / "models" / "ecauwb.pt",
                  out_path: Path = None) -> dict:
    from ecauwb_model import ECAUWBNet
    import config

    model = ECAUWBNet(
        cir_len=50, n_aux=7, ch1=16, ch2=16, k1=5, k2=3,
        eca_k=3, aux_hid=32, feat_dim=16, clf_hid=32, dropout=0.0,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 57)   # (1, cir_len + n_aux)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    # Timed runs — single-sample, no batch
    times_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times_ms.append((time.perf_counter() - t0) * 1e3)

    times_ms = np.array(times_ms)
    result = {
        "device":      device_name,
        "platform":    platform.platform(),
        "n_runs":      n_runs,
        "mean_ms":     float(np.mean(times_ms)),
        "median_ms":   float(np.median(times_ms)),
        "std_ms":      float(np.std(times_ms)),
        "p5_ms":       float(np.percentile(times_ms,  5)),
        "p95_ms":      float(np.percentile(times_ms, 95)),
        "p99_ms":      float(np.percentile(times_ms, 99)),
        "min_ms":      float(times_ms.min()),
        "max_ms":      float(times_ms.max()),
        "samples_ms":  times_ms.tolist(),   # full distribution for plotting
    }

    if out_path is None:
        tag = device_name.lower().replace(" ", "_")
        out_path = ROOT / "logs" / f"latency_{tag}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Device: {device_name}  mean={result['mean_ms']:.3f} ms  "
          f"p95={result['p95_ms']:.3f} ms  → {out_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plot (call after collecting results from multiple devices)
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(log_dir: Path = ROOT / "logs",
                       out_path: Path = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pattern = "latency_*.json"
    files = sorted(log_dir.glob(pattern))
    if not files:
        print(f"No {pattern} files found in {log_dir}")
        return

    data = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        if "samples_ms" in r:
            data.append(r)

    if not data:
        print("No sample data found (need 'samples_ms' key in JSON).")
        return

    fig, ax = plt.subplots(figsize=(max(4, len(data) * 2), 5))

    labels  = [d["device"] for d in data]
    samples = [d["samples_ms"] for d in data]
    means   = [d["mean_ms"] for d in data]

    # Violin plot
    vp = ax.violinplot(samples, positions=range(len(data)),
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.6)

    # Mean markers
    ax.scatter(range(len(data)), means, color="red", zorder=5,
               s=40, label="Mean", marker="D")

    # p5–p95 whiskers
    for i, d in enumerate(data):
        ax.vlines(i, d["p5_ms"], d["p95_ms"], color="grey",
                  linewidth=2, alpha=0.7)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Inference Time (ms)", fontsize=11)
    ax.set_title("ECA-UWB Single-Sample Inference Time Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate mean values
    for i, d in enumerate(data):
        ax.annotate(f"{d['mean_ms']:.2f} ms",
                    xy=(i, d["mean_ms"]),
                    xytext=(i + 0.08, d["mean_ms"] + 0.05),
                    fontsize=8, color="red")

    plt.tight_layout()
    if out_path is None:
        out_path = ROOT.parent / "paper_ecauwb" / "figures" / "fig_inference_dist.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=platform.node(),
                        help="Human-readable device name (e.g. 'Raspberry Pi 4')")
    parser.add_argument("--out",    default=None,
                        help="Output JSON path (default: logs/latency_<device>.json)")
    parser.add_argument("--plot",   action="store_true",
                        help="Generate combined figure from all latency_*.json files")
    parser.add_argument("--n_runs", type=int, default=3000)
    args = parser.parse_args()

    import platform
    if args.plot:
        plot_distributions()
    else:
        run_benchmark(args.device, n_runs=args.n_runs,
                      out_path=Path(args.out) if args.out else None)
        plot_distributions()   # regenerate combined figure each time
