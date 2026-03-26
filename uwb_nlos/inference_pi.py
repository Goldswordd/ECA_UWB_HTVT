"""
Real-time NLOS classification on Raspberry Pi.

Reads measurements from DWM1001 tag over /dev/ttyACM0.
Each line = one anchor measurement, CSV format:
  [anchor_id, cir_0..cir_49, feat_0..feat_10]   (62 columns)

Outputs per-anchor NLOS probability at ~80 Hz.

Usage:
  python inference_pi.py                          # ONNX model (recommended)
  python inference_pi.py --model rf               # Random Forest fallback
  python inference_pi.py --port /dev/ttyACM1      # custom port
  python inference_pi.py --log output.csv         # save results to CSV

Requirements on RPi:
  pip install pyserial numpy scipy scikit-learn onnxruntime

ONNX runtime for RPi (ARM64, Python 3.11):
  pip install onnxruntime   # or onnxruntime-aarch64 for older Pi
"""
import argparse
import csv
import pickle
import sys
import time
import threading
import queue
from collections import deque
from pathlib import Path

import numpy as np
import serial

# Add project root to path so config/preprocess are importable from anywhere
sys.path.insert(0, str(Path(__file__).parent))
import config

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
NLOS_THRESHOLD   = 0.5    # probability threshold for NLOS decision
SMOOTH_WINDOW    = 3      # exponential moving average window (samples)
EMA_ALPHA        = 2.0 / (SMOOTH_WINDOW + 1)


# ─────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────

class ONNXInferencer:
    """ONNX Runtime inference — recommended for RPi."""

    def __init__(self, model_path=None):
        import onnxruntime as ort
        model_path = model_path or config.MODEL_DIR / "nlos_classifier.onnx"
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "Run on dev machine: python export_onnx.py, then copy to RPi."
            )
        self.sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        print(f"[ONNX] Loaded: {model_path}")

    def predict(self, cir: np.ndarray, feat: np.ndarray) -> float:
        """
        cir  : (50,)  float32
        feat : (11,)  float32
        returns: NLOS probability in [0, 1]
        """
        out = self.sess.run(None, {
            "cir":  cir[np.newaxis, :, np.newaxis].astype(np.float32),
            "feat": feat[np.newaxis, :].astype(np.float32),
        })[0]
        return float(out[0, 0])


class RFInferencer:
    """Random Forest fallback — works without onnxruntime."""

    def __init__(self, model_path=None, scaler_path=None):
        rf_path  = model_path  or config.MODEL_DIR / "rf_model.pkl"
        sc_path  = scaler_path or config.SCALER_PATH
        if not Path(rf_path).exists():
            raise FileNotFoundError(f"RF model not found: {rf_path}")
        with open(rf_path, "rb") as f:
            self.rf = pickle.load(f)
        with open(sc_path, "rb") as f:
            self.scaler = pickle.load(f)
        print(f"[RF] Loaded: {rf_path}")

    def predict(self, cir: np.ndarray, feat: np.ndarray) -> float:
        feat_sc = self.scaler.transform(feat.reshape(1, -1))
        X       = np.concatenate([cir[np.newaxis], feat_sc], axis=1)
        return float(self.rf.predict_proba(X)[0, 1])


# ─────────────────────────────────────────────────────────────
# Serial parser
# ─────────────────────────────────────────────────────────────

def parse_line(line: str) -> tuple[str, np.ndarray, np.ndarray] | None:
    """
    Parse one CSV line from DWM1001 tag.
    Returns (anchor_id, cir_window, features) or None if parse fails.

    Expected format (config.SERIAL_HAS_ANCHOR_ID = True):
      A0,cir_0,...,cir_49,feat_0,...,feat_10
      → 1 + 50 + 11 = 62 columns

    If SERIAL_HAS_ANCHOR_ID = False:
      cir_0,...,cir_49,feat_0,...,feat_10
      → 61 columns
    """
    try:
        parts = line.strip().split(",")
        if config.SERIAL_HAS_ANCHOR_ID:
            anchor_id = parts[0].strip()
            data = np.array(parts[1:], dtype=np.float32)
        else:
            anchor_id = "A?"
            data = np.array(parts, dtype=np.float32)

        expected = config.CIR_LEN + config.N_FEATURES
        if len(data) < expected:
            return None

        cir  = data[config.SERIAL_CIR_START_COL - (1 if config.SERIAL_HAS_ANCHOR_ID else 0):
                    config.SERIAL_CIR_START_COL - (1 if config.SERIAL_HAS_ANCHOR_ID else 0) + config.CIR_LEN]
        feat = data[config.SERIAL_FEAT_START_COL - (1 if config.SERIAL_HAS_ANCHOR_ID else 0):
                    config.SERIAL_FEAT_START_COL - (1 if config.SERIAL_HAS_ANCHOR_ID else 0) + config.N_FEATURES]
        return anchor_id, cir, feat

    except (ValueError, IndexError):
        return None


# ─────────────────────────────────────────────────────────────
# Per-anchor state (for smoothing and EKF R-matrix output)
# ─────────────────────────────────────────────────────────────

class AnchorState:
    def __init__(self, anchor_id: str):
        self.id       = anchor_id
        self.ema_prob = 0.5    # smoothed NLOS probability
        self.n_total  = 0
        self.n_nlos   = 0

    def update(self, raw_prob: float) -> float:
        """Update EMA and return smoothed probability."""
        self.ema_prob = EMA_ALPHA * raw_prob + (1 - EMA_ALPHA) * self.ema_prob
        self.n_total += 1
        if raw_prob > NLOS_THRESHOLD:
            self.n_nlos += 1
        return self.ema_prob

    @property
    def nlos_rate(self) -> float:
        return self.n_nlos / max(self.n_total, 1)

    @property
    def r_scale(self) -> float:
        """
        Adaptive EKF measurement noise scale factor.
        Low confidence (NLOS likely) → large R → trust IMU more.
        Range: [1.0, 20.0]
        """
        return 1.0 + 19.0 * self.ema_prob  # linear 1→20


# ─────────────────────────────────────────────────────────────
# Serial reader thread
# ─────────────────────────────────────────────────────────────

class SerialReader(threading.Thread):
    def __init__(self, port: str, baud: int, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.port      = port
        self.baud      = baud
        self.out_queue = out_queue
        self._stop     = threading.Event()
        self.n_lines   = 0
        self.n_errors  = 0

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
            print(f"[Serial] Connected: {self.port} @ {self.baud} baud")
        except serial.SerialException as e:
            print(f"[Serial] ERROR: {e}")
            return

        while not self._stop.is_set():
            try:
                raw = ser.readline()
                if raw:
                    line = raw.decode("ascii", errors="ignore")
                    self.n_lines += 1
                    self.out_queue.put(line)
            except Exception:
                self.n_errors += 1

        ser.close()

    def stop(self):
        self._stop.set()


# ─────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────

def run_inference(args):
    # ── Load model ───────────────────────────────────────────
    if args.model == "onnx":
        try:
            inferencer = ONNXInferencer()
        except (ImportError, FileNotFoundError) as e:
            print(f"[ONNX] {e}\nFalling back to RF…")
            inferencer = RFInferencer()
    else:
        inferencer = RFInferencer()

    # ── Load scaler (only needed for ONNX — RF has its own) ──
    scaler = None
    if isinstance(inferencer, ONNXInferencer):
        from preprocess import load_scaler
        scaler = load_scaler()

    # ── Anchor state tracking ────────────────────────────────
    anchors: dict[str, AnchorState] = {}

    # ── Optional CSV logging ─────────────────────────────────
    log_file = None
    csv_writer = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([
            "timestamp_s", "anchor_id",
            "nlos_prob_raw", "nlos_prob_ema", "nlos_label",
            "r_scale", "latency_ms"
        ])
        print(f"[Log] Saving to {args.log}")

    # ── Serial queue ─────────────────────────────────────────
    line_queue: queue.Queue = queue.Queue(maxsize=500)
    reader = SerialReader(args.port, config.SERIAL_BAUD, line_queue)
    reader.start()

    print(f"\n[Inference] Running. NLOS threshold={NLOS_THRESHOLD}  "
          f"EMA alpha={EMA_ALPHA:.3f}")
    print("  Columns: anchor | NLOS_prob | EMA_prob | label | R_scale | lat(ms)")
    print("─" * 70)

    t_start  = time.time()
    n_proc   = 0
    lat_deque = deque(maxlen=100)   # last 100 latencies for FPS display

    try:
        while True:
            # Process all available lines without blocking
            try:
                line = line_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            parsed = parse_line(line)
            if parsed is None:
                continue

            anchor_id, cir, feat = parsed

            # ── Scale features ────────────────────────────────
            if scaler is not None:
                feat_sc = scaler.transform(feat.reshape(1, -1)).squeeze(0)
            else:
                feat_sc = feat  # RF handles scaling internally

            # ── Inference ─────────────────────────────────────
            t0 = time.perf_counter()
            raw_prob = inferencer.predict(cir, feat_sc)
            lat_ms   = (time.perf_counter() - t0) * 1e3
            lat_deque.append(lat_ms)

            # ── Update anchor state ───────────────────────────
            if anchor_id not in anchors:
                anchors[anchor_id] = AnchorState(anchor_id)
            state    = anchors[anchor_id]
            ema_prob = state.update(raw_prob)
            label    = "NLOS" if ema_prob > NLOS_THRESHOLD else "LOS "
            r_scale  = state.r_scale
            n_proc  += 1

            # ── Console output ────────────────────────────────
            bar_len = int(ema_prob * 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            print(
                f"\r  {anchor_id:>4} | {raw_prob:.3f} | {ema_prob:.3f} | "
                f"{label} | R×{r_scale:5.1f} | {lat_ms:.2f}ms  [{bar}]",
                end="", flush=True
            )

            # Print summary line every 4 anchors (one full cycle)
            if n_proc % 4 == 0:
                fps  = len(lat_deque) / max(sum(lat_deque) / 1e3, 1e-9)
                mean_lat = np.mean(lat_deque)
                print(
                    f"\n  ── cycle {n_proc//4:5d}  "
                    f"FPS={fps:.0f}  lat={mean_lat:.2f}ms  "
                    f"up={time.time()-t_start:.0f}s  "
                    f"NLOS rates: " +
                    " ".join(f"{k}={v.nlos_rate:.2f}"
                             for k, v in sorted(anchors.items()))
                )

            # ── CSV log ───────────────────────────────────────
            if csv_writer:
                csv_writer.writerow([
                    f"{time.time():.4f}", anchor_id,
                    f"{raw_prob:.4f}", f"{ema_prob:.4f}",
                    1 if ema_prob > NLOS_THRESHOLD else 0,
                    f"{r_scale:.2f}", f"{lat_ms:.3f}",
                ])

    except KeyboardInterrupt:
        print("\n\n[Inference] Stopped by user.")
    finally:
        reader.stop()
        if log_file:
            log_file.close()

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Processed {n_proc} samples in {time.time()-t_start:.1f}s")
    if lat_deque:
        print(f"Avg latency: {np.mean(lat_deque):.3f} ms")
    for aid, state in sorted(anchors.items()):
        print(f"  {aid}: {state.n_total} samples, "
              f"NLOS rate={state.nlos_rate:.3f}, "
              f"final R×{state.r_scale:.1f}")


# ─────────────────────────────────────────────────────────────
# Offline replay (for testing without hardware)
# ─────────────────────────────────────────────────────────────

def replay_demo(args):
    """
    Replay mode: simulate live inference from a pre-recorded CSV
    (or generate dummy data if no file given).
    Usage:
      python inference_pi.py --replay recorded_data.csv
      python inference_pi.py --replay DEMO   # synthetic data
    """
    from preprocess import build_dataset, load_scaler
    from scipy.special import expit

    print("[Replay] Loading test data for demo …")
    splits = build_dataset(use_oiud=False)
    cir_te, feat_te, y_te = splits[6], splits[7], splits[8]

    try:
        inferencer = ONNXInferencer()
        scaler     = load_scaler()
        mode       = "ONNX"
    except Exception:
        inferencer = RFInferencer()
        scaler     = None
        mode       = "RF"

    print(f"[Replay] Using {mode} model on {len(y_te)} test samples")
    anchor_ids = ["A0", "A1", "A2", "A3"]

    correct = 0
    latencies = []
    for i in range(min(200, len(y_te))):
        cir  = cir_te[i].squeeze(-1)
        feat = feat_te[i]
        if scaler is not None:
            feat_sc = scaler.transform(feat.reshape(1,-1)).squeeze(0)
        else:
            feat_sc = feat

        t0       = time.perf_counter()
        raw_prob = inferencer.predict(cir, feat_sc)
        lat_ms   = (time.perf_counter() - t0) * 1e3
        latencies.append(lat_ms)

        pred  = 1 if raw_prob > NLOS_THRESHOLD else 0
        correct += (pred == int(y_te[i]))
        anchor = anchor_ids[i % 4]
        label  = "NLOS" if pred else "LOS "
        truth  = "NLOS" if y_te[i] else "LOS "
        ok     = "✓" if pred == y_te[i] else "✗"
        print(f"  [{i:>4}] {anchor} | prob={raw_prob:.3f} | "
              f"pred={label} | true={truth} {ok} | {lat_ms:.2f}ms")
        time.sleep(0.005)

    n = min(200, len(y_te))
    print(f"\nAccuracy on {n} samples: {correct/n:.4f}")
    print(f"Avg latency: {np.mean(latencies):.3f} ms  "
          f"p95: {np.percentile(latencies,95):.3f} ms")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time UWB NLOS inference for DWM1001 system"
    )
    parser.add_argument("--port",    default=config.SERIAL_PORT)
    parser.add_argument("--model",   default="onnx", choices=["onnx", "rf"])
    parser.add_argument("--log",     default=None,   help="CSV log file path")
    parser.add_argument("--replay",  default=None,
                        help="Offline replay: 'DEMO' or path to CSV")
    args = parser.parse_args()

    if args.replay is not None:
        replay_demo(args)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
