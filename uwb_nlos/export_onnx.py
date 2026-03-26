"""
Export trained CNN to ONNX for lightweight inference on Raspberry Pi.

ONNX + onnxruntime is the recommended deployment path:
  - No need to install PyTorch on RPi
  - onnxruntime-linux-aarch64 supports Python 3.9-3.12
  - ~3-5x faster than PyTorch CPU on ARM

Usage:
  python export_onnx.py               # export from models/best_model.pt
  python export_onnx.py --validate    # run a quick accuracy check after export
  python export_onnx.py --quantize    # INT8 dynamic quantization (even smaller)
"""
import argparse
import pickle
import time

import numpy as np

import config


def export(model_pt_path=None, onnx_path=None, opset: int = 17):
    import torch
    from model import NLOSClassifier

    model_pt_path = model_pt_path or config.KERAS_MODEL_PATH.with_suffix(".pt")
    onnx_path     = onnx_path     or (config.MODEL_DIR / "nlos_classifier.onnx")

    if not model_pt_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_pt_path}")

    model = NLOSClassifier(n_feat=config.N_FEATURES)
    model.load_state_dict(torch.load(model_pt_path, map_location="cpu"))
    model.eval()
    model.export_onnx(str(onnx_path), opset=opset)

    size_kb = onnx_path.stat().st_size / 1024
    print(f"  File size: {size_kb:.1f} KB")
    return onnx_path


def quantize_int8(onnx_path=None, out_path=None):
    """
    Dynamic INT8 quantization (weights only — safe for CIR CNN).
    Reduces size ~4x and typically speeds up inference ~2x on CPU.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_path = onnx_path or (config.MODEL_DIR / "nlos_classifier.onnx")
    out_path  = out_path  or config.TFLITE_INT8_PATH.with_suffix(".onnx")
    out_path  = config.MODEL_DIR / "nlos_classifier_int8.onnx"

    quantize_dynamic(
        str(onnx_path),
        str(out_path),
        weight_type=QuantType.QUInt8,
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"INT8 ONNX saved → {out_path}  ({size_kb:.1f} KB)")
    return out_path


def validate(onnx_path=None):
    """Quick sanity check: run ONNX model on test set and print accuracy."""
    import onnxruntime as ort
    from preprocess import build_dataset
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    onnx_path = onnx_path or (config.MODEL_DIR / "nlos_classifier.onnx")
    splits    = build_dataset(use_oiud=False)
    cir_te, feat_te, y_te = splits[6], splits[7], splits[8]

    sess  = ort.InferenceSession(str(onnx_path),
                                  providers=["CPUExecutionProvider"])

    # Batch inference
    BATCH = 512
    probs_all = []
    for i in range(0, len(y_te), BATCH):
        out = sess.run(None, {
            "cir":  cir_te[i:i+BATCH],
            "feat": feat_te[i:i+BATCH],
        })[0]
        probs_all.append(out.squeeze(1))

    probs = np.concatenate(probs_all)
    preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds)
    auc = roc_auc_score(y_te, probs)
    print(f"\nONNX validation — acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # Latency on this machine
    cir1  = cir_te[:1]
    feat1 = feat_te[:1]
    for _ in range(50):
        sess.run(None, {"cir": cir1, "feat": feat1})
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        sess.run(None, {"cir": cir1, "feat": feat1})
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"Latency: mean={np.mean(times):.3f} ms  "
          f"p95={np.percentile(times,95):.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate",  action="store_true")
    parser.add_argument("--quantize",  action="store_true")
    args = parser.parse_args()

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    onnx_path = export()

    if args.quantize:
        try:
            quantize_int8(onnx_path)
        except ImportError:
            print("onnxruntime.quantization not available — skipping INT8.")

    if args.validate:
        validate(onnx_path)


if __name__ == "__main__":
    main()
