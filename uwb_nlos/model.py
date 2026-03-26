"""
Dual-Branch Lightweight 1D CNN for UWB NLOS Classification.

Architecture:
  ┌─ CIR Branch ─────────────────────────────────────────────┐
  │  Input (50, 1)                                            │
  │  Conv1d(1→16, k=5, pad=2) → BN → ReLU                    │
  │  MaxPool1d(2) → (25, 16)                                  │
  │  Conv1d(16→32, k=3, pad=1) → BN → ReLU                   │
  │  MaxPool1d(2) → (12, 32)                                  │
  │  Conv1d(32→32, k=3, pad=1) → BN → ReLU                   │
  │  GlobalAvgPool → (32,)                                    │
  └───────────────────────────────────────────────────────────┘
         ↓ concat (32+16=48)
  ┌─ Feature Branch ──────────────────────────────────────────┐
  │  Input (11,)                                              │
  │  Linear(11→32) → BN → ReLU                               │
  │  Linear(32→16) → BN → ReLU                               │
  │  → (16,)                                                  │
  └───────────────────────────────────────────────────────────┘
         ↓
  Fusion: Linear(48→32) → ReLU → Dropout(0.3)
           → Linear(32→1) → Sigmoid

  Total params: ~7 K  |  Target latency on RPi4: < 2 ms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CIRBranch(nn.Module):
    """1D CNN operating on the 50-sample CIR window."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                    # 50 → 25

            # Block 2
            nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                    # 25 → 12

            # Block 3  (no pool — preserves some spatial info)
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)      # (N, 32, 12) → (N, 32, 1)

    def forward(self, x):
        # x: (N, 50, 1) → permute to (N, 1, 50) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        x = self.gap(x).squeeze(-1)             # (N, 32)
        return x


class _FeatureBranch(nn.Module):
    """Small MLP operating on the 11 global diagnostic features."""

    def __init__(self, n_feat: int = 11):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_feat, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)                   # (N, 16)


class NLOSClassifier(nn.Module):
    """
    Dual-branch classifier.

    Inputs
    ------
    cir   : (N, 50, 1)  float32 — CIR window, normalised by RXPACC
    feat  : (N, 11)     float32 — global features, StandardScaler applied

    Output
    ------
    logit  : (N, 1)  raw logit  (use sigmoid for probability)
    """

    def __init__(self, n_feat: int = 11, dropout: float = 0.3):
        super().__init__()
        self.cir_branch  = _CIRBranch()
        self.feat_branch = _FeatureBranch(n_feat)
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, cir: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        z_cir  = self.cir_branch(cir)           # (N, 32)
        z_feat = self.feat_branch(feat)         # (N, 16)
        z      = torch.cat([z_cir, z_feat], dim=1)  # (N, 48)
        return self.fusion(z)                   # (N, 1) — raw logit

    def predict_proba(self, cir: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Return NLOS probability in [0, 1]."""
        return torch.sigmoid(self.forward(cir, feat))

    # ── ONNX export ──────────────────────────────────────────
    def export_onnx(self, path: str, opset: int = 17):
        """
        Export to ONNX for onnxruntime inference on RPi.
        The exported model takes two inputs: 'cir' and 'feat',
        and returns one output: 'nlos_prob' (NLOS probability).
        """
        self.eval()
        dummy_cir  = torch.zeros(1, 50, 1)
        dummy_feat = torch.zeros(1, 11)

        # Wrap to output sigmoid probability directly
        class _Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, c, f): return torch.sigmoid(self.m(c, f))

        torch.onnx.export(
            _Wrapper(self),
            (dummy_cir, dummy_feat),
            path,
            input_names   = ["cir", "feat"],
            output_names  = ["nlos_prob"],
            dynamic_axes  = {
                "cir":       {0: "batch"},
                "feat":      {0: "batch"},
                "nlos_prob": {0: "batch"},
            },
            opset_version = opset,
            dynamo        = False,   # use legacy TorchScript-based exporter
        )
        print(f"ONNX model saved → {path}")


# ─────────────────────────────────────────────────────────────
# Baseline models (scikit-learn, always available)
# ─────────────────────────────────────────────────────────────

def build_rf_baseline():
    """Random Forest baseline — fast, no GPU needed, works on RPi."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators  = 300,
        max_depth     = 20,
        min_samples_leaf = 4,
        n_jobs        = -1,
        random_state  = 42,
        class_weight  = "balanced",
    )


def build_svm_baseline():
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler as _SS
    return Pipeline([
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, random_state=42)),
    ])


# ─────────────────────────────────────────────────────────────
# Model summary helper
# ─────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = NLOSClassifier()
    print(model)
    print(f"\nTotal trainable parameters: {count_params(model):,}")

    cir  = torch.randn(4, 50, 1)
    feat = torch.randn(4, 11)
    prob = model.predict_proba(cir, feat)
    print(f"Output shape: {prob.shape}  values: {prob.detach().squeeze().tolist()}")
