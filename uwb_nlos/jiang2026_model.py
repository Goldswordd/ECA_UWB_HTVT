"""
Jiang et al., "Optimizing UWB CIR Truncation for NLOS Identification on MCU
Deployment," IEEE Communications Letters, vol. 30, pp. 1091-1095, 2026.
DOI: 10.1109/LCOMM.2026.3662481

Architecture (parallel dual-channel):
  CIR (B, 63) ──► MultiScaleBlock (kernels=[3,5,7,9,11,13,15,17], 4 filters ea.)
                 → (B, 32, 63)
                 ──► DeepBlock(k=7) + MaxPool(2) → (B, 32, 31)
                 ──► DeepBlock(k=5) + MaxPool(2) → (B, 32, 15) ──┐
                 → GAP ──────────────────────────────── (B, 32)  │ residual
                                                                  │
                 ──► Self-Attention(h=8, d=32) → GAP ── (B, 32)  │
  Aux (B, 8) ──► Dense(32) + ReLU ─────────────────── (B, 32)   │
                                                                   │
  Concat[(B,32)+(B,32)+(B,32)] = (B, 96)                        │
  ──► Dense(64)+ReLU+Dropout(0.4)                                │
  ──► Dense(8) +ReLU+Dropout(0.2)                                │
  ──► Dense(1) → logit (B,)                                      │

CIR truncation: X=6 look-back, Y=63 total
  window = CIR[FP_IDX-6 : FP_IDX+57]

8 Auxiliary features (by RF importance, Fig.5):
  RXPACC, RANGE, CIR_PWR, FP_AMP3, MAX_NOISE, FP_AMP2, FP_AMP1, STDEV_NOISE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Sub-modules ───────────────────────────────────────────────────────────────

class MultiScaleBlock(nn.Module):
    """
    Parallel convolutions with kernels [3,5,7,9,11,13,15,17], 4 filters each.
    Concatenated output: (B, 32, L).
    """
    KERNELS = [3, 5, 7, 9, 11, 13, 15, 17]

    def __init__(self, in_ch: int = 1, n_filters: int = 4):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, n_filters, k, padding=k // 2)
            for k in self.KERNELS
        ])
        self.bn = nn.BatchNorm1d(len(self.KERNELS) * n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_ch, L) → (B, 32, L)"""
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        return F.relu(self.bn(out))


class DeepBlock(nn.Module):
    """Conv1d + BN + ReLU + MaxPool(2,2)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class FocalLoss(nn.Module):
    """
    Binary Focal Loss  FL = -α_t · (1-p_t)^γ · log(p_t)
    Default: γ=2, α=0.8 (weight for NLOS positive class).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t     = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss    = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return loss.mean()


# ── Main model ────────────────────────────────────────────────────────────────

class Jiang2026Net(nn.Module):
    """
    Full Jiang 2026 network (PyTorch, CPU-friendly).
    Inputs:
      cir  : (B, cir_len=63)  — truncated CIR, standardized
      aux  : (B, n_aux=8)     — 8 physical diagnostic parameters, standardized
    Output:
      logit (B,)
    """

    def __init__(
        self,
        cir_len:  int   = 63,
        n_aux:    int   = 8,
        d_model:  int   = 32,
        n_heads:  int   = 8,
        dropout:  float = 0.4,
        dropout2: float = 0.2,
    ):
        super().__init__()
        self.cir_len = cir_len

        # ── CIR branch ──────────────────────────────────────────────────────
        self.ms_block = MultiScaleBlock(in_ch=1, n_filters=4)   # → (B, 32, L)
        self.deep1    = DeepBlock(32, d_model, kernel_size=7)    # → (B, 32, L//2)
        self.deep2    = DeepBlock(d_model, d_model, kernel_size=5)  # → (B, 32, L//4)
        self.gap_cnn  = nn.AdaptiveAvgPool1d(1)                  # residual (B, 32)

        # ── Self-Attention branch ────────────────────────────────────────────
        self.attn     = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=0.1, batch_first=True
        )
        self.gap_attn = nn.AdaptiveAvgPool1d(1)                  # (B, 32)

        # ── Aux branch (Params Mapping) ──────────────────────────────────────
        self.params_mlp = nn.Sequential(
            nn.Linear(n_aux, d_model),
            nn.ReLU(),
        )

        # ── Classifier head (3 × d_model = 96 → 64 → 8 → 1) ────────────────
        self.classifier = nn.Sequential(
            nn.Linear(3 * d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 8),           nn.ReLU(), nn.Dropout(dropout2),
            nn.Linear(8, 1),
        )

    def forward(
        self,
        cir: torch.Tensor,
        aux: torch.Tensor,
    ) -> torch.Tensor:
        """
        cir : (B, cir_len)
        aux : (B, n_aux)
        → logit (B,)
        """
        # ── CIR branch ──────────────────────────────────────────────────────
        x = cir.unsqueeze(1)              # (B, 1, L)
        x = self.ms_block(x)              # (B, 32, L)
        x = self.deep1(x)                 # (B, 32, L//2)
        x = self.deep2(x)                 # (B, 32, L//4)

        # GAP residual (from CNN)
        feat_cnn  = self.gap_cnn(x).squeeze(-1)   # (B, 32)

        # ── Self-Attention ───────────────────────────────────────────────────
        seq = x.permute(0, 2, 1)                  # (B, T, 32)
        attn_out, _ = self.attn(seq, seq, seq)    # (B, T, 32)
        feat_attn = self.gap_attn(
            attn_out.permute(0, 2, 1)             # (B, 32, T)
        ).squeeze(-1)                             # (B, 32)

        # ── Aux branch ───────────────────────────────────────────────────────
        feat_aux = self.params_mlp(aux)            # (B, 32)

        # ── Fusion + classification ──────────────────────────────────────────
        fused = torch.cat([feat_cnn, feat_attn, feat_aux], dim=1)  # (B, 96)
        return self.classifier(fused).squeeze(-1)                   # (B,)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
