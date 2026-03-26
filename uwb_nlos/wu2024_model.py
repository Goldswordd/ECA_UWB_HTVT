"""
Wu et al., "Self-Attention-Assisted TinyML With Effective Representation for
UWB NLOS Identification," IEEE Internet of Things Journal, vol. 11, no. 15,
pp. 25471-25480, Aug. 2024.  DOI: 10.1109/JIOT.2024.3349462

Two-stage training:
  Stage 1 — Pretrain a 5-FC-layer MLP on the 57-D mixed dataset
             (50-seq CIR + 7 channel characteristics)
  Stage 2 — Freeze first 3 layers of pretrained MLP; attach self-attention
             module and retrain a new trimmed classifier

57-D mixed dataset (Fig. 3 of paper):
  CIR sequence : 50 samples from FP_INDEX-2 to FP_INDEX+47
  Channel chars: Distance, MaxNoise, StdNoise, FP_AMP1, FP_AMP2, FP_AMP3, CIR_PWR
                 (7 raw DW1000 diagnostics)

Architecture after stage 2 (Fig. 4b):
  frozen{Dense(57→30,linear) → BN → Dense(30→16,ReLU)}
    → Q=Dense(16→16)+BN, K=Dense(16→16)+BN, V=Dense(16→16)+BN
    → scaled dot-product attention over 16 feature "tokens"
    → Dense(16→42,ReLU) → Dropout(0.4) → Dense(42→16,ReLU) → Dropout(0.2) → Dense(16→1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared encoder (first 3 "layers" of pretrained MLP) ───────────────────────

class FrozenEncoder(nn.Module):
    """
    Layers 1-3 of the pretrained MLP (frozen in Stage 2).
      Dense(57→30, linear) → BN(30) → Dense(30→16, ReLU)
    Output: (B, 16)
    """
    def __init__(self, in_dim: int = 57):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 30)
        self.bn1 = nn.BatchNorm1d(30)
        self.fc2 = nn.Linear(30, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.fc1(x))      # linear activation → BN
        return F.relu(self.fc2(x))      # ReLU


# ── Stage 1: Full pretrained MLP (5 FC + 3 BN) ────────────────────────────────

class PretrainedMLP(nn.Module):
    """
    Stage-1 pretrained classifier (5 FC layers + 3 BN layers).
    Mirrors Fig. 4(a) of the paper.
    Input : (B, 57)
    Output: logit (B,)
    """
    def __init__(self, in_dim: int = 57, dropout: float = 0.2):
        super().__init__()
        self.encoder = FrozenEncoder(in_dim)          # layers 1-3 → (B, 16)
        self.fc3     = nn.Linear(16, 64)
        self.bn3     = nn.BatchNorm1d(64)
        self.fc4     = nn.Linear(64, 16)
        self.bn4     = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(dropout)
        self.fc5     = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        return self.fc5(x).squeeze(-1)   # logit (B,)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Stage 2: Self-attention + trimmed classifier ───────────────────────────────

class Wu2024Net(nn.Module):
    """
    Stage-2 model (Fig. 4b):
      • Frozen encoder (weights loaded from Stage 1, requires_grad=False)
      • Self-attention over 16 feature "tokens" (each token = 1 scalar):
          Q, K, V each come from Linear(16→16)+BN applied to the encoder output.
          Treats (B,16) as a sequence (B, 16, 1); computes (16×16) attention map.
      • Trimmed classifier: Dense(16→42,ReLU)+Drop → Dense(42→16,ReLU)+Drop → Dense(16→1)

    The scaled dot-product attention re-weights the 16 extracted features
    by learning their pairwise relevance — analogous to Eq. (2) in the paper.
    """

    def __init__(
        self,
        in_dim:   int   = 57,
        dropout1: float = 0.4,
        dropout2: float = 0.2,
    ):
        super().__init__()

        # ── Frozen encoder ──────────────────────────────────────────────────
        self.encoder = FrozenEncoder(in_dim)

        # ── Q, K, V projections (each Linear(16→16) + BN) ──────────────────
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.bn_q   = nn.BatchNorm1d(16)
        self.bn_k   = nn.BatchNorm1d(16)
        self.bn_v   = nn.BatchNorm1d(16)

        # ── Trimmed classifier ───────────────────────────────────────────────
        self.clf = nn.Sequential(
            nn.Linear(16, 42), nn.ReLU(), nn.Dropout(dropout1),
            nn.Linear(42, 16), nn.ReLU(), nn.Dropout(dropout2),
            nn.Linear(16, 1),
        )

    def freeze_encoder(self):
        """Call after loading pretrained encoder weights to freeze them."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _self_attention(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat : (B, 16)
        Treat each of the 16 features as a 1-D token → sequence (B, 16, 1).
        Compute scaled dot-product attention → context (B, 16).
        """
        Q = F.relu(self.bn_q(self.q_proj(feat))).unsqueeze(-1)   # (B, 16, 1)
        K = F.relu(self.bn_k(self.k_proj(feat))).unsqueeze(-1)   # (B, 16, 1)
        V = F.relu(self.bn_v(self.v_proj(feat))).unsqueeze(-1)   # (B, 16, 1)

        # scores_{i,j} = Q_i * K_j / sqrt(1)  → (B, 16, 16)
        scores = torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(1.0)
        attn   = torch.softmax(scores, dim=-1)           # (B, 16, 16)
        ctx    = torch.bmm(attn, V).squeeze(-1)          # (B, 16)
        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)                   # (B, 16)
        ctx  = self._self_attention(feat)        # (B, 16)
        return self.clf(ctx).squeeze(-1)         # logit (B,)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_params_total(self) -> int:
        return sum(p.numel() for p in self.parameters())
