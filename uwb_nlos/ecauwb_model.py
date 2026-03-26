"""
ECA-UWB: Efficient Channel Attention for Lightweight UWB NLOS Identification
(Proposed architecture for Q1 paper)

Key innovations vs prior work:
  1. ECA module (Efficient Channel Attention, 3 params only) on CNN feature maps
     instead of costly SE/CBAM attention or heavy MHA.
  2. Gated feature fusion — learned dynamic weighting between CNN-CIR features
     and raw channel diagnostics (vs fixed concat or simple sum).
  3. End-to-end single-stage training (vs Wu2024's 2-stage pipeline).

Architecture:
  Input: 57-D = [50-pt CIR window | 7 raw channel diagnostics]

  CIR Branch (CNN + ECA):
    (B,1,50) → Conv1d(1→16, k=5) → BN → ReLU → MaxPool(2)
             → ECA(C=16, k=3)                      ← 3 params
             → Conv1d(16→16, k=3) → BN → ReLU → GAP → (B,16)

  Aux Branch (lightweight MLP):
    (B,7) → Linear(7→32) → ReLU → Linear(32→16) → ReLU → (B,16)

  Gated Fusion:
    g = σ( Linear(32→2)([f_cir; f_aux]) )          ← scalar per branch
    f_fused = g₁·f_cir + g₂·f_aux                  ← (B,16)

  Classifier:
    f_fused → Linear(16→32) → ReLU → Dropout(0.3) → Linear(32→1)

Total trainable params: ~2,438
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Efficient Channel Attention ─────────────────────────────────────────────

class ECA(nn.Module):
    """
    Efficient Channel Attention (Wang et al., ECA-Net, CVPR 2020).
    Applies 1-D conv of size k over channel GAP vector — only k params.

    Input : (B, C, L)
    Output: (B, C, L)  — channel-wise re-weighted
    """
    def __init__(self, k: int = 3):
        super().__init__()
        assert k % 2 == 1, "k must be odd for same-padding"
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        y = x.mean(dim=-1, keepdim=True)       # GAP: (B, C, 1)
        y = y.transpose(1, 2)                   # (B, 1, C)
        y = self.conv(y)                        # (B, 1, C)
        w = torch.sigmoid(y).transpose(1, 2)   # (B, C, 1)
        return x * w                            # (B, C, L)


# ── CIR Branch (CNN + ECA) ──────────────────────────────────────────────────

class CIRBranch(nn.Module):
    """
    Two-stage 1-D CNN with ECA attention on the intermediate feature map.
    Input : (B, 1, 50)
    Output: (B, out_ch)   — after Global Average Pooling
    """
    def __init__(self, in_len: int = 50, ch1: int = 16, ch2: int = 16,
                 k1: int = 5, k2: int = 3, eca_k: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(1,   ch1, k1, padding=k1 // 2)
        self.bn1   = nn.BatchNorm1d(ch1)
        self.pool  = nn.MaxPool1d(2)
        self.eca   = ECA(k=eca_k)
        self.conv2 = nn.Conv1d(ch1, ch2, k2, padding=k2 // 2)
        self.bn2   = nn.BatchNorm1d(ch2)
        self.out_dim = ch2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 50)
        x = F.relu(self.bn1(self.conv1(x)))    # (B, ch1, 50)
        x = self.pool(x)                        # (B, ch1, 25)
        x = self.eca(x)                         # (B, ch1, 25) — ECA
        x = F.relu(self.bn2(self.conv2(x)))    # (B, ch2, 25)
        x = x.mean(dim=-1)                     # GAP → (B, ch2)
        return x


# ── Aux Branch (MLP) ────────────────────────────────────────────────────────

class AuxBranch(nn.Module):
    """
    Lightweight 2-layer MLP for raw channel diagnostics.
    Input : (B, n_aux)
    Output: (B, out_dim)
    """
    def __init__(self, n_aux: int = 7, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_aux, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Gated Fusion ────────────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    """
    Learn a pair of scalar gates (g1, g2) from the concatenated features,
    then return g1·f_cir + g2·f_aux.

    Both branches must have the same output dimension (feat_dim).
    """
    def __init__(self, feat_dim: int = 16):
        super().__init__()
        self.gate = nn.Linear(feat_dim * 2, 2)

    def forward(self, f_cir: torch.Tensor, f_aux: torch.Tensor) -> torch.Tensor:
        # f_cir, f_aux: (B, feat_dim)
        g = torch.sigmoid(self.gate(torch.cat([f_cir, f_aux], dim=-1)))  # (B, 2)
        return g[:, 0:1] * f_cir + g[:, 1:2] * f_aux                    # (B, feat_dim)


# ── Full ECA-UWB Model ───────────────────────────────────────────────────────

class ECAUWBNet(nn.Module):
    """
    ECA-UWB: Efficient Channel Attention UWB NLOS Identifier.

    Input shape: (B, 57)  — [CIR(50) | Aux(7)]
    Output     : logit (B,)

    Hyper-parameters (defaults mirror the design in the Q1 paper proposal):
      cir_len   : CIR window length (50)
      n_aux     : number of raw channel diagnostics (7)
      ch1, ch2  : filters in first / second conv layer (16, 16)
      k1, k2    : kernel sizes (5, 3)
      eca_k     : ECA conv kernel (3 → 3 params)
      aux_hid   : hidden units in Aux MLP (32)
      feat_dim  : unified feature dimension for both branches (16)
      clf_hid   : hidden units in classifier MLP (32)
      dropout   : dropout rate before final linear (0.3)
    """

    def __init__(
        self,
        cir_len:  int   = 50,
        n_aux:    int   = 7,
        ch1:      int   = 16,
        ch2:      int   = 16,
        k1:       int   = 5,
        k2:       int   = 3,
        eca_k:    int   = 3,
        aux_hid:  int   = 32,
        feat_dim: int   = 16,
        clf_hid:  int   = 32,
        dropout:  float = 0.3,
    ):
        super().__init__()
        assert ch2 == feat_dim, "ch2 must equal feat_dim for gated fusion"

        self.cir_len = cir_len
        self.n_aux   = n_aux

        self.cir_branch = CIRBranch(cir_len, ch1, ch2, k1, k2, eca_k)
        self.aux_branch = AuxBranch(n_aux, aux_hid, feat_dim)
        self.fusion     = GatedFusion(feat_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, clf_hid), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hid, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 57)
        cir = x[:, :self.cir_len].unsqueeze(1)   # (B, 1, 50)
        aux = x[:, self.cir_len:]                 # (B, 7)

        f_cir = self.cir_branch(cir)              # (B, 16)
        f_aux = self.aux_branch(aux)              # (B, 16)
        fused = self.fusion(f_cir, f_aux)         # (B, 16)

        return self.classifier(fused).squeeze(-1) # logit (B,)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def branch_params(self) -> dict:
        return {
            "cir_branch":  sum(p.numel() for p in self.cir_branch.parameters()),
            "eca_only":    sum(p.numel() for p in self.cir_branch.eca.parameters()),
            "aux_branch":  sum(p.numel() for p in self.aux_branch.parameters()),
            "fusion":      sum(p.numel() for p in self.fusion.parameters()),
            "classifier":  sum(p.numel() for p in self.classifier.parameters()),
        }


# ── Ablation variants ────────────────────────────────────────────────────────

class ECAUWBNet_NoECA(ECAUWBNet):
    """Ablation A: replace ECA with identity (no channel attention)."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.cir_branch.eca = nn.Identity()


class ECAUWBNet_ConcatFusion(ECAUWBNet):
    """Ablation B: replace gated fusion with simple concatenation + linear."""
    def __init__(self, **kw):
        super().__init__(**kw)
        feat_dim = kw.get("feat_dim", 16)
        self.fusion     = nn.Identity()          # placeholder; overridden below
        self.fusion_lin = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cir = x[:, :self.cir_len].unsqueeze(1)
        aux = x[:, self.cir_len:]
        f_cir = self.cir_branch(cir)
        f_aux = self.aux_branch(aux)
        fused = F.relu(self.fusion_lin(torch.cat([f_cir, f_aux], dim=-1)))
        return self.classifier(fused).squeeze(-1)


class ECAUWBNet_NoBranch(ECAUWBNet):
    """Ablation C: CIR branch only (no aux features)."""
    def __init__(self, **kw):
        super().__init__(**kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cir   = x[:, :self.cir_len].unsqueeze(1)
        f_cir = self.cir_branch(cir)
        # skip aux + fusion; feed CIR features directly to classifier
        return self.classifier(f_cir).squeeze(-1)
