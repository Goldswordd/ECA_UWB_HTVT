"""
Si et al., "A Lightweight CIR-Based CNN With MLP for NLOS/LOS Identification
in a UWB Positioning System," IEEE Communications Letters, vol. 27, no. 5,
pp. 1332–1336, May 2023.

Architecture (end-to-end, original paper uses 2-stage training):
  CIR (B,1016) ──► Conv1d(6,k=5) ──► AvgPool(3) ──► Conv1d(12,k=6) ──►
                   AvgPool(4) ──► Flatten ──► Linear ──► scalar (B,1) ─┐
                                                                        cat
  Aux (B,11)   ──────────────────────────────────────────────────────── ┘
               (B, 12) ──► Linear(5) ──► ReLU ──► Linear(5) ──► ReLU ──►
               Linear(1) ──► logit (B,)

CIR length flow for cir_len=1016:
  1016 -Conv1(k=5)→ 1012 -Pool(k=3,s=3)→ 337 -Conv2(k=6)→ 332 -Pool(k=4,s=4)→ 83
  FCL input: 83 × 12 = 996
"""

import torch
import torch.nn as nn


def _fcl_size(cir_len: int) -> int:
    L = cir_len - 4   # conv1(k=5): cir_len - 5 + 1
    L = L // 3        # pool(k=3, s=3)
    L = L - 5         # conv2(k=6): L - 6 + 1
    L = L // 4        # pool(k=4, s=4)
    return L * 12


class CIREncoder(nn.Module):
    """1D CNN on full raw CIR → single scalar feature (B,1)."""

    def __init__(self, cir_len: int = 1016):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  6,  kernel_size=5)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(6,  12, kernel_size=6)
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.fc    = nn.Linear(_fcl_size(cir_len), 1)

    def forward(self, cir: torch.Tensor) -> torch.Tensor:
        """cir: (B, cir_len) → (B, 1)"""
        x = cir.unsqueeze(1)               # (B, 1, L)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        return self.fc(x.flatten(1))       # (B, 1)


class Si2023Net(nn.Module):
    """
    Full Si2023 model.

    Args
    ----
    cir_len  : full CIR length (1016 for eWINE/DWM1000)
    n_manual : number of manual/engineered features (11 in our setup,
               paper uses 6 = 3 proposed + 3 existing)
    """

    def __init__(self, cir_len: int = 1016, n_manual: int = 11):
        super().__init__()
        self.encoder = CIREncoder(cir_len)
        mlp_in = 1 + n_manual               # CNN scalar + manual features
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 5), nn.ReLU(),
            nn.Linear(5, 5),      nn.ReLU(),
            nn.Linear(5, 1),
        )

    def forward(self, cir: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        cir  : (B, cir_len)
        feat : (B, n_manual)
        → logit (B,)
        """
        z = torch.cat([self.encoder(cir), feat], dim=1)   # (B, 1+n_manual)
        return self.mlp(z).squeeze(-1)                     # (B,)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
