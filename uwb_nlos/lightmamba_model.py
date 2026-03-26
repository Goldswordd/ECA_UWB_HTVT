"""
LightMamba: Pure-PyTorch implementation of the architecture described in

  Wang et al., "LightMamba: A Resource-Efficient Deep-Learning Method for UWB
  NLOS Identification Based on Selective State-Space Modeling,"
  IEEE Internet of Things Journal, vol. 13, no. 5, pp. 8735-8748, 2026.

No CUDA / mamba-ssm package needed.  Works on CPU.

Key design choices (from paper, Table I-II):
  • eWINE: CIR_len=1016, T=8 segments (127 pts each), 14 aux features
  • coding_dim=32, 1 Mamba layer  →  ~25.8k parameters
  • Mamba d_state=16, d_conv=4, expand=2  (d_inner=64)
  • Dropout=0.3, AdamW(lr=1e-3, wd=1e-4), BCEWithLogitsLoss, patience=30

Architecture flow (Fig. 3):
  CIR (B,1016) ──► segment(B,T,127) ──► Linear ──► LN ──► GELU ──► (B,T,d) ─┐
                                                                               ⊕ fused (B,T,d)
  Aux (B,14)   ──► Linear ──► LN ──► GELU ──► expand T ──► (B,T,d) ──────────┘
                                                                              │
                                               LN + Mamba + Dropout + residual│
                                                                              │
                                      AdaptiveAvgPool1d(1) ──► squeeze (B,d) │
                                      Middle: Linear+LN+GELU+Dropout          │
                                      Classifier: BN+Linear+GELU+Drop+Linear  │
                                                                              ▼
                                                                         logit (B,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Selective SSM (Mamba core)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Input-selective state-space model (Mamba-style).

    Processes a sequence (B, L, d_model).
    For L=T=8 (after CIR segmentation), the sequential scan is trivial on CPU.

    Parameters follow Gu & Dao, "Mamba: Linear-time sequence modeling
    with selective state spaces," arXiv:2312.00752.
    """

    def __init__(self, d_model: int = 32, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.d_inner = int(expand * d_model)   # 64 for default

        # ── input projection: split into x and gating branch z ──────────
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # ── depthwise causal conv over sequence dim ──────────────────────
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,   # causal: trim right later
            bias=True,
        )

        # ── input-dependent SSM projections: Δ, B, C ────────────────────
        # x_proj → (1 for Δ_raw, d_state for B, d_state for C)
        self.x_proj  = nn.Linear(self.d_inner, 1 + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # ── A matrix: fixed diagonal, learned as log for stability ───────
        # Init: A_{n} = n  (n=1..d_state), per Mamba paper
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)   # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # ── D: skip-connection scalar per channel ────────────────────────
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ── output projection back to d_model ────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, d_model)
        Returns (B, L, d_model)
        """
        B, L, _ = x.shape
        d_inner  = self.d_inner
        d_state  = self.d_state

        # ── 1. Project + gate ────────────────────────────────────────────
        xz      = self.in_proj(x)            # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)       # each (B, L, d_inner)

        # ── 2. Causal depthwise conv ─────────────────────────────────────
        x_conv = x_in.transpose(1, 2)        # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal trim
        x_conv = x_conv.transpose(1, 2)      # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # ── 3. Input-dependent Δ, B, C ───────────────────────────────────
        ssm_in   = self.x_proj(x_conv)       # (B, L, 1 + 2*d_state)
        dt_raw   = ssm_in[:, :, :1]          # (B, L, 1)
        B_ssm    = ssm_in[:, :, 1:1+d_state] # (B, L, d_state)
        C_ssm    = ssm_in[:, :, 1+d_state:]  # (B, L, d_state)

        delta = F.softplus(self.dt_proj(dt_raw))  # (B, L, d_inner)

        # ── 4. Discretize A, B  (ZOH) ────────────────────────────────────
        A = -torch.exp(self.A_log.float())   # (d_inner, d_state), always negative

        # deltaA : (B, L, d_inner, d_state)
        deltaA   = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )
        # deltaB_u : (B, L, d_inner, d_state)
        deltaB_u = (
            delta.unsqueeze(-1)
            * B_ssm.unsqueeze(2)
            * x_conv.unsqueeze(-1)
        )

        # ── 5. Sequential scan (L=8 → ~8 small matmuls, fast on CPU) ────
        h   = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys  = []
        for i in range(L):
            h  = deltaA[:, i] * h + deltaB_u[:, i]   # (B, d_inner, d_state)
            y  = (h * C_ssm[:, i].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y)

        y = torch.stack(ys, dim=1)   # (B, L, d_inner)

        # ── 6. Skip connection + SiLU gate ───────────────────────────────
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.out_proj(y)      # (B, L, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Full LightMamba model
# ─────────────────────────────────────────────────────────────────────────────

class LightMamba(nn.Module):
    """
    LightMamba for UWB NLOS identification.

    Args
    ----
    cir_len     : total CIR length (1016 for eWINE/DWM1000)
    aux_dim     : number of auxiliary channel features (14 for eWINE)
    T           : number of CIR time-segments  (8 for eWINE)
    coding_dim  : latent / coding dimension     (32 for optimal config)
    num_layers  : number of stacked Mamba layers (1 is optimal per paper)
    d_state     : SSM state dimension            (16)
    dropout     : dropout probability            (0.3)
    """

    def __init__(
        self,
        cir_len:    int = 1016,
        aux_dim:    int = 14,
        T:          int = 8,
        coding_dim: int = 32,
        num_layers: int = 1,
        d_state:    int = 16,
        dropout:    float = 0.3,
    ):
        super().__init__()
        assert cir_len % T == 0, \
            f"cir_len ({cir_len}) must be divisible by T ({T})"

        self.T        = T
        self.seg_len  = cir_len // T    # 127 for eWINE
        self.coding_dim = coding_dim

        # ── CIR branch ──────────────────────────────────────────────────
        # Eq. (5): reshape → (B,T,seg_len)
        # Eq. (6): linear + LN + GELU
        self.cir_proj = nn.Linear(self.seg_len, coding_dim, bias=False)
        self.cir_norm = nn.LayerNorm(coding_dim)
        self.cir_drop = nn.Dropout(dropout)

        # ── Aux branch ──────────────────────────────────────────────────
        # Eq. (7): linear + LN + GELU
        # Eq. (8): repeat T times
        self.aux_proj = nn.Linear(aux_dim, coding_dim, bias=False)
        self.aux_norm = nn.LayerNorm(coding_dim)
        self.aux_drop = nn.Dropout(dropout)

        # ── Mamba blocks with residual (Eq. 10) ─────────────────────────
        self.mamba_norms  = nn.ModuleList([nn.LayerNorm(coding_dim)  for _ in range(num_layers)])
        self.mamba_layers = nn.ModuleList([SelectiveSSM(coding_dim, d_state=d_state) for _ in range(num_layers)])
        self.mamba_drops  = nn.ModuleList([nn.Dropout(dropout)       for _ in range(num_layers)])

        # ── Middle block ─────────────────────────────────────────────────
        # AdaptiveAvgPool1d(1) → squeeze → Linear → LN → GELU → Dropout
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.mid_lin  = nn.Linear(coding_dim, coding_dim)
        self.mid_norm = nn.LayerNorm(coding_dim)
        self.mid_drop = nn.Dropout(dropout)

        # ── Classifier ───────────────────────────────────────────────────
        # BN → Linear(d, d/2) → GELU → Dropout → Linear(d/2, 1)
        self.bn      = nn.BatchNorm1d(coding_dim)
        self.cls_fc1 = nn.Linear(coding_dim, coding_dim // 2)
        self.cls_drop = nn.Dropout(dropout)
        self.cls_fc2 = nn.Linear(coding_dim // 2, 1)

    # ------------------------------------------------------------------
    def forward(self, cir: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        cir : (B, cir_len)  — full normalized CIR
        aux : (B, aux_dim)  — auxiliary features (standardized)
        Returns logit (B,)  — use BCEWithLogitsLoss during training
        """
        B = cir.shape[0]

        # CIR branch  ────────────────────────────────────────────────────
        x_seg  = cir.view(B, self.T, self.seg_len)           # (B, T, seg_len)
        h_cir  = F.gelu(self.cir_norm(self.cir_proj(x_seg))) # (B, T, d)
        h_cir  = self.cir_drop(h_cir)

        # Aux branch  ────────────────────────────────────────────────────
        h_aux  = F.gelu(self.aux_norm(self.aux_proj(aux)))   # (B, d)
        h_aux  = self.aux_drop(h_aux)
        h_aux  = h_aux.unsqueeze(1).expand(-1, self.T, -1)   # (B, T, d)

        # Fusion (Eq. 9): element-wise addition ─────────────────────────
        h = h_cir + h_aux                                    # (B, T, d)

        # Mamba blocks (Eq. 10): LN → Mamba → Dropout + residual ────────
        for norm, mamba, drop in zip(self.mamba_norms, self.mamba_layers, self.mamba_drops):
            h = h + drop(mamba(norm(h)))

        # Temporal pooling: (B, T, d) → (B, d) ──────────────────────────
        h = self.pool(h.transpose(1, 2)).squeeze(-1)         # (B, d)

        # Middle block ───────────────────────────────────────────────────
        h = F.gelu(self.mid_norm(self.mid_lin(h)))
        h = self.mid_drop(h)

        # Classifier ─────────────────────────────────────────────────────
        h  = F.gelu(self.cls_fc1(self.bn(h)))
        h  = self.cls_drop(h)
        return self.cls_fc2(h).squeeze(-1)                   # (B,) logit

    # ------------------------------------------------------------------
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
