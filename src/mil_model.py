"""
Cell-level Multiple Instance Learning (MIL) model.

Each patient is a bag of cells. A shared encoder maps each cell to a hidden vector;
gated attention (Ilse et al.) aggregates cells into a patient embedding, then ptau / Thal heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    if n_layers < 1:
        raise ValueError("mil_encoder_layers must be >= 1")
    layers: list[nn.Module] = []
    d_in = in_dim
    for layer_i in range(n_layers - 1):
        layers.extend(
            [
                nn.Linear(d_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        d_in = hidden_dim
    layers.append(nn.Linear(d_in, hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MILModel(nn.Module):
    """
    Args:
        cell_input_dim: flattened cell size (n_genes * n_dims) or n_dims after gene pooling
        mil_hidden_dim: encoder output dim per cell
        mil_attn_dim: attention bottleneck
        mil_encoder_layers: number of linear blocks in cell encoder (>=1)
        n_thal_classes: K for nominal CE
        dropout: dropout on encoder
        thal_ordinal: if True, thal head outputs K-1 logits (BCE)
    """

    def __init__(
        self,
        cell_input_dim: int,
        n_thal_classes: int,
        mil_hidden_dim: int = 128,
        mil_attn_dim: int = 64,
        mil_encoder_layers: int = 2,
        dropout: float = 0.2,
        thal_ordinal: bool = False,
    ):
        super().__init__()
        self.cell_input_dim = cell_input_dim
        self.mil_hidden_dim = mil_hidden_dim
        self.mil_attn_dim = mil_attn_dim
        self.thal_ordinal = bool(thal_ordinal)
        self.n_thal_classes = n_thal_classes

        self.cell_encoder = _build_mlp(
            cell_input_dim, mil_hidden_dim, mil_encoder_layers, dropout
        )

        self.attention_v = nn.Linear(mil_hidden_dim, mil_attn_dim)
        self.attention_u = nn.Linear(mil_hidden_dim, mil_attn_dim)
        self.attention_w = nn.Linear(mil_attn_dim, 1)

        self.ptau_head = nn.Linear(mil_hidden_dim, 1)
        thal_out = (n_thal_classes - 1) if self.thal_ordinal else n_thal_classes
        self.thal_head = nn.Linear(mil_hidden_dim, thal_out)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B, L, G, D) no pooling or (B, L, F) after gene pooling
        mask: optional (B, L) bool, True = valid cell ; False = pad (eval only)
        Returns: (ptau_pred, thal_logits)
        """
        if x.dim() == 4:
            b, ell, g, d = x.shape
            h_in = x.reshape(b * ell, g * d)
            reshape_back = (b, ell, self.mil_hidden_dim)
        elif x.dim() == 3:
            b, ell, f = x.shape
            h_in = x.reshape(b * ell, f)
            reshape_back = (b, ell, self.mil_hidden_dim)
        else:
            raise ValueError(f"MILModel expects x dim 3 or 4, got {x.dim()}")

        h = self.cell_encoder(h_in).view(*reshape_back)

        if mask is not None:
            h = h * mask.unsqueeze(-1).float()

        a = torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
        a = self.attention_w(a).squeeze(-1)

        if mask is not None:
            a = a.masked_fill(~mask, -1e9)

        a = torch.softmax(a, dim=1)
        patient_vec = (a.unsqueeze(-1) * h).sum(dim=1)

        ptau_pred = self.ptau_head(patient_vec).squeeze(-1)
        thal_logits = self.thal_head(patient_vec)
        return ptau_pred, thal_logits
