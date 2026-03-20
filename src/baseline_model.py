"""Baseline ranking model: ptau regression + thal classification."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def build_mlp_trunk(
    first_layer_input: int,
    hidden_sizes: List[int],
    dropout: float,
) -> Tuple[nn.Sequential, int]:
    """Linear stack: in -> h0 -> h1 -> ... with ReLU + dropout after each hidden layer."""
    if not hidden_sizes:
        raise ValueError("hidden_sizes must be non-empty for custom MLP")
    layers: List[nn.Module] = []
    prev = first_layer_input
    for h in hidden_sizes:
        if h < 1:
            raise ValueError(f"hidden_sizes must be positive ints, got {h}")
        layers.extend(
            [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        prev = h
    return nn.Sequential(*layers), prev


class BaselineRankingModel(nn.Module):
    """Baseline model with two heads: ptau regression and thal classification."""

    def __init__(
        self,
        input_dim=16,
        input_shape=None,
        n_thal_classes=None,
        use_attention=False,
        num_heads=1,
        dropout=0.2,
        n_hidden_layers=1,
        hidden_dim_3layer=32,
        mlp_hidden_sizes: Optional[List[int]] = None,
        thal_ordinal=False,
    ):
        super(BaselineRankingModel, self).__init__()

        self.use_attention = use_attention
        self.n_hidden_layers = n_hidden_layers
        self.thal_ordinal = bool(thal_ordinal)
        self.n_thal_classes = n_thal_classes

        # Handle 2D input (n_genes, n_dims) or 1D input (flattened)
        if input_shape is not None and len(input_shape) == 2:
            # 2D input: (n_genes, n_dims) - can apply attention on genes
            n_genes, n_dims = input_shape
            flattened_dim = n_genes * n_dims
            self.flatten_input = nn.Flatten()

            if use_attention:
                # Self-attention on gene dimension before flattening
                # Input: (batch, n_genes, n_dims), Output: (batch, n_genes, n_dims)
                self.attention = nn.MultiheadAttention(
                    embed_dim=n_dims,
                    num_heads=num_heads,
                    batch_first=True,
                    dropout=0.1
                )
            else:
                self.attention = None
            # After attention (if used), flatten to (batch, n_genes * n_dims)
            first_layer_input = flattened_dim
        else:
            # 1D input: already flattened - attention not applicable
            self.flatten_input = nn.Identity()
            self.attention = None
            first_layer_input = input_dim

        if mlp_hidden_sizes is not None and len(mlp_hidden_sizes) > 0:
            self.shared, hidden_dim = build_mlp_trunk(
                first_layer_input, mlp_hidden_sizes, dropout
            )
        elif n_hidden_layers == 1:
            # Single hidden layer; size from hidden_dim_3layer (e.g. 32)
            hidden_dim = hidden_dim_3layer
            self.shared = nn.Sequential(
                nn.Linear(first_layer_input, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif n_hidden_layers == 3:
            # Original 3-layer MLP (shrinking: h//2 -> h//4 -> h//8)
            h = hidden_dim_3layer
            self.shared = nn.Sequential(
                nn.Linear(first_layer_input, h // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h // 2, h // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h // 4, h // 8),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            hidden_dim = max(h // 8, 1)  # last layer output size (ensure >= 1)
        else:
            raise ValueError("n_hidden_layers must be 1 or 3")

        self._hidden_dim = hidden_dim

        # Head 1: P-tau regression (ReLU output for log1p(p-tau)), already have log1p applied in the data
        self.ptau_head = nn.Linear(self._hidden_dim, 1)

        if n_thal_classes is None:
            raise ValueError("n_thal_classes must be specified")
        # Ordinal: K-1 cumulative logits (P(Y>=k)); else K-way classification
        thal_out = (n_thal_classes - 1) if self.thal_ordinal else n_thal_classes
        self.thal_head = nn.Linear(self._hidden_dim, thal_out)

    def forward(self, x):
        # Apply attention only for 2D input (batch, n_genes, n_dims)
        if self.use_attention and self.attention is not None and len(x.shape) == 3:
            # 2D input: (batch, n_genes, n_dims) - apply attention on gene dimension
            x_attn, _ = self.attention(x, x, x)  # Self-attention
            x = self.flatten_input(x_attn)
        else:
            # Flatten if input is 2D, otherwise pass through
            x = self.flatten_input(x)

        shared_features = self.shared(x)
        ptau_pred = self.ptau_head(shared_features).squeeze(-1)
        thal_pred = self.thal_head(shared_features)  # (batch_size, n_thal_classes) logits (no softmax; CrossEntropyLoss applies it)
        return ptau_pred, thal_pred
