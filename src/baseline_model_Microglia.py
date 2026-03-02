"""Baseline model for Microglia: CERAD and Braak as two classification heads (no regression)."""

import torch
import torch.nn as nn

# CERAD class order: moderate before frequent (used for string -> index mapping in training)
CERAD_CLASS_ORDER = ["none", "sparse", "moderate", "frequent"]
# Braak values in data are 2, 3, 5, 6 -> map to indices 0, 1, 2, 3
BRAAK_TO_IDX = {2: 0, 3: 1, 5: 2, 6: 3}


class BaselineRankingModelMicroglia(nn.Module):
    """Baseline model with two classification heads: CERAD and Braak (both categories)."""

    def __init__(
        self,
        input_dim=16,
        input_shape=None,
        n_cerad_classes=None,
        n_braak_classes=None,
        use_attention=False,
        num_heads=1,
        dropout=0.2,
        n_hidden_layers=1,
        hidden_dim_3layer=32,
    ):
        super(BaselineRankingModelMicroglia, self).__init__()

        self.use_attention = use_attention
        self.n_hidden_layers = n_hidden_layers

        if input_shape is not None and len(input_shape) == 2:
            n_genes, n_dims = input_shape
            flattened_dim = n_genes * n_dims
            self.flatten_input = nn.Flatten()

            if use_attention:
                self.attention = nn.MultiheadAttention(
                    embed_dim=n_dims,
                    num_heads=num_heads,
                    batch_first=True,
                    dropout=0.1,
                )
            else:
                self.attention = None
            first_layer_input = flattened_dim
        else:
            self.flatten_input = nn.Identity()
            self.attention = None
            first_layer_input = input_dim

        if n_hidden_layers == 1:
            hidden_dim = 4  # keep small for low sample count
            self.shared = nn.Sequential(
                nn.Linear(first_layer_input, 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(8, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        elif n_hidden_layers == 3:
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
            hidden_dim = max(h // 8, 1)
        else:
            raise ValueError("n_hidden_layers must be 1 or 3")

        self._hidden_dim = hidden_dim

        if n_cerad_classes is None or n_braak_classes is None:
            raise ValueError("n_cerad_classes and n_braak_classes must be specified")
        self.cerad_head = nn.Linear(self._hidden_dim, n_cerad_classes)
        self.braak_head = nn.Linear(self._hidden_dim, n_braak_classes)

    def forward(self, x):
        if self.use_attention and self.attention is not None and len(x.shape) == 3:
            x_attn, _ = self.attention(x, x, x)
            x = self.flatten_input(x_attn)
        else:
            x = self.flatten_input(x)

        shared_features = self.shared(x)
        cerad_pred = self.cerad_head(shared_features)
        braak_pred = self.braak_head(shared_features)
        return cerad_pred, braak_pred
