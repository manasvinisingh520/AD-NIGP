"""
Gene encoder with softmax-weighted pooling.
For each gene g: w_g = softmax(MLP(gene_vec_g)); patient_vec = Σ_g w_g · gene_vec_g
Shared trunk, ptau (Huber), thal (multiclass K logits, CrossEntropy).
"""

import torch
import torch.nn as nn


def gene_entropy_loss(w):
    """w: (B, G). Encourages uniform attention over genes."""
    return -(w * torch.log(w + 1e-8)).sum(dim=1).mean()


def make_ordinal_targets(y, n_classes):
    """y: (batch,) integer labels. thresholds=[1,2,...,n_classes-1]; target_k = 1 if y >= k else 0."""
    thresholds = torch.arange(1, n_classes, device=y.device)
    return (y.unsqueeze(1) >= thresholds).float()


def ordinal_logits_to_class(logits):
    """Convert K-1 cumulative logits (P(Y>=k)) to class predictions."""
    probs_above = torch.sigmoid(logits)
    probs_above = torch.cat([
        torch.ones_like(probs_above[:, :1]),
        probs_above,
        torch.zeros_like(probs_above[:, :1])
    ], dim=1)
    probs = probs_above[:, :-1] - probs_above[:, 1:]
    return torch.argmax(probs, dim=1)


class GeneEncoderModel(nn.Module):
    """
    w_g = softmax(MLP(gene_vec_g)); patient_vec = Σ_g w_g · gene_vec_g
    Shared trunk, ptau (Huber), thal (multiclass K logits, CrossEntropy).
    """

    def __init__(self, n_dims, n_thal_classes, dropout=0.2, n_genes=None, thal_ordinal=False):
        super().__init__()
        # n_genes unused, kept for API compatibility
        self.n_dims = n_dims
        self.n_thal_classes = n_thal_classes
        self.thal_ordinal = bool(thal_ordinal)

        # MLP for attention weights: gene_dim → gene_dim/2 → 1
        hidden = max(1, n_dims // 2)
        self.weight_mlp = nn.Sequential(
            nn.Linear(n_dims, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Shared trunk
        trunk_hidden = 16
        self.shared = nn.Sequential(
            nn.Linear(n_dims, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ptau head (regression, ReLU for log1p non-negative)
        self.ptau_head = nn.Linear(trunk_hidden, 1)

        # thal head: ordinal -> K-1 cumulative logits; else K class logits
        thal_out = (n_thal_classes - 1) if self.thal_ordinal else n_thal_classes
        self.thal_head = nn.Linear(trunk_hidden, thal_out)

    def forward(self, x):
        """
        x: (B, G, D) - batch, n_genes, n_dims
        """
        # w_g = softmax(MLP(gene_vec_g)) over genes
        logits = self.weight_mlp(x).squeeze(-1)  # (B, G)
        w = torch.softmax(logits, dim=1)  # (B, G)

        # patient_vec = Σ_g w_g · gene_vec_g
        patient_vec = (w.unsqueeze(-1) * x).sum(dim=1)  # (B, D)

        # Shared trunk
        h = self.shared(patient_vec)

        ptau = self.ptau_head(h).squeeze(-1)
        thal_logits = self.thal_head(h)

        return ptau, thal_logits, w


class GeneEncoderModelMicroglia(nn.Module):
    """
    Same softmax-weighted pooling as GeneEncoderModel.
    Two classification heads: CERAD and Braak (no regression).
    forward returns (cerad_logits, braak_logits, w).
    """

    def __init__(self, n_dims, n_cerad_classes, n_braak_classes, dropout=0.2, n_genes=None):
        super().__init__()
        self.n_dims = n_dims
        self.n_cerad_classes = n_cerad_classes
        self.n_braak_classes = n_braak_classes

        hidden = max(1, n_dims // 2)
        self.weight_mlp = nn.Sequential(
            nn.Linear(n_dims, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        trunk_hidden = 16
        self.shared = nn.Sequential(
            nn.Linear(n_dims, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.cerad_head = nn.Linear(trunk_hidden, n_cerad_classes)
        self.braak_head = nn.Linear(trunk_hidden, n_braak_classes)

    def forward(self, x):
        """
        x: (B, G, D) - batch, n_genes, n_dims
        """
        logits = self.weight_mlp(x).squeeze(-1)  # (B, G)
        w = torch.softmax(logits, dim=1)  # (B, G)
        patient_vec = (w.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        h = self.shared(patient_vec)
        cerad_logits = self.cerad_head(h)
        braak_logits = self.braak_head(h)
        return cerad_logits, braak_logits, w
