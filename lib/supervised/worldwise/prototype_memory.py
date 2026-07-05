"""
Predicate Prototype Memory (Plugin I-6, TEMPURA-style)
=======================================================

Class-conditional EMA prototypes for the long-tailed contacting head.

During training, an EMA prototype is maintained per contacting class from the
relationship tokens of pairs labeled with that class. At prediction time each
relationship token cross-attends over the prototype bank and receives a gated
residual update before the contacting head — pulling tail-class pairs toward
their (rarely seen) class centroid in feature space, which loss-side
reweighting alone cannot do.

Differences from TEMPURA (Nag et al., CVPR 2023), kept deliberately simple for
a first plugin round: single-level prototypes (no compositional memory) and a
gated-residual read instead of the full memory-diffusion unit.

The prototype bank lives in buffers, so it is checkpointed with the model and
frozen naturally at eval (updates only run in training mode with GT provided).
"""

import logging

import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class PredicatePrototypeMemory(nn.Module):
    """
    EMA predicate prototypes + gated cross-attention enhancement.

    Args:
        d_rel: Relationship token dimension.
        num_classes: Number of predicate classes in the bank (contacting: 17).
        momentum: EMA decay for prototype updates.
        n_heads: Attention heads for the prototype read.
        dropout: Dropout on the attention read.
    """

    def __init__(
        self,
        d_rel: int = 256,
        num_classes: int = 17,
        momentum: float = 0.99,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_rel = d_rel
        self.num_classes = num_classes
        self.momentum = momentum

        # Prototype bank — buffers: checkpointed, never gradient-trained
        self.register_buffer("prototypes", torch.zeros(num_classes, d_rel))
        self.register_buffer(
            "proto_initialized", torch.zeros(num_classes, dtype=torch.bool)
        )

        self.read_attn = nn.MultiheadAttention(
            embed_dim=d_rel, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        # Gated residual: how much prototype evidence each token accepts
        self.gate = nn.Sequential(
            nn.Linear(d_rel * 2, d_rel),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_rel)

    @torch.no_grad()
    def update(
        self,
        rel_tokens: torch.Tensor,
        gt_multihot: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> None:
        """
        EMA-update prototypes from labeled relationship tokens.

        Args:
            rel_tokens: (T, K, d_rel)
            gt_multihot: (T, K, num_classes) float multi-hot labels.
            pair_valid: (T, K) bool.
        """
        valid = pair_valid.bool()
        if not valid.any():
            return

        tokens = rel_tokens[valid].detach().float()      # (M, D)
        labels = gt_multihot[valid].float()               # (M, C)

        counts = labels.sum(dim=0)                        # (C,)
        seen = counts > 0
        if not seen.any():
            return

        class_means = (labels.t() @ tokens) / counts.clamp(min=1).unsqueeze(-1)  # (C, D)

        m = self.momentum
        proto = self.prototypes
        init = self.proto_initialized

        # First observation seeds the prototype; afterwards EMA
        fresh = seen & ~init
        ema = seen & init
        proto[fresh] = class_means[fresh]
        proto[ema] = m * proto[ema] + (1.0 - m) * class_means[ema]
        init |= seen

    def forward(
        self,
        rel_tokens: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Enhance relationship tokens by reading from the prototype bank.

        Args:
            rel_tokens: (T, K, d_rel)
            pair_valid: (T, K) bool.

        Returns:
            enhanced: (T, K, d_rel)
        """
        T, K, D = rel_tokens.shape
        if K == 0 or not self.proto_initialized.any():
            # Nothing to read from yet (first steps of training)
            return rel_tokens

        # K/V = initialized prototypes only
        bank = self.prototypes[self.proto_initialized]          # (C_init, D)
        bank = bank.unsqueeze(0).expand(T, -1, -1).to(rel_tokens.dtype)

        read, _ = self.read_attn(
            query=rel_tokens, key=bank, value=bank,
        )  # (T, K, D)

        g = self.gate(torch.cat([rel_tokens, read], dim=-1))    # (T, K, D)
        enhanced = self.norm(rel_tokens + g * read)

        return enhanced * pair_valid.unsqueeze(-1).to(enhanced.dtype)
