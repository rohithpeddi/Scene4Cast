"""
Associative Retriever (Batched, Bidirectional)
================================================

Per-object cross-attention that auto-completes masked tokens by
retrieving visual features from ALL visible appearances of that
object across ALL T frames.

For each object n:
  Q = masked tokens across T (frames where unseen)
  K/V = visible tokens across T (frames where visible)

This replaces the sequential episodic memory store/retrieve pattern.
All T frames are available simultaneously — no causal constraint.

The retriever also supports view-aware biasing: camera pose features
are added to Q (current viewpoint) and K (capture viewpoint) so the
attention naturally favours memory entries from similar viewpoints.
"""

import logging

import torch
import torch.nn as nn
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class AssociativeRetriever(nn.Module):
    """
    Bidirectional per-object cross-attention over all T frames.

    For each object, masked tokens query visible tokens from ANY frame.
    Visible tokens are also refined (self-contextualized) via the same
    cross-attention mechanism.

    Supports view-aware retrieval via camera pose feature bias on Q/K.

    Args:
        d_model: Token dimension.
        n_layers: Number of cross-attention layers.
        n_heads: Attention heads.
        d_feedforward: FFN hidden dim.
        dropout: Dropout probability.
        d_camera: Camera feature dim for view-aware Q/K bias.
        cross_object: Plugin I-7 — masked slots may also retrieve from OTHER
            objects' visible tokens (scene context), not just their own
            appearances. Slot-identity embeddings are added to Q/K so
            same-object retrieval stays preferred, and padding slots are
            pruned before attention so cost is bounded by the real object
            count, not N_max.
        max_slots: Size of the slot-identity embedding table (cross_object).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        d_feedforward: int = 512,
        dropout: float = 0.1,
        d_camera: int = 128,
        cross_object: bool = False,
        max_slots: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_camera = d_camera
        self.cross_object = cross_object
        self.max_slots = max_slots

        if self.cross_object:
            self.slot_id_emb = nn.Embedding(max_slots, d_model)

        # Cross-attention: Q = all tokens, K/V = visible tokens only
        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms1 = nn.ModuleList()
        self.cross_norms2 = nn.ModuleList()
        self.cross_ffns = nn.ModuleList()

        for _ in range(n_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=n_heads,
                    dropout=dropout, batch_first=True,
                )
            )
            self.cross_norms1.append(nn.LayerNorm(d_model))
            self.cross_norms2.append(nn.LayerNorm(d_model))
            self.cross_ffns.append(nn.Sequential(
                nn.Linear(d_model, d_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_feedforward, d_model),
                nn.Dropout(dropout),
            ))

        # View-aware pose projections
        self.query_pose_proj = nn.Linear(d_camera, d_model)
        self.key_pose_proj = nn.Linear(d_camera, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        visibility_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        cam_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Bidirectional per-object cross-attention over all T frames.

        Each object's masked tokens retrieve from that same object's
        visible tokens across all frames. Visible tokens are also
        refined by attending to other visible appearances.

        Args:
            tokens:          (T, N, d_model) — scaffold tokens (visible + masked).
            visibility_mask: (T, N) bool — True where object is visible.
            valid_mask:      (T, N) bool — True for real objects.
            cam_feats:       (T, N, d_camera) or None — camera-relative features.

        Returns:
            completed: (T, N, d_model) — auto-completed tokens.
        """
        T, N, D = tokens.shape
        device = tokens.device

        # --- Reshape: (T, N, D) → (N, T, D) — per-object temporal sequences ---
        x = tokens.permute(1, 0, 2)                  # (N, T, D)
        vis = visibility_mask.permute(1, 0).contiguous()  # (N, T)
        val = valid_mask.permute(1, 0).contiguous()       # (N, T)

        # --- View-aware bias ---
        if cam_feats is not None:
            cam = cam_feats.permute(1, 0, 2)          # (N, T, d_camera)
            q_bias = self.query_pose_proj(cam)        # (N, T, D)
            k_bias = self.key_pose_proj(cam)          # (N, T, D)
        else:
            q_bias = torch.zeros_like(x)
            k_bias = torch.zeros_like(x)

        # --- Cross-object path (plugin I-7) ---
        if self.cross_object:
            return self._forward_cross_object(x, vis, val, q_bias, k_bias)

        # --- Build K/V from visible tokens only ---
        # Key padding mask for cross-attention: True = ignore
        # For each object, ignore frames where it's NOT visible (or invalid)
        kv_active = vis & val  # (N, T) — True where visible + valid
        kv_padding_mask = ~kv_active  # (N, T) — True = ignore

        # Failsafe: if an object has NO visible frames, unmask first frame
        all_masked = kv_padding_mask.all(dim=1)  # (N,)
        if all_masked.any():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[all_masked, 0] = False

        # --- Cross-attention layers ---
        query = x + q_bias  # Add view bias to queries
        key = x + k_bias    # Add view bias to keys
        value = x            # Clean semantic content — NO bias

        for i in range(len(self.cross_attn_layers)):
            # Cross-attention: Q = all tokens, K = visible with view bias, V = visible (clean)
            attn_out, _ = self.cross_attn_layers[i](
                query=query, key=key, value=value,
                key_padding_mask=kv_padding_mask,
            )
            query = self.cross_norms1[i](query + attn_out)
            ffn_out = self.cross_ffns[i](query)
            query = self.cross_norms2[i](query + ffn_out)

            # Iteratively refine K/V: visible positions get the evolved query
            # so subsequent layers attend to progressively richer representations
            vis_mask = kv_active.unsqueeze(-1).float()  # (N, T, 1)
            key = torch.where(kv_active.unsqueeze(-1), query + k_bias, key)
            value = torch.where(kv_active.unsqueeze(-1), query, value)

        # --- Zero out invalid ---
        query = query * val.unsqueeze(-1).float()

        # --- Reshape back: (N, T, D) → (T, N, D) ---
        completed = query.permute(1, 0, 2)  # (T, N, D)

        return completed

    def _forward_cross_object(
        self,
        x: torch.Tensor,
        vis: torch.Tensor,
        val: torch.Tensor,
        q_bias: torch.Tensor,
        k_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Plugin I-7: one attention scope over ALL objects' tokens.

        Every token queries the visible tokens of EVERY object across all
        frames. Slot-identity embeddings on Q and K let attention prefer the
        same object while still allowing scene-context retrieval. Padding
        slots (never valid in any frame) are pruned before attention, so the
        sequence length is bounded by the real object count × T.

        Args:
            x, q_bias, k_bias: (N, T, D) — per-object sequences + view biases.
            vis, val: (N, T) bool.

        Returns:
            completed: (T, N, D)
        """
        N, T, D = x.shape
        device = x.device

        # --- Prune slots that are never valid (pure padding) ---
        active = val.any(dim=1)                       # (N,)
        if not active.any():
            return (x * val.unsqueeze(-1).float()).permute(1, 0, 2)
        active_idx = active.nonzero(as_tuple=True)[0]  # (n_a,)
        n_a = active_idx.shape[0]

        x_a = x[active_idx]            # (n_a, T, D)
        qb_a = q_bias[active_idx]
        kb_a = k_bias[active_idx]
        vis_a = vis[active_idx]        # (n_a, T)
        val_a = val[active_idx]

        # --- Slot identity embedding (same for Q and K of one object) ---
        slot_ids = active_idx.clamp(max=self.max_slots - 1)         # (n_a,)
        slot_emb = self.slot_id_emb(slot_ids).unsqueeze(1)           # (n_a, 1, D)

        # --- Flatten to a single sequence: (1, n_a*T, D) ---
        flat = lambda t: t.reshape(1, n_a * T, -1)
        query = flat(x_a + qb_a + slot_emb)
        key = flat(x_a + kb_a + slot_emb)
        value = flat(x_a)                                            # clean content

        kv_active = (vis_a & val_a).reshape(1, n_a * T)              # (1, n_a*T)
        kv_padding_mask = ~kv_active
        if kv_padding_mask.all():
            kv_padding_mask = kv_padding_mask.clone()
            kv_padding_mask[0, 0] = False

        for i in range(len(self.cross_attn_layers)):
            attn_out, _ = self.cross_attn_layers[i](
                query=query, key=key, value=value,
                key_padding_mask=kv_padding_mask,
            )
            query = self.cross_norms1[i](query + attn_out)
            ffn_out = self.cross_ffns[i](query)
            query = self.cross_norms2[i](query + ffn_out)

            # Iteratively refine K/V at visible positions (as in the
            # per-object path) so later layers see evolved representations.
            kv_mask = kv_active.unsqueeze(-1)
            key = torch.where(kv_mask, query + flat(kb_a + slot_emb.expand(n_a, T, D)), key)
            value = torch.where(kv_mask, query, value)

        # --- Scatter back to full slot layout ---
        out_a = query.reshape(n_a, T, D)
        completed = torch.zeros_like(x)
        completed[active_idx] = out_a
        completed = completed * val.unsqueeze(-1).float()

        return completed.permute(1, 0, 2)  # (T, N, D)
