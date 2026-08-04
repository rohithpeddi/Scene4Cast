"""
W-USG: World-adapted USG-Par (Batched)
========================================

External lightweight baseline — the relation machinery of USG-Par
(Universal Scene Graph Generation, Wu et al., CVPR 2025) transplanted onto
the WSGG substrate. Sits BESIDE the nested method ladder, not inside it:
it is neither a superset nor a subset of any ladder tier.

What is kept from the shared substrate (identical to W-STTran):
  - vectorized_lks_buffer: zero-order-hold memory so the model can emit
    predictions for unobserved slots (USG-Par is pixels-only and has no
    such mechanism; without it the method cannot score occpair at all)
  - GlobalStructuralEncoder over world-frame OBB corners
  - LKSTokenizer fusion (d_camera=0 — no camera encoder)
  - NodePredictor and the pair-token forming of RelationshipPredictor
    (person ⊕ object ⊕ union-ROI ⊕ CLIP text — USG-Par likewise fuses
    visual queries with text embeddings)

What replaces the ladder's relation stack (SpatialGNN 3D-PE encoder +
TemporalEdgeAttention), following USG-Par's design:
  - ObjectContextEncoder: plain per-frame transformer over object tokens
    with NO 3D positional encoding — the analog of USG-Par's shared mask
    decoder refining object queries (its 2D pipeline has no 3D PE)
  - USGRelationDecoder: a transformer *decoder* in which pair queries
    self-attend and cross-attend to the frame's object-token memory — the
    analog of USG-Par's relation proposal constructor + relation decoder
  - Text-centric alignment logits (cosine similarity between projected
    object tokens and the frozen CLIP class embeddings) consumed by
    WUSGLoss's contrastive term — the analog of USG-Par's text-centric
    scene contrastive learning
  - NO per-pair temporal attention: USG-Par predicts video relations
    per-frame after object association; slot identity is given in WSGG,
    so its object associator collapses to the identity map

Single-pass pipeline:
  1. vectorized_lks_buffer(visual, vis, valid)      → (T, N, d_roi), (T, N)
  2. GlobalStructuralEncoder(corners)                → (T, N, d_struct)
  3. LKSTokenizer(struct, buffer, staleness)         → (T, N, d_model)
  4. ObjectContextEncoder(tokens, valid)             → (T, N, d_model)
  5. NodePredictor(enriched)                         → (T, N, C)
  6. batched_form_and_attend(enriched, logits,...)   → (T, K_max, d_rel)
  7. USGRelationDecoder(rel, enriched, masks)        → (T, K_max, d_rel)
  8. batched_predict(rel_tokens, valid)              → distributions
  9. align_proj(enriched) · clip_embeddings / temp   → (T, N, C) align logits

All differentiable except the LKS buffer (step 1).
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from lib.supervised.baselines.lks_buffer.lks_memory import vectorized_lks_buffer
from lib.supervised.baselines.lks_buffer.lks_tokenizer import LKSTokenizer

from lib.supervised.components import (
    GlobalStructuralEncoder, NodePredictor, RelationshipPredictor,
)


class ObjectContextEncoder(nn.Module):
    """Per-frame transformer over object tokens WITHOUT 3D positional encoding.

    The analog of USG-Par's shared mask decoder query refinement: object
    queries exchange context through plain self-attention. Deliberately
    weaker than the ladder's SpatialGNN (which injects a 3D spatial PE) —
    USG-Par's 2D pipeline has no 3D-aware inter-object encoding.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 d_feedforward: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
        )

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (T, N, d_model)
            valid_mask: (T, N) bool — True for real objects.

        Returns:
            enriched: (T, N, d_model), padding zeroed.
        """
        padding_mask = ~valid_mask  # True = ignore
        # Failsafe: an all-padded frame makes softmax NaN — unmask slot 0
        all_invalid = padding_mask.all(dim=1)
        if all_invalid.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        enriched = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return enriched * valid_mask.unsqueeze(-1).to(enriched.dtype)


class USGRelationDecoder(nn.Module):
    """Transformer decoder: pair queries attend to the frame's object memory.

    The analog of USG-Par's relation proposal constructor + relation decoder:
    each pair query is iteratively refined by (a) self-attention across the
    frame's pair set and (b) cross-attention into the object tokens.
    """

    def __init__(self, d_rel: int, d_model: int, n_layers: int, n_heads: int,
                 d_feedforward: int, dropout: float):
        super().__init__()
        self.memory_proj = (
            nn.Identity() if d_model == d_rel else nn.Linear(d_model, d_rel)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=d_rel, nhead=n_heads, dim_feedforward=d_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_rel),
        )

    def forward(
        self,
        rel_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        pair_valid: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rel_tokens: (T, K, d_rel) — pair queries.
            object_tokens: (T, N, d_model) — enriched object memory.
            pair_valid: (T, K) bool.
            valid_mask: (T, N) bool.

        Returns:
            refined: (T, K, d_rel), padding zeroed.
        """
        tgt_padding = ~pair_valid
        mem_padding = ~valid_mask
        # Failsafes for fully padded frames (NaN-free attention)
        all_tgt_invalid = tgt_padding.all(dim=1)
        if all_tgt_invalid.any():
            tgt_padding = tgt_padding.clone()
            tgt_padding[all_tgt_invalid, 0] = False
        all_mem_invalid = mem_padding.all(dim=1)
        if all_mem_invalid.any():
            mem_padding = mem_padding.clone()
            mem_padding[all_mem_invalid, 0] = False

        memory = self.memory_proj(object_tokens)
        refined = self.decoder(
            rel_tokens, memory,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=mem_padding,
        )
        return refined * pair_valid.unsqueeze(-1).to(refined.dtype)


class WUSG(nn.Module):
    """
    W-USG — USG-Par-style relation decoding on the WSGG substrate.

    Args:
        config: Method config namespace. W-USG-specific keys (all optional):
            n_usg_decoder_layers: relation-decoder depth (default 2)
            align_temperature: text-alignment softmax temperature (default 0.07)
        num_object_classes: Object categories.
        attention_class_num: Attention relationship classes.
        spatial_class_num: Spatial relationship classes.
        contact_class_num: Contacting relationship classes.
    """

    def __init__(
        self,
        config,
        num_object_classes: int = 37,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
    ):
        super().__init__()
        self.config = config
        self.num_object_classes = num_object_classes

        # Module 1: Global Structural Encoder — 3D OBB geometry in world frame
        self.global_structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: LKS Tokenizer (geometry + raw buffer fusion, NO camera)
        self.tokenizer = LKSTokenizer(
            d_struct=config.d_struct,
            d_detector_roi=config.d_detector_roi,
            d_model=config.d_model,
            d_camera=0,
        )

        # Module 3: Object context encoder (no 3D PE — see class docstring)
        self.object_context_encoder = ObjectContextEncoder(
            d_model=config.d_model,
            n_layers=config.n_gnn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 4: Node predictor
        self.node_predictor = NodePredictor(
            d_memory=config.d_model,
            num_classes=num_object_classes,
        )

        # Module 5: Relationship predictor (pair-token forming + heads reused;
        # its internal self-attention acts as the decoder's pre-refinement)
        clip_path = getattr(config, 'clip_embeddings_path', '')
        clip_path = Path(config.data_path) / clip_path if clip_path else None
        self.rel_predictor = RelationshipPredictor(
            d_model=config.d_model,
            d_text=config.d_text,
            d_rel=config.d_rel,
            d_union_roi=config.d_union_roi,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
            clip_embeddings_path=clip_path,
            n_rel_layers=config.n_rel_layers,
            n_rel_heads=config.n_rel_heads,
            dropout=config.dropout,
        )

        # Module 6: USG relation decoder (replaces TemporalEdgeAttention)
        self.relation_decoder = USGRelationDecoder(
            d_rel=config.d_rel,
            d_model=config.d_model,
            n_layers=getattr(config, 'n_usg_decoder_layers', 2),
            n_heads=config.n_rel_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 7: Text-centric alignment head — projects object tokens into
        # the frozen CLIP embedding space (USG-Par's text-centric contrast)
        d_clip = self.rel_predictor.clip_embeddings.shape[1]
        self.align_proj = nn.Linear(config.d_model, d_clip)
        self.align_temperature = float(getattr(config, 'align_temperature', 0.07))

    def _text_alignment_logits(
        self, enriched_all: torch.Tensor, valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity of projected object tokens vs CLIP class embeddings.

        Returns:
            align_logits: (T, N, C) — scaled cosine similarities, padding zeroed.
        """
        clip_emb = self.rel_predictor.clip_embeddings[: self.num_object_classes]
        proj = F.normalize(self.align_proj(enriched_all), dim=-1)        # (T, N, d_clip)
        text = F.normalize(clip_emb.to(proj.dtype), dim=-1)              # (C, d_clip)
        logits = proj @ text.t() / self.align_temperature                # (T, N, C)
        return logits * valid_mask.unsqueeze(-1).to(logits.dtype)

    def forward(
        self,
        visual_features_seq: torch.Tensor,
        corners_seq: torch.Tensor,
        valid_mask_seq: torch.Tensor,
        visibility_mask_seq: torch.Tensor,
        person_idx_seq: torch.Tensor,
        object_idx_seq: torch.Tensor,
        pair_valid: torch.Tensor,
        camera_pose_seq: Optional[torch.Tensor] = None,
        union_features_seq: Optional[torch.Tensor] = None,
        node_labels_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a full video in a single batched forward pass.

        Args:
            visual_features_seq: (T, N_max, d_roi)
            corners_seq:         (T, N_max, 8, 3)
            valid_mask_seq:      (T, N_max) bool
            visibility_mask_seq: (T, N_max) bool
            person_idx_seq:      (T, K_max) long
            object_idx_seq:      (T, K_max) long
            pair_valid:          (T, K_max) bool
            camera_pose_seq:     (T, 4, 4) or None — accepted but unused
            union_features_seq:  (T, K_max, d_union_roi) or None
            node_labels_seq:     (T, N_max) long or None (predcls GT labels)

        Returns:
            dict with (T, ...) padded tensors, incl. `align_logits` (T, N, C)
            for the text-centric contrastive loss.
        """
        # ==================== Step 1: LKS buffer (raw features) ====================
        buffer_all, staleness_all = vectorized_lks_buffer(
            raw_visual=visual_features_seq,
            visibility_mask=visibility_mask_seq,
            valid_mask=valid_mask_seq,
        )  # (T, N, d_detector_roi), (T, N)

        # ==================== Step 2: Global structural encoding ====================
        struct_all, _ = self.global_structural_encoder(
            corners_seq, valid_mask_seq,
        )  # (T, N, d_struct)

        # ==================== Step 3: Tokenizer (no camera features) ====================
        tokens_all = self.tokenizer(
            geometry_tokens=struct_all,
            buffer_features=buffer_all,
            valid_mask=valid_mask_seq,
            cam_feats=None,
            staleness=staleness_all,
        )  # (T, N, d_model)

        # ==================== Step 4: Object context encoder ====================
        enriched_all = self.object_context_encoder(
            tokens_all, valid_mask_seq,
        )  # (T, N, d_model)

        # ==================== Step 5: Node prediction ====================
        node_logits_all = self.node_predictor(enriched_all)  # (T, N, num_classes)

        # ==================== Step 6: Pair-query forming ====================
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq,
            pair_valid, union_features_seq,
            node_class_override=node_labels_seq,
        )  # (T, K_max, d_rel), (T, K_max)

        # ==================== Step 7: USG relation decoder ====================
        refined_rel = self.relation_decoder(
            rel_tokens, enriched_all, pair_valid_out, valid_mask_seq,
        )

        # ==================== Step 8: Predict distributions ====================
        edge_out = self.rel_predictor.batched_predict(refined_rel, pair_valid_out)

        # ==================== Step 9: Text-centric alignment logits ====================
        align_logits = self._text_alignment_logits(enriched_all, valid_mask_seq)

        return {
            "node_logits": node_logits_all,
            "attention_logits": edge_out["attention_logits"],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
            "spatial_logits": edge_out["spatial_logits"],
            "contacting_logits": edge_out["contacting_logits"],
            "align_logits": align_logits,
        }
