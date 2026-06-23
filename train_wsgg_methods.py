"""
WSGG Training Methods (Padded Tensor API)
==========================================

Per-method training classes. Each overrides:
  - init_model()               → create model
  - init_loss_fn()             → create loss module
  - is_temporal()              → sequential or frame-shuffled
  - process_train_video(batch) → forward + loss dict
  - process_test_video(batch)  → inference

The dataset returns a single dict with (T, N_max, ...) and (T, K_max, ...)
pre-padded tensors per video. No per-frame loops needed.

Methods:
  - w_sttran      : W-STTran      (world-adapted STTran, simplest baseline)
  - w_sttran_pp   : W-STTran++    (+ camera, motion, temporal edge attention)
  - w_dsgdetr     : W-DSGDetr     (+ temporal object encoder)
  - w_dsgdetr_pp  : W-DSGDetr++   (+ camera, motion, ego-motion)
  - worldwise     : WorldWise     (MWAE-based full proposed method)

Usage:
  python train_wsgg_methods.py --config configs/methods/predcls/worldwise_predcls_dinov2b.yaml
"""

import logging

import torch

from wsgg_base import load_wsgg_config
from train_wsgg_base import TrainWSGGBase

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    """Move all tensor values in batch dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ============================================================================
# W-STTran (World-adapted STTran — simplest baseline)
# ============================================================================

class TrainWSTTran(TrainWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.baselines.w_sttran.w_sttran import WSTTran

        self._model = WSTTran(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.baselines.w_sttran.loss import WSTTranLoss
        self._loss_fn = WSTTranLoss(
            lambda_vlm=self._conf.lambda_vlm,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],
            corners_seq=b["corners"],
            valid_mask_seq=b["valid_mask"],
            visibility_mask_seq=b["visibility_mask"],
            person_idx_seq=b["person_idx"],
            object_idx_seq=b["object_idx"],
            pair_valid=b["pair_valid"],
            camera_pose_seq=b.get("camera_poses"),
            union_features_seq=b.get("union_features"),
        )

        losses = self._loss_fn(
            predictions=pred,
            gt_attention=b["gt_attention"],
            gt_spatial=b["gt_spatial"],
            gt_contacting=b["gt_contacting"],
            pair_valid=b["pair_valid"],
            visibility_mask=b["visibility_mask"],
            person_idx=b["person_idx"],
            object_idx=b["object_idx"],
            valid_mask=b.get("valid_mask"),
            gt_node_labels=b.get("object_classes"),
        )

        return losses

    def process_test_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],
            corners_seq=b["corners"],
            valid_mask_seq=b["valid_mask"],
            visibility_mask_seq=b["visibility_mask"],
            person_idx_seq=b["person_idx"],
            object_idx_seq=b["object_idx"],
            pair_valid=b["pair_valid"],
            camera_pose_seq=b.get("camera_poses"),
            union_features_seq=b.get("union_features"),
        )

        T = b["visual_features"].shape[0]
        if T > 0:
            return {
                "attention_distribution": pred["attention_distribution"][-1],
                "spatial_distribution": pred["spatial_distribution"][-1],
                "contacting_distribution": pred["contacting_distribution"][-1],
            }
        return None


# ============================================================================
# W-STTran++ (Enhanced: + camera, motion, temporal edge attention)
# ============================================================================

class TrainWSTTranPP(TrainWSTTran):
    """W-STTran++ trainer — same API, only model class differs."""

    def init_model(self):
        from lib.supervised.baselines.w_sttran.w_sttran_pp import WSTTranPP

        self._model = WSTTranPP(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# W-DSGDetr (World-adapted DSGDetr — with temporal object encoder)
# ============================================================================

class TrainWDSGDetr(TrainWSTTran):
    """W-DSGDetr trainer — same batched API as W-STTran, only model differs."""

    def init_model(self):
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr import WDSGDetr

        self._model = WDSGDetr(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.baselines.w_dsgdetr.loss import WDSGDetrLoss
        self._loss_fn = WDSGDetrLoss(
            lambda_vlm=self._conf.lambda_vlm,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )


# ============================================================================
# W-DSGDetr++ (Enhanced: + camera, motion, ego-motion)
# ============================================================================

class TrainWDSGDetrPP(TrainWDSGDetr):
    """W-DSGDetr++ trainer — same API, only model class differs."""

    def init_model(self):
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr_pp import WDSGDetrPP

        self._model = WDSGDetrPP(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# WorldWise (MWAE-based — full proposed method with ablation support)
# ============================================================================

class TrainWorldWise(TrainWSGGBase):
    """
    WorldWise trainer — MWAE-based with config-flag ablation support.

    Differs from the baselines in that the forward pass takes a masking
    probability (p_mask_visible) and the loss consumes a reconstruction
    target (corners), matching the masked world auto-encoder objective.
    """

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldwise.worldwise import WorldWise

        self._model = WorldWise(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.worldwise.loss import WorldWiseLoss
        self._loss_fn = WorldWiseLoss(
            lambda_vlm=self._conf.lambda_vlm,
            lambda_recon=self._conf.lambda_reconstruction,
            lambda_recon_dominance=self._conf.lambda_recon_dominance,
            p_simulate_unseen=self._conf.p_simulate_unseen,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],
            corners_seq=b["corners"],
            valid_mask_seq=b["valid_mask"],
            visibility_mask_seq=b["visibility_mask"],
            person_idx_seq=b["person_idx"],
            object_idx_seq=b["object_idx"],
            pair_valid=b["pair_valid"],
            p_mask_visible=getattr(self._conf, 'p_mask_visible', 0.3),
            camera_pose_seq=b.get("camera_poses"),
        )

        losses = self._loss_fn(
            predictions=pred,
            gt_attention=b["gt_attention"],
            gt_spatial=b["gt_spatial"],
            gt_contacting=b["gt_contacting"],
            pair_valid=b["pair_valid"],
            visibility_mask=b["visibility_mask"],
            person_idx=b["person_idx"],
            object_idx=b["object_idx"],
            valid_mask=b.get("valid_mask"),
            corners=b.get("corners"),
            gt_node_labels=b.get("object_classes"),
        )

        return losses

    def process_test_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],
            corners_seq=b["corners"],
            valid_mask_seq=b["valid_mask"],
            visibility_mask_seq=b["visibility_mask"],
            person_idx_seq=b["person_idx"],
            object_idx_seq=b["object_idx"],
            pair_valid=b["pair_valid"],
            camera_pose_seq=b.get("camera_poses"),
        )

        T = b["visual_features"].shape[0]
        if T > 0:
            return {
                "attention_distribution": pred["attention_distribution"][-1],
                "spatial_distribution": pred["spatial_distribution"][-1],
                "contacting_distribution": pred["contacting_distribution"][-1],
            }
        return None


# ============================================================================
# Entry Point
# ============================================================================

METHOD_MAP = {
    # Baseline adaptations (FasterRCNN / ResNet50 backbone)
    "w_sttran": TrainWSTTran,
    "w_sttran_pp": TrainWSTTranPP,
    "w_dsgdetr": TrainWDSGDetr,
    "w_dsgdetr_pp": TrainWDSGDetrPP,
    # WorldWise (Dino backbones — ablation via config flags)
    "worldwise": TrainWorldWise,
}


def main():
    conf = load_wsgg_config()
    method_name = conf.method_name

    if method_name not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method_name}. Choose from: {list(METHOD_MAP.keys())}")

    trainer_cls = METHOD_MAP[method_name]
    trainer = trainer_cls(conf)
    trainer.init_method_training()


if __name__ == "__main__":
    main()
