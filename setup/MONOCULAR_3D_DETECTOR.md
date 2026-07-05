# Monocular 3D Detector Architecture

The detector upstream of everything in this repo: given a single RGB frame, it
produces 2D detections **and** 3D oriented bounding boxes (8 corners in camera
space). Its trained checkpoints power the ROI-feature extraction that feeds the
WSGG methods (see [RUN_WSGG.md](RUN_WSGG.md) Step 1), and its predicted 3D
boxes become the `corners` input in sgdet mode.

Code: [lib/detector/monocular3d/](../lib/detector/monocular3d/) ·
Model: [models/dino_mono_3d.py](../lib/detector/monocular3d/models/dino_mono_3d.py) ·
Loss: [losses/ovmono3d_loss.py](../lib/detector/monocular3d/losses/ovmono3d_loss.py)

## Overall pipeline

```
image ─► DINO ViT backbone (frozen) ─► SimpleFeaturePyramid (p2..p6)
      ─► Faster R-CNN RPN ─► ROI heads (2D cls + box)
                                 │  shared 1024-d FC features
                                 ▼
                          3D prediction head
             dims(3) · yaw(sin,cos) · depth(1) · center-offset(2) · μ(1)
                                 ▼
              pinhole back-projection → 8 corners (camera space)
```

## Components

### Backbone (`Dinov3ModelBackbone` + `SimpleFeaturePyramid`)

- HuggingFace ViT selected via `model` key from `MODEL_REGISTRY`:
  `v2` = DINOv2-Base (768-d), `v2l` = DINOv2-Large (1024-d),
  `v3l` = DINOv3-ViT-L/16 (1024-d, gated — needs `HF_TOKEN`).
- The ViT is **frozen** (only adapter params train); patch tokens are reshaped
  into a 2D feature map and fed to a ViTDet-style `SimpleFeaturePyramid`
  (scale factors 4/2/1/0.5 + max-pool top level → p2..p6, 256 channels, BN).
- Images are **not** resized/normalized by the RCNN transform
  (`_NoOpRCNNTransform` batches only, padding to patch-size divisibility);
  the dataset performs Pi-3-compatible aspect-preserving resize
  (`pixel_limit: 255000`), so predicted boxes live in Pi-3 image space —
  consistent with the world-annotation pipeline.
- A ResNet50-FPN variant (`ResNetMonocular3D` in
  [models/resnet_mono_3d.py](../lib/detector/monocular3d/models/resnet_mono_3d.py),
  FasterRCNN-ResNet50-FPN-V2) provides the baseline-backbone counterpart.

### 2D detection

Standard torchvision Faster R-CNN: `AnchorGenerator` with one anchor size per
FPN level (32…512, 3 aspect ratios), `MultiScaleRoIAlign` (7×7), two-FC box
head (1024-d shared features), class + box regression predictors. AG classes
(~37, auto-detected from the dataset when `num_classes: null`).

### 3D head (`_Mono3DPredictionLayers`)

A lightweight branch over the **shared 1024-d ROI features**, concatenated
with the normalized 2D box (4) and camera intrinsics fx,fy,cx,cy (4)
(both divided by `input_reference_size` to stay O(1)):

- `dims` (l,w,h) via softplus, `yaw` as normalized (sin, cos), `depth` via
  softplus, `center_offset` (Δu, Δv from the 2D box center), and an
  **uncertainty scalar μ** per box.
- Targeted initialization keeps the initial 3D loss sane (moderate depth ≈
  1.3 m, ~0.7 m dims, identity rotation, μ = 0).
- `_compute_3d_corners` reconstructs the 8 corners: yaw-rotated dimension
  offsets around a center obtained by back-projecting (u+Δu, v+Δv, depth)
  through the pinhole intrinsics. Division order is float16-safe.

Two integration modes (`head_3d_mode`):
- **`unified`** — `Mono3DRoIHeads` wraps the ROI heads; the 3D branch shares
  the box-head FC features directly inside the ROI forward.
- **`separate`** — `SeparateMono3DHead` hooks the box head's output features
  and wraps `select_training_samples` to capture matched proposals; the 3D
  loss is computed outside the ROI heads. (The production checkpoints —
  `*_separate` — use this mode.)

Two head versions (`head_3d_version`): `v1` = as above; `v2` adds per-ROI
depth statistics (median/mean/std/min/max) from pre-computed depth maps
(`depth_maps_dir`) as extra input.

During training, positive proposals for the 3D loss are capped at
`max_3d_proposals` (default 64) to bound memory.

### Loss — OVMono3D-style uncertainty-weighted disentangled Chamfer

Per matched box *i*:  `L_i = √2 · exp(−μ_i) · L3D_i + μ_i` — the network
learns to down-weight boxes it cannot resolve (μ grows) while μ itself is
penalized.

`L3D_i` = sum of **geometry-level disentangled** terms + a holistic term: for
each attribute group (xy-center, z/depth, dims, yaw), reconstruct a box from
the *predicted* value of that attribute and *GT* values of all others, and
apply a smooth-L1 Chamfer distance over the 8 corners; plus
Chamfer(pred corners, GT corners). Attributes are recovered from corners via
xy-PCA (rotation) and projected extents (dims), robust to corner ordering;
degenerate boxes are filtered.

Total training loss = weighted 2D terms (`weight_cls/box/obj/rpn`) +
`weight_3d` × 3D term, with an optional staged ramp
(`weight_3d_ramp_epochs`: 2D-only first, then ramp 3D in — the V1 configs
disable the ramp and train 3D from epoch 1).

## Training infrastructure

`DinoAGTrainer3D` ([trainer.py](../lib/detector/monocular3d/trainer.py)):
dataclass `TrainConfig` populated from YAML + CLI overrides
([train.py](../lib/detector/monocular3d/train.py)); AdamW + warmup/cosine;
optional HuggingFace Accelerate (`use_accelerator`); gradient accumulation and
clipping; per-epoch full-state checkpoints
(`{working_dir}/{experiment_name}/checkpoint_N/checkpoint_state.pth`);
WandB logging; per-epoch qualitative plots. The dataset
([datasets/ag_dataset_3d.py](../lib/detector/monocular3d/datasets/ag_dataset_3d.py))
pairs AG frames with 3D OBB annotations from the world-annotation pipeline and
carries per-frame camera intrinsics.

## Evaluation

`python -m lib.detector.monocular3d.evaluate` →
[evaluation/evaluate_3d.py](../lib/detector/monocular3d/evaluation/evaluate_3d.py):
2D detection quality plus 3D box quality on detections matched to GT at 2D IoU
≥ `iou_match_2d_eval` (0.5). See [RUN_MON3D.md](RUN_MON3D.md) for commands.

## Role in the WSGG pipeline

1. Train the detector per backbone (`resnet50`, `v2`, `v2l`, `v3l`).
2. Feature extraction (`datasets/preprocess/features/extract_roi_features_*.py`)
   loads a detector checkpoint and dumps per-video PKLs: ROI features (1024-d),
   2D boxes, predicted 3D corners, labels, pair indices, union features.
3. The WSGG methods consume those PKLs — predcls uses GT boxes (detector
   supplies features), sgdet uses the detector's own detections and 3D boxes.
