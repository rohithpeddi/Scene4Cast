# Running the Monocular 3D Detector (Training & Evaluation)

Commands for training and evaluating the monocular 3D detector described in
[MONOCULAR_3D_DETECTOR.md](MONOCULAR_3D_DETECTOR.md). The detector is trained
**once per backbone**; its checkpoints then feed the WSGG feature extraction
(see [RUN_WSGG.md](RUN_WSGG.md) Step 1).

## 0. Prerequisites

- ActionGenome at `{data_path}` with frames, 2D annotations, and the 3D OBB
  world annotations (`world_3d_annotations_path`, or the default location the
  dataset resolves).
- **DINOv3 only:** the HuggingFace checkpoint is gated — export a token before
  training: `export HF_TOKEN=<your token>`.
- WandB account if `use_wandb: true` (project name set per config).

## 1. Configs

One YAML per backbone × head mode under [configs/detector/](../configs/detector/):

| Config | Backbone | Head mode |
|---|---|---|
| `resnet50_separate_v1.yaml` / `resnet50_unified_v1.yaml` | ResNet50-FPN-V2 | separate / unified |
| `dinov2_saurabh_separate_v1.yaml` (+ `_v1` unified) | DINOv2-Base (`v2`) | separate / unified |
| `dinov2l_saurabh_separate_v1.yaml` (+ unified) | DINOv2-Large (`v2l`) | separate / unified |
| `dinov3l_saurabh_separate_v1.yaml` (+ unified) | DINOv3-ViT-L/16 (`v3l`) | separate / unified |

The production checkpoints used by feature extraction are the **`*_separate`**
variants (`head_3d_mode: separate`, `weight_3d: 1.0`, ramp disabled).

Key fields you will most often touch (full list = `TrainConfig` in
[trainer.py](../lib/detector/monocular3d/trainer.py)):

- `experiment_name`, `working_dir`, `save_path`, `data_path` — paths; edit for
  your machine (the committed configs carry cluster-specific paths).
- `model` (`v2`/`v2l`/`v3l`) + `patch_size` (14 for DINOv2, 16 for DINOv3).
- `head_3d_mode` (`separate`/`unified`), `head_3d_version` (`v1`, or `v2`
  with `depth_maps_dir` pointing at pre-computed depth maps).
- `batch_size`, `epochs` (70), `lr` (1e-4), `gradient_accumulation_steps`.
- Loss weights `weight_cls/box/obj/rpn/3d` and `weight_3d_ramp_epochs`
  (0 = 3D trains from epoch 1).
- `use_accelerator: true` to run under HuggingFace Accelerate.

## 2. Training

```bash
# From the repo root
python -m lib.detector.monocular3d.train \
    --config configs/detector/dinov3l_saurabh_separate_v1.yaml

# Every TrainConfig field is exposed as a CLI override:
python -m lib.detector.monocular3d.train \
    --config configs/detector/dinov3l_saurabh_separate_v1.yaml \
    --data_path /data/rohith/ag --batch_size 32 --lr 5e-5 --use_wandb false

# Resume from a checkpoint directory
python -m lib.detector.monocular3d.train \
    --config configs/detector/dinov3l_saurabh_separate_v1.yaml \
    --ckpt checkpoint_43
```

With `use_accelerator: true`, launch under Accelerate for multi-GPU:

```bash
accelerate launch -m lib.detector.monocular3d.train \
    --config configs/detector/dinov3l_saurabh_separate_v1.yaml
```

Checkpoints land at
`{working_dir}/{experiment_name}/checkpoint_N/checkpoint_state.pth`
(full training state: model + optimizer + scheduler + scaler), one per epoch.
The WSGG feature configs reference specific epochs, e.g.
`v1_dinov3l_separate/checkpoint_43/checkpoint_state.pth`.

Notes:
- The ViT backbone is frozen — only the FPN, RPN, ROI heads, and 3D head
  train, so DINO runs are lighter than the parameter count suggests.
- Watch `loss_3d_raw` early: the targeted head initialization keeps it O(1);
  a blow-up usually means bad intrinsics or corrupt 3D annotations for a
  video.

## 3. Evaluation

```bash
python -m lib.detector.monocular3d.evaluate \
    --checkpoint {working_dir}/{experiment_name}/checkpoint_43 \
    --data_path /data/rohith/ag
```

Reports 2D detection metrics plus 3D box quality (Chamfer / IoU-style
attribute errors) on detections matched to GT at 2D IoU ≥ 0.5
(`iou_match_2d_eval`). Implementation:
[evaluation/evaluate_3d.py](../lib/detector/monocular3d/evaluation/evaluate_3d.py)
(and `evaluate_2d.py` for pure 2D).

## 4. Hand-off to WSGG

After training and picking a checkpoint epoch:

1. Point the feature-extraction configs
   (`configs/features/{predcls,sgdet}/ex_roi_feat_*.yaml`) at the checkpoint.
2. Run the extraction per mode (see [RUN_WSGG.md](RUN_WSGG.md) Step 1).
3. The WSGG configs select the corresponding features via their
   `feature_model` key (`resnet50` / `dinov2b` / `dinov2l` / `dinov3l`).
