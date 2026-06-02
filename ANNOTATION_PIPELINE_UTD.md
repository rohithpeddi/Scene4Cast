# Unified Test Dataset (UTD) Annotation Pipeline
## Complete Guide to Generating Corrected Annotations

---

## Overview

This document is the **runnable** guide for generating all annotations for the
**test dataset** using **manually corrected floor transforms** stored in Firebase.

**Key Assumption:** You have already manually corrected the floor transforms and
stored them in Firebase under:
- `worldframe_obb/floor_corrections/<video_key>/latest`
- `worldframe_obb/xy_alignments/<video_key>/latest`
- `worldframe_obb/final_alignments/<video_key>/latest`

**Conventions used below**
- All commands are run **from the repository root**
  (`c:\Users\rohit\PycharmProjects\WorldSGG`), using full script paths.
- All paths, `phase: test`, and per-script settings come from
  [`configs/annotation_utd.yaml`](configs/annotation_utd.yaml). The flags shown
  are therefore mostly optional (they mirror the config) — they are included for
  clarity and so you can override per-run.
- Every command includes **`--overwrite`** so existing outputs are regenerated
  (the request is a full rebuild). Two scripts have no `--overwrite` flag and are
  noted explicitly (`gt_generator.py` always rewrites; `download_*` re-fetches).

> ⚠️ **Phase consistency is critical.** The corrected pipeline reads/writes under
> `world_annotations/test/...`. The raw prerequisite generator must therefore run
> with `--phase test` so its output lands in `world_annotations/test/bbox_annotations_3d_obb`
> (not the flat `world_annotations/bbox_annotations_3d_obb`). The config's
> `phase: test` already enforces this; the commands also pass it explicitly.

---

## Transformation Order Overview

The corrected annotation pipeline applies these transforms in sequence:

```
x_canonical = T_XY ∘ T_delta ∘ T_auto ∘ x_world
```

1. **T_auto** — automatic floor alignment from SMPL-scene correspondences,
   baked into the raw `bbox_annotations_3d_obb/*.pkl` as `global_floor_sim`.
   *(This is the value the manual correction replaces / builds on.)*
2. **T_delta** (from `floor_correction`) — manual delta: Euler angles + scale + translation.
3. **T_XY** (from `xy_alignment`) — aligns corrected floor normal → +Z, local X → world X.
4. **Combined transform** (from `final_alignment.combined_transform`) — a precomputed
   4×4 that, if present, replaces manual composition of steps 1–3.

---

## What gets run, and what does NOT

| Layer | Scripts | Run for corrected test build? |
|---|---|---|
| **Raw prerequisites** | `raw/gt_generator.py`, `raw/bb3D_generator_gt_obb.py` | **YES** — only if `world_annotations/test/bbox_annotations_3d_obb` doesn't already exist. Floor-independent; reused as-is. |
| **Corrected pipeline** | `download_floor_manual_corrections.py`, `corrected_world_bbox_generator.py`, `corrected_frame_bbox_generator.py`, `corrected_4d_bbox_generator.py` | **YES** |
| **Relationships + scene graph** | `augment_relationships_test.py`, `combine_world4d_relationships_test.py`, `world_scene_graph_generator.py` | **YES** |
| **Raw downstream (stale)** | `raw/bb3D_bridge_generator_obb.py`, `raw/frame_bbox_3D_gt_obb.py`, `raw/frame_to_world4D_annotations.py` | **NO** — superseded by the corrected scripts; their outputs use the old auto floor and are not consumed by the corrected scene-graph path. |

---

## Pipeline Execution Order (TEST Dataset)

### Stage 0 — Raw prerequisites *(run once; skip if already present)*

These produce the floor-independent world OBBs the corrected pipeline forks from.
Skip this stage if `world_annotations/test/bbox_annotations_3d_obb/` is already populated.

**0a. Ground-truth COCO annotations** (`raw/gt_generator.py`)
*No `--overwrite` flag — it always (re)writes the JSONs.*
```bash
python datasets/preprocess/annotations/raw/gt_generator.py
```
Output: `/data/rohith/ag/ag4D/gt_annotations/<video_id>/...json`

**0b. Raw world 3D OBBs + auto floor** (`raw/bb3D_generator_gt_obb.py`)
```bash
python datasets/preprocess/annotations/raw/bb3D_generator_gt_obb.py \
  --phase test \
  --overwrite
```
Output: `/data/rohith/ag/world_annotations/test/bbox_annotations_3d_obb/<video_id>.pkl`
(contains `global_floor_sim`, floor mesh `gv/gf/gc`, per-frame world OBBs).

> The entry point now runs the full batch for the requested `--phase` (previously it
> ran a single sample). Use `--split <CODE>` to limit to one shard, or
> `--video <id>.mp4` for a single video.

---

### Stage 1 — Download corrected floor transforms (`download_floor_manual_corrections.py`)
*No `--overwrite` flag here uses `--overwrite` to re-download; otherwise it skips existing.*
```bash
python datasets/preprocess/annotations/download_floor_manual_corrections.py \
  --overwrite
```
- Input: Firebase
- Output: `/data/rohith/ag/world_annotations/manual_corrections/<video_key>.pkl`

Check availability first (optional):
```bash
python datasets/preprocess/annotations/download_floor_manual_corrections.py --list
```

---

### Stage 2 — Corrected world-frame 3D BBoxes (`corrected_world_bbox_generator.py`)
```bash
python datasets/preprocess/annotations/corrected_world_bbox_generator.py \
  --phase test \
  --corrections-only \
  --overwrite
```
- Inputs: `manual_corrections/<key>.pkl`, raw `test/bbox_annotations_3d_obb/<id>.pkl`
  (for `global_floor_sim` fallback + floor mesh), Pi3 `predictions.npz`.
- Output: `/data/rohith/ag/world_annotations/test/bbox_annotations_3d_obb_corrected/<id>.pkl`

---

### Stage 3 — Corrected frame-level 3D BBoxes (`corrected_frame_bbox_generator.py`)
Produces FINAL (canonical) **and** CAMERA-frame views in one run via `--mode both`.
*(Side branch — used for training/visualization; not required by the scene graph.)*
```bash
python datasets/preprocess/annotations/corrected_frame_bbox_generator.py \
  --phase test \
  --mode both \
  --corrections-only \
  --overwrite
```
- Input: `test/bbox_annotations_3d_obb_corrected/<id>.pkl` + Pi3 `predictions.npz`
- Outputs:
  - `world_annotations/test/bbox_annotations_3d_obb_corrected_final/<id>.pkl`
  - `world_annotations/test/bbox_annotations_3d_obb_corrected_camera/<id>.pkl`

*(For a single view use `--mode final` or `--mode camera`.)*

---

### Stage 4 — Corrected 4D BBoxes (object-permanence filling) (`corrected_4d_bbox_generator.py`)
```bash
python datasets/preprocess/annotations/corrected_4d_bbox_generator.py \
  --phase test \
  --corrections-only \
  --overwrite
```
- Input: `test/bbox_annotations_3d_obb_corrected/<id>.pkl`
- Output: `/data/rohith/ag/world_annotations/test/bbox_annotations_4d_corrected/<id>.pkl`

---

### Stage 5 — Augment relationships (`augment_relationships_test.py`)
Independent of the bbox branch; needs only GT + Firebase relationship corrections.
*(This script has **no `--phase`** flag — it is test-specific.)*
```bash
python datasets/preprocess/annotations/augment_relationships_test.py \
  --overwrite
```
- Inputs: AG GT annotations, `manual_corrections/` (correction relationships)
- Output: `/data/rohith/ag/wsg_2d_augmentations/<id>.pkl`

---

### Stage 6 — Merge relationships + corrected 4D BBoxes (`combine_world4d_relationships_test.py`)
Reads the **`bbox_annotations_4d_corrected`** dir (NOT the stale raw `bbox_annotations_4d`).
```bash
python datasets/preprocess/annotations/combine_world4d_relationships_test.py \
  --phase test \
  --overwrite
```
- Inputs: `wsg_2d_augmentations/<id>.pkl` (Stage 5) + `test/bbox_annotations_4d_corrected/<id>.pkl` (Stage 4)
- Output: `/data/rohith/ag/world4d_rel_annotations/test/<id>.pkl`

---

### Stage 7 — Final unified scene graphs (`world_scene_graph_generator.py`)
*(This script has **no `--corrections-only`** flag — it auto-discovers available videos.)*
```bash
python datasets/preprocess/annotations/world_scene_graph_generator.py \
  --phase test \
  --overwrite
```
- Inputs: `wsg_2d_augmentations/<id>.pkl` + `test/bbox_annotations_4d_corrected/<id>.pkl`
- Output: `/data/rohith/ag/world_scene_graph/<id>.pkl`

---

## Complete Execution Script (full rebuild, overwrite everything)

```bash
#!/bin/bash
set -e
# Run from the repository root.
ANN=datasets/preprocess/annotations

# --- Stage 0: raw prerequisites (skip if test/bbox_annotations_3d_obb already exists) ---
echo "== Stage 0a: GT annotations =="
python $ANN/raw/gt_generator.py

echo "== Stage 0b: raw world 3D OBB (auto floor), test only =="
python $ANN/raw/bb3D_generator_gt_obb.py --phase test --overwrite

# --- Stage 1: Firebase corrections ---
echo "== Stage 1: download manual corrections =="
python $ANN/download_floor_manual_corrections.py --overwrite

# --- Stage 2: corrected world bboxes ---
echo "== Stage 2: corrected world bboxes =="
python $ANN/corrected_world_bbox_generator.py --phase test --corrections-only --overwrite

# --- Stage 3: corrected frame bboxes (FINAL + CAMERA) ---
echo "== Stage 3: corrected frame bboxes (both views) =="
python $ANN/corrected_frame_bbox_generator.py --phase test --mode both --corrections-only --overwrite

# --- Stage 4: corrected 4D bboxes ---
echo "== Stage 4: corrected 4D bboxes =="
python $ANN/corrected_4d_bbox_generator.py --phase test --corrections-only --overwrite

# --- Stage 5: augment relationships ---
echo "== Stage 5: augment relationships =="
python $ANN/augment_relationships_test.py --overwrite

# --- Stage 6: merge relationships + corrected 4D ---
echo "== Stage 6: combine relationships + 4D =="
python $ANN/combine_world4d_relationships_test.py --phase test --overwrite

# --- Stage 7: scene graphs ---
echo "== Stage 7: world scene graphs =="
python $ANN/world_scene_graph_generator.py --phase test --overwrite

echo "== Done. =="
```

---

## Dependencies Between Stages

```
Stage 0a gt_generator ──► Stage 0b bb3D_generator_gt_obb (test/bbox_annotations_3d_obb)
                                         │
Stage 1 download_floor_manual_corrections│ (Firebase → manual_corrections/)
                          └──────────────┤
                                         ▼
                         Stage 2 corrected_world_bbox_generator
                              (test/bbox_annotations_3d_obb_corrected)
                                ├───────────────┬──────────────────────┐
                                ▼               ▼                       
                  Stage 4 corrected_4d   Stage 3 corrected_frame  (side branch:
                  (..._4d_corrected)     (..._corrected_final/_camera)  not on SG path)
                                │
   Stage 5 augment_relationships_test (wsg_2d_augmentations/)  ── independent ──┐
                                │                                               │
                                ▼                                               ▼
                  Stage 6 combine_world4d_relationships_test ◄──────────────────┘
                              (world4d_rel_annotations/test/)
                                │
                                ▼
                  Stage 7 world_scene_graph_generator (world_scene_graph/)
```

**Critical dependency:** Stage 2 blocks all downstream geometry stages; Stages 5 and (4) both feed Stages 6/7.

---

## Per-script flag reference (post-refactor)

| Script | Valid flags |
|---|---|
| `raw/gt_generator.py` | `--config --ag_root_directory --output_dir` |
| `raw/bb3D_generator_gt_obb.py` | `--config --ag_root_directory --dynamic_scene_dir_path --output_human_dir_path --phase --split --video --overwrite --visualize` |
| `download_floor_manual_corrections.py` | `--config --output-dir --video --list --overwrite` |
| `corrected_world_bbox_generator.py` | `--config --ag_root_directory --dynamic_scene_dir_path --manual_corrections_dir --phase --split --video --corrections-only --gdino-score-threshold --overwrite --visualize` |
| `corrected_frame_bbox_generator.py` | `--config --ag_root_directory --dynamic_scene_dir_path --phase --split --video --mode --corrections-only --overwrite --visualize` |
| `corrected_4d_bbox_generator.py` | `--config --ag_root_directory --dynamic_scene_dir_path --phase --split --video --corrections-only --overwrite --visualize` |
| `augment_relationships_test.py` | `--config --ag_root_directory --corrections_dir --output_dir --video --overwrite` *(no `--phase`)* |
| `combine_world4d_relationships_test.py` | `--config --ag_root_directory --phase --video --overwrite` |
| `world_scene_graph_generator.py` | `--config --ag_root_directory --phase --augmented_rel_dir --bbox_4d_corrected_dir --output_dir --video --overwrite` *(no `--corrections-only`)* |

All paths and `phase` default from `configs/annotation_utd.yaml`; CLI flags override config, which overrides built-in defaults.

---

## Output Directory Structure

```
/data/rohith/ag/
├── ag4D/gt_annotations/                                  # Stage 0a
├── world_annotations/
│   ├── manual_corrections/                               # Stage 1
│   └── test/
│       ├── bbox_annotations_3d_obb/                      # Stage 0b (raw prerequisite)
│       ├── bbox_annotations_3d_obb_corrected/            # Stage 2
│       ├── bbox_annotations_3d_obb_corrected_final/      # Stage 3 (final view)
│       ├── bbox_annotations_3d_obb_corrected_camera/     # Stage 3 (camera view)
│       └── bbox_annotations_4d_corrected/                # Stage 4
├── wsg_2d_augmentations/                                 # Stage 5
├── world4d_rel_annotations/test/                         # Stage 6
└── world_scene_graph/                                    # Stage 7
```

---

## Single-video runs (debugging)

Every generation script accepts `--video <id>.mp4` to process one video:
```bash
python datasets/preprocess/annotations/raw/bb3D_generator_gt_obb.py        --phase test --video 001YG.mp4 --overwrite
python datasets/preprocess/annotations/corrected_world_bbox_generator.py   --phase test --video 001YG.mp4 --overwrite
python datasets/preprocess/annotations/corrected_4d_bbox_generator.py      --phase test --video 001YG.mp4 --overwrite
python datasets/preprocess/annotations/combine_world4d_relationships_test.py --phase test --video 001YG.mp4 --overwrite
python datasets/preprocess/annotations/world_scene_graph_generator.py      --phase test --video 001YG.mp4 --overwrite
```

---

## Monitoring & Validation

```bash
# How many corrections are in Firebase
python datasets/preprocess/annotations/download_floor_manual_corrections.py --list

# Output counts per stage
ls /data/rohith/ag/world_annotations/test/bbox_annotations_3d_obb_corrected/ | wc -l
ls /data/rohith/ag/world_annotations/test/bbox_annotations_4d_corrected/ | wc -l
find /data/rohith/ag/world_scene_graph -name "*.pkl" | wc -l
```

---

## Troubleshooting

**Stage 2: "no manual corrections found" / "no corrected world bbox PKL"**
→ Run Stage 1; verify `world_annotations/manual_corrections/<key>.pkl` exists.

**Stage 2: corrected output is empty / raw OBB not found**
→ Phase mismatch. Ensure Stage 0b ran with `--phase test` so the raw OBBs are in
`world_annotations/test/bbox_annotations_3d_obb/` (where Stage 2 reads them).

**Stage 3/visualization tries to open Rerun**
→ Use `main()` (the default entry point now) without `--visualize`. `--visualize`
requires `--video` and is for manual inspection only.

**Stage 6 label-mismatch warnings**
→ Check `verify.txt` in the output dir for per-frame 2D/3D label mismatches.

---

## Notes

- The raw downstream outputs (`bbox_annotations_3d_obb_bridge`, `_final`,
  `bbox_annotations_4d`) are **not** part of this corrected build. Leaving any old
  copies around is harmless — nothing in Stages 2–7 reads them.
- Estimated time: ~45–90 min for 50–100 test videos; peak ~8–16 GB RAM per video.
  Scripts process one video at a time — wrap stages in GNU Parallel/Ray for scale.
