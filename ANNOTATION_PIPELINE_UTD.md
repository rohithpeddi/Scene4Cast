# Unified Test Dataset (UTD) Annotation Pipeline
## Complete Guide to Generating Corrected Annotations

---

## Overview

This document outlines the complete pipeline for generating all annotations for the **test dataset** using **manually corrected floor transforms** downloaded from Firebase.

**Key Assumption:** You have already manually corrected the floor transforms and stored them in Firebase under the paths:
- `worldframe_obb/floor_corrections/<video_key>/latest`
- `worldframe_obb/xy_alignments/<video_key>/latest`  
- `worldframe_obb/final_alignments/<video_key>/latest`

---

## Transformation Order Overview

The corrected annotation pipeline applies 4 transformations in sequence:

```
x_canonical = T_XY ∘ T_delta ∘ T_auto ∘ x_world
```

1. **T_auto** (in original PKLs)  
   - Automatic floor alignment from SMPL-scene correspondences
   - Already baked into original `bbox_annotations_3d_obb/*.pkl`

2. **T_delta** (from `floor_correction`)  
   - Manual floor correction with delta transform
   - Euler angles (rx, ry, rz) + scale + translation
   - Applied to floor mesh, bboxes, and point clouds

3. **T_XY** (from `xy_alignment`)  
   - Automated XY-plane alignment
   - Aligns corrected floor normal → +Z axis
   - Aligns local X-axis → world X-axis

4. **Combined Transform** (from `final_alignment.combined_transform`)  
   - Pre-computed 4×4 matrix combining all 3 steps
   - If present, this replaces manual composition of steps 1–3

---

## Pipeline Execution Order (TEST Dataset)

### Stage 1: Download Corrected Floor Transforms
**Script:** `download_floor_manual_corrections.py`  
**Time:** ~5-30 seconds (depends on Firebase latency)

```bash
cd datasets/preprocess/annotations
python download_floor_manual_corrections.py \
  --output-dir /data/rohith/ag/world_annotations/manual_corrections
```

**Input:** Firebase
**Output:** `/data/rohith/ag/world_annotations/manual_corrections/<video_key>.pkl`

**Output Structure:**
```
{
  "video_id": "001YG.mp4",
  "video_key": "001YG",
  "floor_correction": {...},        # Step 2 (delta_transform)
  "xy_alignment": {...},            # Step 3 (alignment_transform)
  "final_alignment": {...},         # Step 4 (combined_transform_4x4)
  "download_metadata": {
    "has_floor_correction": bool,
    "has_xy_alignment": bool,
    "has_final_alignment": bool,
  }
}
```

---

### Stage 2: Generate Corrected World-Frame 3D BBoxes
**Script:** `corrected_world_bbox_generator.py`  
**Time:** ~10 mins per video (depends on object count and point density)

```bash
python corrected_world_bbox_generator.py \
  --ag_root_directory /data/rohith/ag \
  --phase test \
  --corrections-only \
  --overwrite
```

**Inputs:**
- `/data/rohith/ag/world_annotations/manual_corrections/<video_key>.pkl` (corrected transforms)
- `/data/rohith/ag/world_annotations/bbox_annotations_3d_obb/<video_id>.pkl` (original bboxes)

**Output:** `/data/rohith/ag/world_annotations/test/bbox_annotations_3d_obb_corrected/<video_id>.pkl`

**Key Operations:**
- Loads original world bboxes (AABB, OBB-floor-parallel, OBB-arbitrary)
- Applies corrected floor transform from downloaded Firebase data
- Stores corrected corners + metadata in world frame

---

### Stage 3: Generate Corrected Frame-Level 3D BBoxes
**Script:** `corrected_frame_bbox_generator.py`  
**Time:** ~5-10 mins per video

This stage creates two outputs:

#### 3a. FINAL-Frame BBoxes (Canonical/Gravity-Aligned)
```bash
python corrected_frame_bbox_generator.py \
  --ag_root_directory /data/rohith/ag \
  --dynamic_scene_dir_path /data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic \
  --phase test \
  --mode final \
  --corrections-only \
  --overwrite
```

**Output:** `/data/rohith/ag/world_annotations/test/bbox_annotations_3d_obb_corrected_final/<video_id>.pkl`

**Contains:**
- BBox corners transformed to FINAL (canonical) frame
- Point cloud transformed to FINAL frame
- Camera poses transformed to FINAL frame
- Floor mesh in FINAL frame

#### 3b. CAMERA-Frame BBoxes (Per-Frame Camera Coords)
```bash
python corrected_frame_bbox_generator.py \
  --ag_root_directory /data/rohith/ag \
  --dynamic_scene_dir_path /data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic \
  --phase test \
  --mode camera \
  --corrections-only \
  --overwrite
```

**Output:** `/data/rohith/ag/world_annotations/test/bbox_annotations_3d_obb_corrected_camera/<video_id>.pkl`

**Contains:**
- Per-frame object BBox corners in camera coordinate system
- Per-frame point cloud in camera coordinates
- T_world2cam transforms

---

### Stage 4: Generate Corrected 4D BBoxes (Object Permanence Filling)
**Script:** `corrected_4d_bbox_generator.py`  
**Time:** ~10-15 mins per video

```bash
python corrected_4d_bbox_generator.py \
  --ag_root_directory /data/rohith/ag \
  --dynamic_scene_dir_path /data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic \
  --phase test \
  --corrections-only \
  --overwrite
```

**Input:** `/data/rohith/ag/world_annotations/test/bbox_annotations_3d_obb_corrected/<video_id>.pkl`

**Output:** `/data/rohith/ag/world_annotations/test/bbox_annotations_4d_corrected/<video_id>.pkl`

**Key Operations:**
- Extracts 3D world bboxes from corrected source
- Applies object-permanence filling (fills in missing detections for static objects)
- Uses static-union logic to bridge gaps
- Outputs per-frame and temporally-filled annotations in FINAL coords

---

### Stage 5: Augment Relationships with Manual Corrections
**Script:** `augment_relationships_test.py`  
**Time:** ~2-5 mins per video

```bash
python augment_relationships_test.py \
  --ag_root_directory /data/rohith/ag \
  --output_dir /data/rohith/ag/wsg_2d_augmentations \
  --corrections_dir /data/rohith/ag/world_annotations/manual_corrections \
  --phase test \
  --overwrite
```

**Inputs:**
- Original GT annotations (from AG dataset)
- Human-corrected annotations (manual corrections from Firebase)

**Output:** `/data/rohith/ag/wsg_2d_augmentations/<video_id>.pkl`

**Contains:**
```
{
  "video_id": str,
  "frames": {
    "video_id/000042.png": {
      "person_bbox": np.ndarray,
      "objects": [
        {
          "class": str,
          "label": str,
          "source": "gt" | "correction",  # Tags GT vs manually corrected
          "attention": [str],
          "contacting": [str],
          "spatial": [str],
          "bbox_2d": np.ndarray,
        },
        ...
      ]
    },
    ...
  }
}
```

---

### Stage 6: Merge Relationships with Corrected 4D BBoxes
**Script:** `combine_world4d_relationships_test.py`  
**Time:** ~5-10 mins per video

```bash
python combine_world4d_relationships_test.py \
  --ag_root_directory /data/rohith/ag \
  --augmented_rel_dir /data/rohith/ag/wsg_2d_augmentations \
  --bbox_4d_dir /data/rohith/ag/world_annotations/test/bbox_annotations_4d_corrected \
  --output_dir /data/rohith/ag/world4d_rel_annotations/test \
  --overwrite
```

**Inputs:**
- Augmented relationships (from Stage 5)
- Corrected 4D BBoxes (from Stage 4)

**Output:** `/data/rohith/ag/world4d_rel_annotations/test/<video_id>.pkl`

**Key Operations:**
- Merges 2D relationships with 3D bbox data
- Verifies label consistency between 2D and 3D annotations
- Extracts person 3D data (primary actor)
- Filters camera poses to match frames with relationship data
- Stores unified per-frame object + relationship + 3D bbox information

---

### Stage 7: Generate Final Unified Scene Graphs
**Script:** `world_scene_graph_generator.py`  
**Time:** ~5-10 mins per video

```bash
python world_scene_graph_generator.py \
  --ag_root_directory /data/rohith/ag \
  --augmented_rel_dir /data/rohith/ag/wsg_2d_augmentations \
  --bbox_4d_dir /data/rohith/ag/world_annotations/test/bbox_annotations_4d_corrected \
  --output_dir /data/rohith/ag/world_scene_graph \
  --corrections-only \
  --overwrite
```

**Inputs:**
- Augmented relationships
- Corrected 4D bboxes

**Output:** `/data/rohith/ag/world_scene_graph/<video_id>.pkl`

**Contains:**
```
{
  "video_id": str,
  "frames": {
    "video_id/000042.png": {
      "person": {...},          # person bbox + relationships
      "objects": [
        {
          "label": str,
          "class_idx": int,
          "bbox_3d": {...},     # 3D bbox in FINAL frame
          "relationships": {
            "attention": [...],
            "contacting": [...],
            "spatial": [...]
          },
          "source": str,        # "gt", "correction", "gdino"
        },
        ...
      ],
      "camera_pose": np.ndarray,  # 4×4 extrinsics
    },
    ...
  },
  "metadata": {
    "has_corrections": bool,
    "correction_sources": [...],
  }
}
```

---

## Complete Execution Script

Run all stages sequentially:

```bash
#!/bin/bash
set -e

CONFIG="/path/to/configs/annotation_utd.yaml"
AG_ROOT="/data/rohith/ag"
DYNAMIC_SCENE="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic"

echo "=========================================="
echo "Stage 1: Download Manual Corrections"
echo "=========================================="
cd datasets/preprocess/annotations
python download_floor_manual_corrections.py \
  --output-dir "$AG_ROOT/world_annotations/manual_corrections"

echo "=========================================="
echo "Stage 2: Generate Corrected World BBoxes"
echo "=========================================="
python corrected_world_bbox_generator.py \
  --ag_root_directory "$AG_ROOT" \
  --phase test \
  --corrections-only \
  --overwrite

echo "=========================================="
echo "Stage 3a: Generate FINAL-Frame BBoxes"
echo "=========================================="
python corrected_frame_bbox_generator.py \
  --ag_root_directory "$AG_ROOT" \
  --dynamic_scene_dir_path "$DYNAMIC_SCENE" \
  --phase test \
  --mode final \
  --corrections-only \
  --overwrite

echo "=========================================="
echo "Stage 3b: Generate CAMERA-Frame BBoxes"
echo "=========================================="
python corrected_frame_bbox_generator.py \
  --ag_root_directory "$AG_ROOT" \
  --dynamic_scene_dir_path "$DYNAMIC_SCENE" \
  --phase test \
  --mode camera \
  --corrections-only \
  --overwrite

echo "=========================================="
echo "Stage 4: Generate 4D BBoxes"
echo "=========================================="
python corrected_4d_bbox_generator.py \
  --ag_root_directory "$AG_ROOT" \
  --dynamic_scene_dir_path "$DYNAMIC_SCENE" \
  --phase test \
  --corrections-only \
  --overwrite

echo "=========================================="
echo "Stage 5: Augment Relationships"
echo "=========================================="
python augment_relationships_test.py \
  --ag_root_directory "$AG_ROOT" \
  --output_dir "$AG_ROOT/wsg_2d_augmentations" \
  --corrections_dir "$AG_ROOT/world_annotations/manual_corrections" \
  --phase test \
  --overwrite

echo "=========================================="
echo "Stage 6: Combine Relationships + 4D BBoxes"
echo "=========================================="
python combine_world4d_relationships_test.py \
  --ag_root_directory "$AG_ROOT" \
  --overwrite

echo "=========================================="
echo "Stage 7: Generate Scene Graphs"
echo "=========================================="
python world_scene_graph_generator.py \
  --ag_root_directory "$AG_ROOT" \
  --corrections-only \
  --overwrite

echo "=========================================="
echo "✓ All stages completed!"
echo "=========================================="
```

---

## Dependencies Between Stages

```
Stage 1 (Download Corrections)
    ↓
Stage 2 (Corrected World BBoxes)
    ↓
Stage 3 (Frame BBoxes: FINAL + CAMERA)
    ├─→ Stage 3a (FINAL)
    └─→ Stage 3b (CAMERA)
    ↓
Stage 4 (4D BBoxes with Permanence Filling)
    ↓
Stage 5 (Augment Relationships) ←──┐
    ↓                               │
Stage 6 (Merge Relationships + 4D) ←──┘
    ↓
Stage 7 (Final Scene Graphs)
```

**Critical Dependency:** Stage 2 blocks all downstream stages (it's the foundation)

---

## Configuration File Usage

All paths and parameters are centralized in `configs/annotation_utd.yaml`.

To override parameters at runtime:
```bash
# Single video
python script.py --video 001YG.mp4

# Specific split
python script.py --split 04

# Skip existing files
python script.py --corrections-only

# Force regeneration
python script.py --overwrite
```

---

## Output Directory Structure

```
/data/rohith/ag/
├── world_annotations/
│   ├── manual_corrections/              # Stage 1 output
│   │   ├── 001YG.pkl
│   │   ├── 002AB.pkl
│   │   └── ...
│   ├── test/
│   │   ├── bbox_annotations_3d_obb_corrected/         # Stage 2 output
│   │   ├── bbox_annotations_3d_obb_corrected_final/   # Stage 3a output
│   │   ├── bbox_annotations_3d_obb_corrected_camera/  # Stage 3b output
│   │   └── bbox_annotations_4d_corrected/             # Stage 4 output
├── wsg_2d_augmentations/                # Stage 5 output
│   ├── 001YG.pkl
│   └── ...
├── world4d_rel_annotations/
│   └── test/                            # Stage 6 output
│       ├── 001YG.pkl
│       └── ...
└── world_scene_graph/                   # Stage 7 output
    ├── 001YG.pkl
    └── ...
```

---

## Monitoring & Validation

### Check Firebase Corrections
```bash
python download_floor_manual_corrections.py --list
```

### Verify Output Files Exist
```bash
ls -lah /data/rohith/ag/world_annotations/test/bbox_annotations_3d_obb_corrected/ | wc -l
ls -lah /data/rohith/ag/world4d_rel_annotations/test/ | wc -l
```

### Count Processed Videos
```bash
find /data/rohith/ag/world_scene_graph -name "*.pkl" | wc -l
```

---

## Troubleshooting

### Issue: Stage 2 fails with "no corrected world bbox PKL"
**Cause:** Firebase corrections weren't downloaded or don't exist for that video.  
**Solution:** Run Stage 1 first, verify output in `/data/rohith/ag/world_annotations/manual_corrections/`

### Issue: Stage 3 fails with "no camera poses available"
**Cause:** Original points/cameras missing from Pi3 data.  
**Solution:** Verify `/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic/<video>_10/predictions.npz` exists

### Issue: Stage 6 label mismatch warnings
**Cause:** 2D relationship labels don't match 3D bbox object labels.  
**Solution:** Check `verify.txt` in output directory for detailed mismatches

---

## Memory & Performance Notes

- **Peak Memory:** ~8-16 GB per video (depends on frame count and object density)
- **Parallelization:** Each stage processes one video at a time (no built-in batching)
- **Estimated Total Time:** ~45-90 mins for 50-100 test videos

For large batches, consider wrapping stages in a parallel job scheduler (e.g., GNU Parallel, Ray).

---

## Next Steps

After completing all stages:

1. **Validation:** Run evaluation scripts to verify annotation quality
2. **Visualization:** Use Rerun to visualize scene graphs
3. **Export:** Convert PKLs to your downstream format (COCO, JSON, etc.)
4. **Version:** Document the correction sources in metadata
