#!/usr/bin/env bash
# ============================================================================
# run_train_annotation_pipeline.sh
# ============================================================================
# Runs the full corrected annotation pipeline for the TRAIN dataset split.
#
# Prerequisites:
#   - Firebase credentials configured for download_floor_manual_corrections.py
#   - Dynamic scene predictions available at DYNAMIC_SCENE_DIR
#   - Segmentation masks available under AG_ROOT/segmentation/
#
# Usage:
#   chmod +x scripts/run_train_annotation_pipeline.sh
#   ./scripts/run_train_annotation_pipeline.sh
# ============================================================================

set -euo pipefail

# -------------------------------------------------------------------
# Configuration — adjust these paths to match your target machine
# -------------------------------------------------------------------
AG_ROOT="/data/rohith/ag"
DYNAMIC_SCENE_DIR="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic"
MANUAL_CORRECTIONS_DIR="${AG_ROOT}/world_annotations/manual_corrections"
CORRECTIONS_DIR="${AG_ROOT}/wsg_corrections"
AUGMENTATION_OUTPUT_DIR="${AG_ROOT}/wsg_2d_augmentations"

PHASE="train"
SCRIPT_DIR="datasets/preprocess/annotations"

echo "============================================================"
echo "  Corrected Annotation Pipeline — ${PHASE} split"
echo "============================================================"
echo "  AG_ROOT:              ${AG_ROOT}"
echo "  DYNAMIC_SCENE_DIR:    ${DYNAMIC_SCENE_DIR}"
echo "  MANUAL_CORRECTIONS:   ${MANUAL_CORRECTIONS_DIR}"
echo "  PHASE:                ${PHASE}"
echo "============================================================"
echo ""

# -------------------------------------------------------------------
# Step 1: Download floor corrections from Firebase
# -------------------------------------------------------------------
echo "[Step 1/6] Downloading floor corrections from Firebase..."
python ${SCRIPT_DIR}/download_floor_manual_corrections.py \
    --output-dir "${MANUAL_CORRECTIONS_DIR}" \
    --overwrite

echo "[Step 1/6] Done."
echo ""

# -------------------------------------------------------------------
# Step 2: Generate corrected world 3D bboxes (NEEDS MASKS)
# -------------------------------------------------------------------
echo "[Step 2/6] Generating corrected world 3D bboxes..."
python ${SCRIPT_DIR}/corrected_world_bbox_generator.py \
    --ag_root_directory "${AG_ROOT}" \
    --dynamic_scene_dir_path "${DYNAMIC_SCENE_DIR}" \
    --manual_corrections_dir "${MANUAL_CORRECTIONS_DIR}" \
    --phase "${PHASE}" \
    --corrections-only \
    --overwrite

echo "[Step 2/6] Done."
echo ""

# -------------------------------------------------------------------
# Step 3: Generate corrected frame-level bboxes (FINAL + CAMERA)
# -------------------------------------------------------------------
echo "[Step 3/6] Generating corrected frame-level bboxes..."
python ${SCRIPT_DIR}/corrected_frame_bbox_generator.py \
    --ag_root_directory "${AG_ROOT}" \
    --dynamic_scene_dir_path "${DYNAMIC_SCENE_DIR}" \
    --phase "${PHASE}" \
    --mode both \
    --corrections-only \
    --overwrite

echo "[Step 3/6] Done."
echo ""

# -------------------------------------------------------------------
# Step 4: Generate corrected 4D bboxes (object permanence)
# -------------------------------------------------------------------
echo "[Step 4/6] Generating corrected 4D bboxes..."
python ${SCRIPT_DIR}/corrected_4d_bbox_generator.py \
    --ag_root_directory "${AG_ROOT}" \
    --dynamic_scene_dir_path "${DYNAMIC_SCENE_DIR}" \
    --phase "${PHASE}" \
    --corrections-only \
    --overwrite

echo "[Step 4/6] Done."
echo ""

# -------------------------------------------------------------------
# Step 5: Augment train relationships
# NOTE: For train, you may need augment_relationships_train.py instead.
# Update this step if the train augmentation script differs.
# -------------------------------------------------------------------
echo "[Step 5/6] Augmenting train relationships..."
echo "[WARNING] Using augment_relationships_test.py — replace with"
echo "          augment_relationships_train.py if the train augmentation"
echo "          script differs."
python ${SCRIPT_DIR}/augment_relationships_test.py \
    --ag_root_directory "${AG_ROOT}" \
    --corrections_dir "${CORRECTIONS_DIR}" \
    --output_dir "${AUGMENTATION_OUTPUT_DIR}" \
    --overwrite

echo "[Step 5/6] Done."
echo ""

# -------------------------------------------------------------------
# Step 6: Combine 4D bboxes + train relationships
# -------------------------------------------------------------------
echo "[Step 6/6] Combining 4D bboxes with augmented train relationships..."
python ${SCRIPT_DIR}/combine_world4d_relationships_test.py \
    --ag_root_directory "${AG_ROOT}" \
    --phase "${PHASE}" \
    --overwrite

echo "[Step 6/6] Done."
echo ""

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo "============================================================"
echo "  Pipeline complete! Output structure:"
echo ""
echo "  ${AG_ROOT}/world_annotations/${PHASE}/"
echo "    ├── bbox_annotations_3d_obb_corrected/"
echo "    ├── bbox_annotations_3d_obb_corrected_final/"
echo "    ├── bbox_annotations_3d_obb_corrected_camera/"
echo "    └── bbox_annotations_4d_corrected/"
echo ""
echo "  ${AG_ROOT}/wsg_2d_augmentations/"
echo "  ${AG_ROOT}/world4d_rel_annotations/${PHASE}/"
echo "============================================================"
