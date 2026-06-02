#!/usr/bin/env bash
# ============================================================================
# run_test_annotation_pipeline.sh
# ============================================================================
# Runs the full corrected-floor annotation pipeline for the TEST split,
# end to end, regenerating everything (--overwrite).
#
# See ANNOTATION_PIPELINE_UTD.md for the full data-flow description.
#
# Prerequisites:
#   - Firebase credentials configured for download_floor_manual_corrections.py
#   - Dynamic scene predictions at DYNAMIC_SCENE_DIR/<video>_10/predictions.npz
#   - Segmentation masks under AG_ROOT/segmentation/
#   - Run from the repository root:
#       chmod +x scripts/run_test_annotation_pipeline.sh
#       ./scripts/run_test_annotation_pipeline.sh
#
# All paths / phase also live in configs/annotation_utd.yaml (phase: test);
# the CLI flags below mirror that config and override it per-run.
# ============================================================================

set -euo pipefail

# -------------------------------------------------------------------
# Configuration — adjust these paths to match your target machine
# -------------------------------------------------------------------
AG_ROOT="/data/rohith/ag"
DYNAMIC_SCENE_DIR="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic"
MANUAL_CORRECTIONS_DIR="${AG_ROOT}/world_annotations/manual_corrections"
# Human relationship corrections (separate from floor corrections above).
CORRECTIONS_DIR="${AG_ROOT}/wsg_corrections"
AUGMENTATION_OUTPUT_DIR="${AG_ROOT}/wsg_2d_augmentations"

PHASE="test"
SCRIPT_DIR="datasets/preprocess/annotations"

# Stage 0 (raw prerequisites) is EXPENSIVE and usually already done.
# It produces world_annotations/test/bbox_annotations_3d_obb (the floor-independent
# world OBBs the corrected pipeline forks from). Set to 1 to (re)generate them.
RUN_RAW_PREREQS="${RUN_RAW_PREREQS:-0}"

echo "============================================================"
echo "  Corrected Annotation Pipeline — ${PHASE} split"
echo "============================================================"
echo "  AG_ROOT:              ${AG_ROOT}"
echo "  DYNAMIC_SCENE_DIR:    ${DYNAMIC_SCENE_DIR}"
echo "  MANUAL_CORRECTIONS:   ${MANUAL_CORRECTIONS_DIR}"
echo "  PHASE:                ${PHASE}"
echo "  RUN_RAW_PREREQS:      ${RUN_RAW_PREREQS}"
echo "============================================================"
echo ""

# -------------------------------------------------------------------
# Stage 0 (optional): raw prerequisites — gt + world OBBs (auto floor)
# Must run with --phase test so output lands in world_annotations/test/...
# where the corrected pipeline reads it.
# -------------------------------------------------------------------
if [[ "${RUN_RAW_PREREQS}" == "1" ]]; then
    echo "[Step 0a/9] Generating GT annotations (gt_generator.py)..."
    python ${SCRIPT_DIR}/raw/gt_generator.py \
        --ag_root_directory "${AG_ROOT}"
    echo "[Step 0a/9] Done."
    echo ""

    echo "[Step 0b/9] Generating raw world 3D OBBs (auto floor)..."
    python ${SCRIPT_DIR}/raw/bb3D_generator_gt_obb.py \
        --ag_root_directory "${AG_ROOT}" \
        --dynamic_scene_dir_path "${DYNAMIC_SCENE_DIR}" \
        --phase "${PHASE}" \
        --overwrite
    echo "[Step 0b/9] Done."
    echo ""
else
    echo "[Step 0/9] Skipping raw prerequisites (RUN_RAW_PREREQS=0)."
    echo "           Set RUN_RAW_PREREQS=1 to (re)generate"
    echo "           world_annotations/${PHASE}/bbox_annotations_3d_obb."
    echo ""
fi

# -------------------------------------------------------------------
# Step 1: Download floor corrections from Firebase
# -------------------------------------------------------------------
echo "[Step 1/9] Downloading floor corrections from Firebase..."
python ${SCRIPT_DIR}/download_floor_manual_corrections.py \
    --output-dir "${MANUAL_CORRECTIONS_DIR}" \
    --overwrite
echo "[Step 1/9] Done."
echo ""

# -------------------------------------------------------------------
# Step 2: Generate corrected world 3D bboxes (NEEDS MASKS + raw OBBs)
# -------------------------------------------------------------------
echo "[Step 2/9] Generating corrected world 3D bboxes..."
python ${SCRIPT_DIR}/corrected_world_bbox_generator.py \
    --ag_root_directory "${AG_ROOT}" \
    --dynamic_scene_dir_path "${DYNAMIC_SCENE_DIR}" \
    --manual_corrections_dir "${MANUAL_CORRECTIONS_DIR}" \
    --phase "${PHASE}" \
    --corrections-only \
    --overwrite
echo "[Step 2/9] Done."
echo ""

# -------------------------------------------------------------------
# Step 3: Generate corrected frame-level bboxes (FINAL + CAMERA)
# (Side branch — used for training/visualization; not on the scene-graph path.)
# -------------------------------------------------------------------
echo "[Step 3/9] Generating corrected frame-level bboxes (both views)..."
python ${SCRIPT_DIR}/corrected_frame_bbox_generator.py \
    --ag_root_directory "${AG_ROOT}" \
    --dynamic_scene_dir_path "${DYNAMIC_SCENE_DIR}" \
    --phase "${PHASE}" \
    --mode both \
    --corrections-only \
    --overwrite
echo "[Step 3/9] Done."
echo ""

# -------------------------------------------------------------------
# Step 4: Generate corrected 4D bboxes (object permanence)
# -------------------------------------------------------------------
echo "[Step 4/9] Generating corrected 4D bboxes..."
python ${SCRIPT_DIR}/corrected_4d_bbox_generator.py \
    --ag_root_directory "${AG_ROOT}" \
    --dynamic_scene_dir_path "${DYNAMIC_SCENE_DIR}" \
    --phase "${PHASE}" \
    --corrections-only \
    --overwrite
echo "[Step 4/9] Done."
echo ""

# -------------------------------------------------------------------
# Step 5: Augment test relationships (independent of steps 2-4)
# NOTE: augment_relationships_test.py has no --phase flag (it is test-specific).
# -------------------------------------------------------------------
echo "[Step 5/9] Augmenting test relationships..."
python ${SCRIPT_DIR}/augment_relationships_test.py \
    --ag_root_directory "${AG_ROOT}" \
    --corrections_dir "${CORRECTIONS_DIR}" \
    --output_dir "${AUGMENTATION_OUTPUT_DIR}" \
    --overwrite
echo "[Step 5/9] Done."
echo ""

# -------------------------------------------------------------------
# Step 6: Combine corrected 4D bboxes + augmented test relationships
# Reads world_annotations/test/bbox_annotations_4d_corrected (NOT the raw dir).
# -------------------------------------------------------------------
echo "[Step 6/9] Combining 4D bboxes with augmented test relationships..."
python ${SCRIPT_DIR}/combine_world4d_relationships_test.py \
    --ag_root_directory "${AG_ROOT}" \
    --phase "${PHASE}" \
    --overwrite
echo "[Step 6/9] Done."
echo ""

# -------------------------------------------------------------------
# Step 7: Generate final unified world scene graphs
# NOTE: world_scene_graph_generator.py has no --corrections-only flag;
#       it auto-discovers available videos. For test it reads the augmented
#       relationships from wsg_2d_augmentations (its main() default) + the
#       corrected 4D bboxes from world_annotations/test/bbox_annotations_4d_corrected.
# CAVEAT: merge_video() expects per-frame ``observed``/``missing`` relationship
#       lists. The augmentation scripts currently emit a different schema
#       (objects/source), so relationship fields may come out empty until the
#       generator's format handling is reconciled. The geometry merge is fine.
#       (The authoritative relationship+geometry merge is Step 6.)
# -------------------------------------------------------------------
echo "[Step 7/9] Generating world scene graphs..."
python ${SCRIPT_DIR}/world_scene_graph_generator.py \
    --ag_root_directory "${AG_ROOT}" \
    --phase "${PHASE}" \
    --output_dir "${AG_ROOT}/world_annotations/${PHASE}/world_scene_graph" \
    --overwrite
echo "[Step 7/9] Done."
echo ""

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo "============================================================"
echo "  Pipeline complete! Output structure:"
echo ""
echo "  ${AG_ROOT}/world_annotations/${PHASE}/"
echo "    ├── bbox_annotations_3d_obb/                  (raw prerequisite)"
echo "    ├── bbox_annotations_3d_obb_corrected/        (Step 2)"
echo "    ├── bbox_annotations_3d_obb_corrected_final/  (Step 3)"
echo "    ├── bbox_annotations_3d_obb_corrected_camera/ (Step 3)"
echo "    └── bbox_annotations_4d_corrected/            (Step 4)"
echo ""
echo "  ${AG_ROOT}/wsg_2d_augmentations/                       (Step 5)"
echo "  ${AG_ROOT}/world4d_rel_annotations/${PHASE}/           (Step 6)"
echo "  ${AG_ROOT}/world_annotations/${PHASE}/world_scene_graph/ (Step 7)"
echo "============================================================"
