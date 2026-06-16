#!/usr/bin/env bash
# ============================================================================
# run_train_annotation_pipeline.sh
# ============================================================================
# Runs the full corrected-floor annotation pipeline for the TRAIN split,
# end to end, regenerating everything (--overwrite).
#
# IMPORTANT — how TRAIN differs from TEST:
#   * Geometry stages (0-4) are identical in spirit to test, just --phase train.
#   * Relationships come from RAG predictions (augment_relationships_train.py),
#     NOT Firebase human corrections. They are written to
#     world_rel_annotations/train/ and keyed by --mode / --model_name.
#   * combine_world4d_relationships_train.py has been patched to read the
#     corrected, phase-separated 4D dir
#     (world_annotations/train/bbox_annotations_4d_corrected), so the manual
#     floor corrections now flow into the train scene graph.
#
# Prerequisites:
#   - Firebase credentials for download_floor_manual_corrections.py
#   - Dynamic scene predictions at DYNAMIC_SCENE_DIR/<video>_10/predictions.npz
#   - Segmentation masks under AG_ROOT/segmentation/
#   - RAG results under RAG_RESULTS_DIR/<mode>/<model_name>/<video>.pkl
#   - Run from the repository root:
#       chmod +x scripts/run_train_annotation_pipeline.sh
#       ./scripts/run_train_annotation_pipeline.sh
#
# All paths / phase also live in configs/annotation_utd.yaml; the CLI flags
# below override the config per-run.
# ============================================================================

set -euo pipefail

# -------------------------------------------------------------------
# Configuration — adjust these paths to match your target machine
# -------------------------------------------------------------------
AG_ROOT="/data/rohith/ag"
DYNAMIC_SCENE_DIR="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic"
MANUAL_CORRECTIONS_DIR="${AG_ROOT}/world_annotations/manual_corrections"

# RAG relationship settings (train uses RAG-predicted missing-object relations).
# augment_relationships_train.py reads <RAG_RESULTS_DIR>/<RAG_MODE>/<RAG_MODEL>/<video>.pkl
RAG_MODE="predcls"          # predcls | sgdet
RAG_MODEL="qwen3vl"
REL_OUTPUT_DIR="${AG_ROOT}/world_rel_annotations"   # augment_train writes <dir>/<phase>/

PHASE="train"
SCRIPT_DIR="datasets/preprocess/annotations"

# Stage 0 (raw prerequisites) is EXPENSIVE and usually already done.
# Set to 1 to (re)generate world_annotations/train/bbox_annotations_3d_obb.
RUN_RAW_PREREQS="${RUN_RAW_PREREQS:-0}"

echo "============================================================"
echo "  Corrected Annotation Pipeline — ${PHASE} split"
echo "============================================================"
echo "  AG_ROOT:              ${AG_ROOT}"
echo "  DYNAMIC_SCENE_DIR:    ${DYNAMIC_SCENE_DIR}"
echo "  MANUAL_CORRECTIONS:   ${MANUAL_CORRECTIONS_DIR}"
echo "  RAG (mode/model):     ${RAG_MODE} / ${RAG_MODEL}"
echo "  PHASE:                ${PHASE}"
echo "  RUN_RAW_PREREQS:      ${RUN_RAW_PREREQS}"
echo "============================================================"
echo ""

# -------------------------------------------------------------------
# Stage 0 (optional): raw prerequisites — gt + world OBBs (auto floor)
# Must run with --phase train so output lands in world_annotations/train/...
# -------------------------------------------------------------------
if [[ "${RUN_RAW_PREREQS}" == "1" ]]; then
    echo "[Step 0a/9] Generating GT annotations (gt_generator.py)..."
    echo "[WARNING] gt_generator.py currently hardcodes BaseAG(phase='test'), so it"
    echo "          emits TEST-split GT only. Train GT must be produced separately"
    echo "          (or the script generalized) before raw train OBBs can be built."
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
# Step 1: Download floor corrections from Firebase (all videos with corrections)
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
# Step 5: Augment TRAIN relationships from RAG predictions
# (Uses the train-specific script; writes world_rel_annotations/<phase>/.)
# -------------------------------------------------------------------
echo "[Step 5/9] Augmenting train relationships (RAG: ${RAG_MODE}/${RAG_MODEL})..."
python ${SCRIPT_DIR}/augment_relationships_train.py \
    --ag_root_directory "${AG_ROOT}" \
    --output_dir "${REL_OUTPUT_DIR}/${PHASE}" \
    --mode "${RAG_MODE}" \
    --model_name "${RAG_MODEL}" \
    --phase "${PHASE}" \
    --overwrite
echo "[Step 5/9] Done."
echo ""

# -------------------------------------------------------------------
# Step 6: Combine corrected 4D bboxes + augmented train relationships
# combine_world4d_relationships_train.py now reads the corrected, phase-separated
# 4D dir (world_annotations/train/bbox_annotations_4d_corrected).
# -------------------------------------------------------------------
echo "[Step 6/9] Combining 4D bboxes with augmented train relationships..."
python ${SCRIPT_DIR}/combine_world4d_relationships_train.py \
    --ag_root_directory "${AG_ROOT}" \
    --phase "${PHASE}" \
    --rag_mode "${RAG_MODE}" \
    --rag_model "${RAG_MODEL}" \
    --overwrite
echo "[Step 6/9] Done."
echo ""

# -------------------------------------------------------------------
# Step 7: Generate final unified world scene graphs
# Reads train relationships from world_rel_annotations/train + corrected 4D bboxes.
# CAVEAT: merge_video() expects per-frame ``observed``/``missing`` relationship
#       lists. The augmentation scripts emit a different schema, so relationship
#       fields may come out empty until the generator's format handling is
#       reconciled. The geometry merge is fine. (Authoritative relationship+
#       geometry merge is Step 6.)
# -------------------------------------------------------------------
echo "[Step 7/9] Generating world scene graphs..."
python ${SCRIPT_DIR}/world_scene_graph_generator.py \
    --ag_root_directory "${AG_ROOT}" \
    --phase "${PHASE}" \
    --augmented_rel_dir "${REL_OUTPUT_DIR}/${PHASE}" \
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
echo "    ├── bbox_annotations_4d_corrected/            (Step 4)"
echo "    └── world_scene_graph/                        (Step 7)"
echo ""
echo "  ${REL_OUTPUT_DIR}/${PHASE}/                     (Step 5, RAG relationships)"
echo "  ${AG_ROOT}/world4d_rel_annotations/${PHASE}/    (Step 6)"
echo "============================================================"
