#!/usr/bin/env python3
"""
Unified, config-driven and folder-name-based dataset synchronizer for Scene4Cast / WorldSGG.

Supports:
  - Modes: upload (local -> Box), download (Box -> local), sync (bidirectional)
  - Config-driven paths via YAML (--config) for portable execution on any machine or cluster
  - Folder-name-based selection: sync by folder name (e.g. --item active_objects, --item frames_annotated)
    or by numeric index (1 to 11, or all)
  - Worker rate-limiting (--workers 1 by default to respect Box API limits)
  - Chunked streaming upload/download with exponential backoff retries
  - Audit and status checks across Box and local disk (--check)

Supported Datasets / Folder Names:
  1.  annotations                   (Main 2D & 3D Annotations: folder 367501057528)
  2.  world4d_rel_annotations       (WorldWise Training Annotations: file 2449163179187)
  3.  world_annotations             (World 3D Annotations: folder 415504806441)
  4.  scene4cast_data               (Scene4Cast 3D Bounding Box Data: final.zip, camera.zip)
  5.  bbox_annotations_3d_obb_final (Final 3D OBB Bounding Boxes: folder 415505548311)
  6.  frames_annotated              (Annotated Video Frames)
  7.  gt_annotations                (Ground Truth Annotations / gt_annotations_map.pkl)
  8.  active_objects                (Active Objects annotations & sampled videos)
  9.  segmentation                  (Segmentation masks & masked videos: folder 390042982876)
  10. video_splits                  (Video Splits JSON: video_splits.json)
  11. dynamic_scenes                (4D Dynamic Scenes: folder 415500109453)

Usage:
  # Check status across local disk and Box for all datasets
  python scripts/sync_all_datasets_box.py --check

  # Sync by folder name (upload local active_objects to Box)
  python scripts/sync_all_datasets_box.py --item active_objects --mode upload

  # Sync by folder name (dry run preview)
  python scripts/sync_all_datasets_box.py --item bbox_annotations_3d_obb_final --mode upload --dry-run
  python scripts/sync_all_datasets_box.py --item frames_annotated --mode upload --dry-run
  python scripts/sync_all_datasets_box.py --item video_splits.json --mode upload --dry-run

  # Download datasets onto another machine using a config
  python scripts/sync_all_datasets_box.py --config configs/box_sync_template.yaml --item segmentation --mode download
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.box_sync_common import (
    BoxSyncService,
    box_sync_folder_id,
    box_sync_local_root,
    get_box_client,
    get_dataset_file_mapping,
    human_size,
    load_config,
    normalize_item_name,
)
from boxsdk.exception import BoxAPIException

DEFAULT_TARGET_ROOT_FOLDER_ID = "380446756239"  # WorldSGGDataset
DEFAULT_AG_ANNOTATIONS_ID = "367501057528"     # ag_annotations
DEFAULT_WORLD_ANNOTATIONS_ID = "415504806441"  # world_annotations in WorldSGGDataset
DEFAULT_WORLD4D_FILE_ID = "2449163179187"       # world4d_rel_annotations.zip
DEFAULT_SCENE4CAST_FILES = {
    "bbox_annotations_3d_final.zip": "2449163646313",
    "bbox_annotations_3d_obb_camera.zip": "2449160296239",
}
DEFAULT_BBOX_OBB_FINAL_ID = "415505548311"     # bbox_annotations_3d_obb_final
DEFAULT_SEGMENTATION_FOLDER_ID = "390042982876" # segmentation folder in WorldSGGDataset
DEFAULT_VIDEO_SPLITS_FILE_ID = "2218233184203"   # video_splits.json
DEFAULT_DYNAMIC_SCENES_FOLDER_ID = "415500109453"

LOG_DIR = _PROJECT_ROOT / "logs" / "box_sync"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def check_status(cfg: Optional[dict] = None, client=None):
    """Audit the presence and sizes of all datasets on Box and locally."""
    cfg = cfg or load_config()
    client = client or get_box_client(cfg)

    root_id = box_sync_folder_id(cfg, "root", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_base = box_sync_local_root(cfg)

    print(f"\n================ AUDIT: Box Folder {root_id} ================")
    print(f"Local Base Directory: {local_base}")
    target = client.folder(root_id).get()
    print(f"Target Box Folder   : {target.name} ({target.id})\n")

    root_items = {it.name: it for it in target.get_items(limit=1000, fields=["id", "name", "type", "size"])}

    # 1. ag_annotations
    print("------------------------------------------------------------")
    ag_folder_id = box_sync_folder_id(cfg, "annotations", fallback=DEFAULT_AG_ANNOTATIONS_ID)
    ag_local = box_sync_local_root(cfg, "annotations", fallback=str(local_base / "annotations"))
    print(f"Item 1: Main 2D & 3D Annotations (annotations)")
    print(f"  Box Folder ID : {ag_folder_id}")
    print(f"  Local Path    : {ag_local} ({'EXISTS' if ag_local.exists() else 'MISSING'})")
    try:
        ag_f = client.folder(ag_folder_id).get()
        ag_items = {it.name: it for it in ag_f.get_items(limit=100, fields=["id", "name", "type", "size"])}
        expected = [
            "person_bbox.pkl",
            "object_bbox_and_relationship.pkl",
            "object_bbox_and_relationship_filtersmall.pkl",
            "object_classes.txt",
            "relationship_classes.txt",
            "frame_list.txt",
            "monocular3d_bbox_annotations.zip",
            "monocular3d_bbox_annotations_bak.zip",
        ]
        present_count = sum(1 for exp in expected if exp in ag_items)
        print(f"  Box Items     : {present_count}/{len(expected)} core files present on Box")
        if ag_local.exists():
            local_count = sum(1 for exp in expected if (ag_local / exp).exists())
            print(f"  Local Items   : {local_count}/{len(expected)} core files present locally")
    except Exception as e:
        print(f"  Box Error: {e}")

    # 2. world4d_rel_annotations.zip
    print("\n------------------------------------------------------------")
    w4d_local = box_sync_local_root(cfg, "world4d_rel_annotations", fallback=str(local_base / "world4d_rel_annotations.zip"))
    if w4d_local.is_dir():
        w4d_local = w4d_local / "world4d_rel_annotations.zip"
    print(f"Item 2: WorldWise Training Annotations (world4d_rel_annotations.zip)")
    print(f"  Local Path: {w4d_local} ({human_size(w4d_local.stat().st_size) if w4d_local.exists() else 'MISSING'})")
    if "world4d_rel_annotations.zip" in root_items:
        w_file = root_items["world4d_rel_annotations.zip"]
        print(f"  Box File  : PRESENT ({human_size(w_file.size)}) [ID: {w_file.id}]")
    else:
        print("  Box File  : MISSING in target root folder")

    # 3. world_annotations
    print("\n------------------------------------------------------------")
    wa_folder_id = box_sync_folder_id(cfg, "world_annotations", fallback=DEFAULT_WORLD_ANNOTATIONS_ID)
    wa_local = box_sync_local_root(cfg, "world_annotations", fallback=str(local_base / "world_annotations"))
    print(f"Item 3: World 3D Annotations Folder (world_annotations)")
    print(f"  Box Folder ID : {wa_folder_id}")
    print(f"  Local Path    : {wa_local} ({'EXISTS' if wa_local.exists() else 'MISSING'})")
    try:
        wa_f = client.folder(wa_folder_id).get()
        wa_items = {it.name: it for it in wa_f.get_items(limit=100, fields=["id", "name", "type", "size"])}
        has_obb = "bbox_annotations_3d_obb_final" in wa_items
        has_stats = "annotation_stats" in wa_items
        print(f"  Box Content   : bbox_annotations_3d_obb_final/={'YES' if has_obb else 'NO'}, annotation_stats/={'YES' if has_stats else 'NO'}")
    except Exception as e:
        print(f"  Box Error: {e}")

    # 4. Scene4Cast Data
    print("\n------------------------------------------------------------")
    s4c_local = box_sync_local_root(cfg, "scene4cast_data", fallback=str(local_base / "world_annotations"))
    print(f"Item 4: Scene4Cast 3D Bounding Box Data (scene4cast_data)")
    print(f"  Local Path: {s4c_local} ({'EXISTS' if s4c_local.exists() else 'MISSING'})")
    for fn in ["bbox_annotations_3d_final.zip", "bbox_annotations_3d_obb_camera.zip"]:
        in_box = fn in root_items
        box_size = human_size(root_items[fn].size) if in_box else "N/A"
        loc_file = s4c_local / fn if s4c_local.is_dir() else s4c_local
        loc_exists = loc_file.exists()
        loc_size = human_size(loc_file.stat().st_size) if loc_exists else "N/A"
        print(f"  • {fn:35s} Box: {'✓ ' + box_size if in_box else '✗ MISSING'} | Local: {'✓ ' + loc_size if loc_exists else '✗ MISSING'}")

    # 5. bbox_annotations_3d_obb_final
    print("\n------------------------------------------------------------")
    obb_folder_id = box_sync_folder_id(cfg, "bbox_annotations_3d_obb_final", fallback=DEFAULT_BBOX_OBB_FINAL_ID)
    obb_local = box_sync_local_root(cfg, "bbox_annotations_3d_obb_final", fallback=str(local_base / "world_annotations" / "bbox_annotations_3d_obb_final"))
    print(f"Item 5: Final 3D OBB Bounding Boxes (bbox_annotations_3d_obb_final)")
    print(f"  Box Folder ID : {obb_folder_id}")
    print(f"  Local Path    : {obb_local} ({'EXISTS' if obb_local.exists() else 'MISSING'})")
    try:
        obb_f = client.folder(obb_folder_id).get()
        print(f"  Box Folder    : {obb_f.name} ({human_size(obb_f.size)}) [ID: {obb_f.id}]")
    except Exception as e:
        print(f"  Box Error: {e}")

    # 6. frames_annotated
    print("\n------------------------------------------------------------")
    frames_folder_id = box_sync_folder_id(cfg, "frames_annotated", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    frames_local = box_sync_local_root(cfg, "frames_annotated", fallback=str(local_base / "frames_annotated"))
    print(f"Item 6: Annotated Video Frames (frames_annotated)")
    print(f"  Box Target ID : {frames_folder_id}")
    print(f"  Local Path    : {frames_local} ({'EXISTS' if frames_local.exists() else 'MISSING'})")
    in_box_frames = "frames_annotated" in root_items or "frames_annotated.zip" in root_items or "frames_annotated.tar.gz" in root_items
    matching_f = [fn for fn in ["frames_annotated", "frames_annotated.zip", "frames_annotated.tar.gz"] if fn in root_items]
    print(f"  Box Status    : {'PRESENT (' + ', '.join(matching_f) + ')' if in_box_frames else 'Not yet in WorldSGGDataset root'}")

    # 7. gt_annotations
    print("\n------------------------------------------------------------")
    gt_folder_id = box_sync_folder_id(cfg, "gt_annotations", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    gt_local = box_sync_local_root(cfg, "gt_annotations", fallback=str(local_base / "world_annotations" / "gt_annotations_map.pkl"))
    print(f"Item 7: Ground Truth Annotations (gt_annotations)")
    print(f"  Box Target ID : {gt_folder_id}")
    print(f"  Local Target  : {gt_local} ({'EXISTS' if gt_local.exists() else 'MISSING'})")
    in_box_gt = "gt_annotations" in root_items or "gt_annotations_map.pkl" in root_items or "gt_annotations.zip" in root_items
    matching_gt = [fn for fn in ["gt_annotations", "gt_annotations_map.pkl", "gt_annotations.zip"] if fn in root_items]
    print(f"  Box Status    : {'PRESENT (' + ', '.join(matching_gt) + ')' if in_box_gt else 'MISSING in target root'}")

    # 8. active_objects
    print("\n------------------------------------------------------------")
    ao_folder_id = box_sync_folder_id(cfg, "active_objects", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    ao_local = box_sync_local_root(cfg, "active_objects", fallback=str(local_base / "active_objects"))
    print(f"Item 8: Active Objects (active_objects)")
    print(f"  Box Target ID : {ao_folder_id}")
    print(f"  Local Path    : {ao_local} ({'EXISTS' if ao_local.exists() else 'MISSING'})")
    in_box_ao = "active_objects" in root_items or "active_objects.zip" in root_items
    matching_ao = [fn for fn in ["active_objects", "active_objects.zip"] if fn in root_items]
    print(f"  Box Status    : {'PRESENT (' + ', '.join(matching_ao) + ')' if in_box_ao else 'MISSING in target root'}")

    # 9. segmentation
    print("\n------------------------------------------------------------")
    seg_folder_id = box_sync_folder_id(cfg, "segmentation", fallback=DEFAULT_SEGMENTATION_FOLDER_ID)
    seg_local = box_sync_local_root(cfg, "segmentation", fallback=str(local_base / "segmentation"))
    print(f"Item 9: Segmentation Masks & Masked Videos (segmentation)")
    print(f"  Box Folder ID : {seg_folder_id}")
    print(f"  Local Path    : {seg_local} ({'EXISTS' if seg_local.exists() else 'MISSING'})")
    try:
        seg_f = client.folder(seg_folder_id).get()
        print(f"  Box Folder    : {seg_f.name} ({human_size(seg_f.size)}) [ID: {seg_f.id}]")
    except Exception as e:
        print(f"  Box Error: {e}")

    # 10. video_splits.json
    print("\n------------------------------------------------------------")
    vs_folder_id = box_sync_folder_id(cfg, "video_splits", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    vs_local = box_sync_local_root(cfg, "video_splits", fallback=str(local_base / "video_splits.json"))
    if vs_local.is_dir():
        vs_local = vs_local / "video_splits.json"
    print(f"Item 10: Video Splits JSON (video_splits.json)")
    print(f"  Local Path    : {vs_local} ({human_size(vs_local.stat().st_size) if vs_local.exists() else 'MISSING'})")
    in_box_vs = "video_splits.json" in root_items
    print(f"  Box Status    : {'PRESENT in WorldSGGDataset' if in_box_vs else 'MISSING in target root'}")

    # 11. dynamic_scenes
    print("\n------------------------------------------------------------")
    ds_folder_id = box_sync_folder_id(cfg, "dynamic_scenes", fallback=DEFAULT_DYNAMIC_SCENES_FOLDER_ID)
    ds_local = box_sync_local_root(cfg, "dynamic_scenes", fallback="/data2/rohith/ag/ag4D/dynamic_scenes")
    print(f"Item 11: Dynamic Scenes (dynamic_scenes)")
    print(f"  Box Folder ID : {ds_folder_id}")
    print(f"  Local Path    : {ds_local} ({'EXISTS' if ds_local.exists() else 'MISSING'})")
    print("============================================================\n")


# ---------------------------------------------------------------------------
# Individual Dataset Sync Functions
# ---------------------------------------------------------------------------

def sync_item1(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 1: Main 2D & 3D Annotations Folder."""
    client = get_box_client(cfg)
    box_folder_id = box_sync_folder_id(cfg, "annotations", fallback=DEFAULT_AG_ANNOTATIONS_ID)
    local_path = box_sync_local_root(cfg, "annotations", fallback="/data/rohith/ag/annotations")

    print(f"\n[Item 1 / annotations] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box folder ID   : {box_folder_id}")
    print(f"  Workers         : {workers}")

    local_path.mkdir(parents=True, exist_ok=True)
    service = BoxSyncService(
        client=client,
        local_root=local_path,
        box_root_id=box_folder_id,
        mode=mode,
        workers=workers,
        dry_run=dry_run,
        verbose=False,
        zip_threshold=zip_threshold,
    )
    service.sync(target_rel_path="")
    print("[Item 1 / annotations] Finished successfully.")


def sync_item2(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 2: WorldWise Training Annotations (world4d_rel_annotations.zip)."""
    client = get_box_client(cfg)
    root_folder_id = box_sync_folder_id(cfg, "root", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_path = box_sync_local_root(cfg, "world4d_rel_annotations", fallback="/data/rohith/ag/world4d_rel_annotations.zip")
    if local_path.is_dir():
        local_path = local_path / "world4d_rel_annotations.zip"

    print(f"\n[Item 2 / world4d_rel_annotations] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local file path : {local_path}")
    print(f"  Box folder ID   : {root_folder_id}")

    target_folder = client.folder(root_folder_id)
    items = {it.name: it for it in target_folder.get_items(fields=["id", "name", "size"])}

    if mode in ("download", "sync"):
        if "world4d_rel_annotations.zip" in items:
            box_item = items["world4d_rel_annotations.zip"]
            if not local_path.exists() or local_path.stat().st_size != box_item.size:
                print(f"[Item 2] Downloading world4d_rel_annotations.zip ({human_size(box_item.size)}) from Box...")
                if not dry_run:
                    service = BoxSyncService(client=client, local_root=local_path.parent, box_root_id=root_folder_id, workers=workers)
                    service.download_file("world4d_rel_annotations.zip", box_item.id, dest_file_path=local_path)
                    print("[Item 2] Download completed.")
            else:
                print(f"[Item 2] Local file already matches Box ({human_size(box_item.size)}).")
        else:
            print("[Item 2] Notice: world4d_rel_annotations.zip not found on Box.")

    if mode in ("upload", "sync"):
        if local_path.exists():
            file_size = local_path.stat().st_size
            if "world4d_rel_annotations.zip" not in items or items["world4d_rel_annotations.zip"].size != file_size:
                print(f"[Item 2] Uploading {local_path} ({human_size(file_size)}) to Box...")
                if not dry_run:
                    service = BoxSyncService(client=client, local_root=local_path.parent, box_root_id=root_folder_id, workers=workers)
                    service.upload_file("world4d_rel_annotations.zip", file_size, root_folder_id, local_file_path=local_path)
                    print("[Item 2] Upload completed.")
            else:
                print("[Item 2] Box file already up to date with local.")
        else:
            if mode == "upload":
                print(f"[Item 2] Error: Local file does not exist: {local_path}")

    print("[Item 2 / world4d_rel_annotations] Finished.")


def sync_item3(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 3: World 3D Annotations Folder (world_annotations)."""
    client = get_box_client(cfg)
    wa_folder_id = box_sync_folder_id(cfg, "world_annotations", fallback=DEFAULT_WORLD_ANNOTATIONS_ID)
    local_path = box_sync_local_root(cfg, "world_annotations", fallback="/data/rohith/ag/world_annotations")

    print(f"\n[Item 3 / world_annotations] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box folder ID   : {wa_folder_id}")
    print(f"  Workers         : {workers}")

    local_path.mkdir(parents=True, exist_ok=True)
    service = BoxSyncService(
        client=client,
        local_root=local_path,
        box_root_id=wa_folder_id,
        mode=mode,
        workers=workers,
        dry_run=dry_run,
        verbose=False,
        zip_threshold=zip_threshold,
    )
    service.sync(target_rel_path="")
    print("[Item 3 / world_annotations] Finished successfully.")


def sync_item4(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 4: Scene4Cast 3D Bounding Box Data."""
    client = get_box_client(cfg)
    root_folder_id = box_sync_folder_id(cfg, "scene4cast_data", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_path = box_sync_local_root(cfg, "scene4cast_data", fallback="/data/rohith/ag/world_annotations")

    file_mapping = get_dataset_file_mapping(cfg, "scene4cast_data") or DEFAULT_SCENE4CAST_FILES

    print(f"\n[Item 4 / scene4cast_data] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box folder ID   : {root_folder_id}")

    local_path.mkdir(parents=True, exist_ok=True)
    target_folder = client.folder(root_folder_id)
    items = {it.name: it for it in target_folder.get_items(fields=["id", "name", "size"])}

    for filename, fallback_id in file_mapping.items():
        loc_file = local_path / filename
        box_item = items.get(filename)

        if mode in ("download", "sync"):
            if box_item:
                if not loc_file.exists() or loc_file.stat().st_size != box_item.size:
                    print(f"[Item 4] Downloading {filename} ({human_size(box_item.size)}) from Box...")
                    if not dry_run:
                        service = BoxSyncService(client=client, local_root=local_path, box_root_id=root_folder_id, workers=workers)
                        service.download_file(filename, box_item.id, dest_file_path=loc_file)
                        print(f"[Item 4] Downloaded {filename}.")
                else:
                    print(f"[Item 4] Local {filename} already matches Box ({human_size(box_item.size)}).")

        if mode in ("upload", "sync"):
            if loc_file.exists():
                file_size = loc_file.stat().st_size
                if not box_item or box_item.size != file_size:
                    print(f"[Item 4] Uploading {filename} ({human_size(file_size)}) to Box...")
                    if not dry_run:
                        service = BoxSyncService(client=client, local_root=local_path, box_root_id=root_folder_id, workers=workers)
                        service.upload_file(filename, file_size, root_folder_id, local_file_path=loc_file)
                        print(f"[Item 4] Uploaded {filename}.")
                else:
                    print(f"[Item 4] Box {filename} already up to date.")

    print("[Item 4 / scene4cast_data] Finished.")


def sync_item5(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 5: Final 3D OBB Bounding Boxes (bbox_annotations_3d_obb_final)."""
    client = get_box_client(cfg)
    box_folder_id = box_sync_folder_id(cfg, "bbox_annotations_3d_obb_final", fallback=DEFAULT_BBOX_OBB_FINAL_ID)
    local_path = box_sync_local_root(cfg, "bbox_annotations_3d_obb_final", fallback="/data/rohith/ag/world_annotations/bbox_annotations_3d_obb_final")

    print(f"\n[Item 5 / bbox_annotations_3d_obb_final] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box folder ID   : {box_folder_id}")
    print(f"  Workers         : {workers}")

    local_path.mkdir(parents=True, exist_ok=True)
    service = BoxSyncService(
        client=client,
        local_root=local_path,
        box_root_id=box_folder_id,
        mode=mode,
        workers=workers,
        dry_run=dry_run,
        verbose=False,
        zip_threshold=zip_threshold,
    )
    service.sync(target_rel_path="")
    print("[Item 5 / bbox_annotations_3d_obb_final] Finished successfully.")


def sync_item6(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 6: Annotated Video Frames (frames_annotated)."""
    client = get_box_client(cfg)
    root_folder_id = box_sync_folder_id(cfg, "frames_annotated", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_path = box_sync_local_root(cfg, "frames_annotated", fallback="/data/rohith/ag/frames_annotated")

    print(f"\n[Item 6 / frames_annotated] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box root ID     : {root_folder_id}")
    print(f"  Workers         : {workers}")

    local_path.mkdir(parents=True, exist_ok=True)
    service = BoxSyncService(
        client=client,
        local_root=local_path.parent,
        box_root_id=root_folder_id,
        mode=mode,
        workers=workers,
        dry_run=dry_run,
        verbose=False,
        zip_threshold=zip_threshold,
    )
    service.sync(target_rel_path=local_path.name)
    print("[Item 6 / frames_annotated] Finished successfully.")


def sync_item7(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 7: Ground Truth Annotations (gt_annotations / gt_annotations_map.pkl)."""
    client = get_box_client(cfg)
    root_folder_id = box_sync_folder_id(cfg, "gt_annotations", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_target = box_sync_local_root(cfg, "gt_annotations", fallback="/data/rohith/ag/world_annotations/gt_annotations_map.pkl")

    print(f"\n[Item 7 / gt_annotations] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local target    : {local_target}")
    print(f"  Box folder ID   : {root_folder_id}")

    if local_target.is_file():
        # Sync as a single file to Box root
        target_folder = client.folder(root_folder_id)
        items = {it.name: it for it in target_folder.get_items(fields=["id", "name", "size"])}
        filename = local_target.name

        if mode in ("download", "sync") and filename in items:
            b_item = items[filename]
            if not local_target.exists() or local_target.stat().st_size != b_item.size:
                print(f"[Item 7] Downloading {filename} from Box...")
                if not dry_run:
                    service = BoxSyncService(client=client, local_root=local_target.parent, box_root_id=root_folder_id, workers=workers)
                    service.download_file(filename, b_item.id, dest_file_path=local_target)
            else:
                print(f"[Item 7] Local file already matches Box ({human_size(b_item.size)}).")

        if mode in ("upload", "sync") and local_target.exists():
            f_size = local_target.stat().st_size
            if filename not in items or items[filename].size != f_size:
                print(f"[Item 7] Uploading {local_target} ({human_size(f_size)}) to Box...")
                if not dry_run:
                    service = BoxSyncService(client=client, local_root=local_target.parent, box_root_id=root_folder_id, workers=workers)
                    service.upload_file(filename, f_size, root_folder_id, local_file_path=local_target)
                    print("[Item 7] Upload completed.")
            else:
                print("[Item 7] Box file already up to date with local.")
    else:
        # Directory sync
        local_target.mkdir(parents=True, exist_ok=True)
        service = BoxSyncService(
            client=client,
            local_root=local_target.parent,
            box_root_id=root_folder_id,
            mode=mode,
            workers=workers,
            dry_run=dry_run,
            verbose=False,
            zip_threshold=zip_threshold,
        )
        service.sync(target_rel_path=local_target.name)

    print("[Item 7 / gt_annotations] Finished.")


def sync_item8(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 8: Active Objects (active_objects)."""
    client = get_box_client(cfg)
    root_folder_id = box_sync_folder_id(cfg, "active_objects", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_path = box_sync_local_root(cfg, "active_objects", fallback="/data/rohith/ag/active_objects")

    print(f"\n[Item 8 / active_objects] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box root ID     : {root_folder_id}")
    print(f"  Workers         : {workers}")

    local_path.mkdir(parents=True, exist_ok=True)
    service = BoxSyncService(
        client=client,
        local_root=local_path.parent,
        box_root_id=root_folder_id,
        mode=mode,
        workers=workers,
        dry_run=dry_run,
        verbose=False,
        zip_threshold=zip_threshold,
    )
    service.sync(target_rel_path=local_path.name)
    print("[Item 8 / active_objects] Finished successfully.")


def sync_item9(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 9: Segmentation Masks & Masked Videos (segmentation)."""
    client = get_box_client(cfg)
    box_folder_id = box_sync_folder_id(cfg, "segmentation", fallback=DEFAULT_SEGMENTATION_FOLDER_ID)
    local_path = box_sync_local_root(cfg, "segmentation", fallback="/data/rohith/ag/segmentation")

    print(f"\n[Item 9 / segmentation] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box folder ID   : {box_folder_id}")
    print(f"  Workers         : {workers}")

    local_path.mkdir(parents=True, exist_ok=True)
    service = BoxSyncService(
        client=client,
        local_root=local_path,
        box_root_id=box_folder_id,
        mode=mode,
        workers=workers,
        dry_run=dry_run,
        verbose=False,
        zip_threshold=zip_threshold,
    )
    service.sync(target_rel_path="")
    print("[Item 9 / segmentation] Finished successfully.")


def sync_item10(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 10: Video Splits JSON (video_splits.json)."""
    client = get_box_client(cfg)
    root_folder_id = box_sync_folder_id(cfg, "video_splits", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_file = box_sync_local_root(cfg, "video_splits", fallback="/data/rohith/ag/video_splits.json")
    if local_file.is_dir():
        local_file = local_file / "video_splits.json"

    print(f"\n[Item 10 / video_splits] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local file path : {local_file}")
    print(f"  Box folder ID   : {root_folder_id}")

    target_folder = client.folder(root_folder_id)
    items = {it.name: it for it in target_folder.get_items(fields=["id", "name", "size"])}
    filename = "video_splits.json"

    if mode in ("download", "sync"):
        if filename in items:
            b_item = items[filename]
            if not local_file.exists() or local_file.stat().st_size != b_item.size:
                print(f"[Item 10] Downloading {filename} ({human_size(b_item.size)}) from Box...")
                if not dry_run:
                    service = BoxSyncService(client=client, local_root=local_file.parent, box_root_id=root_folder_id, workers=workers)
                    service.download_file(filename, b_item.id, dest_file_path=local_file)
                    print("[Item 10] Download completed.")
            else:
                print(f"[Item 10] Local file already matches Box ({human_size(b_item.size)}).")
        else:
            print(f"[Item 10] Notice: {filename} not found on Box root {root_folder_id}.")

    if mode in ("upload", "sync"):
        if local_file.exists():
            file_size = local_file.stat().st_size
            if filename not in items or items[filename].size != file_size:
                print(f"[Item 10] Uploading {local_file} ({human_size(file_size)}) to Box...")
                if not dry_run:
                    service = BoxSyncService(client=client, local_root=local_file.parent, box_root_id=root_folder_id, workers=workers)
                    service.upload_file(filename, file_size, root_folder_id, local_file_path=local_file)
                    print("[Item 10] Upload completed.")
            else:
                print("[Item 10] Box file already up to date with local.")
        else:
            if mode == "upload":
                print(f"[Item 10] Error: Local file does not exist: {local_file}")

    print("[Item 10 / video_splits] Finished.")


def sync_item11(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync Item 11: Dynamic Scenes (dynamic_scenes)."""
    client = get_box_client(cfg)
    box_folder_id = box_sync_folder_id(cfg, "dynamic_scenes", fallback=DEFAULT_DYNAMIC_SCENES_FOLDER_ID)
    local_path = box_sync_local_root(cfg, "dynamic_scenes", fallback="/data2/rohith/ag/ag4D/dynamic_scenes")

    print(f"\n[Item 11 / dynamic_scenes] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box folder ID   : {box_folder_id}")
    print(f"  Workers         : {workers}")

    # Delegate to specialized dynamic scenes upload logic if available, or direct sync
    script_upload_ds = _PROJECT_ROOT / "scripts" / "upload_dynamic_scenes_box.py"
    if script_upload_ds.exists() and mode == "upload":
        cmd = [
            sys.executable,
            str(script_upload_ds),
            "--mode", "both",
            "--split", "all",
            "--workers", str(workers),
            "--source-dir", str(local_path),
            "--folder-id", str(box_folder_id),
        ]
        if dry_run:
            cmd.append("--dry-run")
        print(f"  Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        service = BoxSyncService(
            client=client,
            local_root=local_path,
            box_root_id=box_folder_id,
            mode=mode,
            workers=workers,
            dry_run=dry_run,
            verbose=False,
            zip_threshold=zip_threshold,
        )
        service.sync(target_rel_path="")

    print("[Item 11 / dynamic_scenes] Finished.")



SYNC_DISPATCH = {
    "1": sync_item1,
    "annotations": sync_item1,
    "2": sync_item2,
    "world4d_rel_annotations": sync_item2,
    "3": sync_item3,
    "world_annotations": sync_item3,
    "4": sync_item4,
    "scene4cast_data": sync_item4,
    "5": sync_item5,
    "bbox_annotations_3d_obb_final": sync_item5,
    "6": sync_item6,
    "frames_annotated": sync_item6,
    "7": sync_item7,
    "gt_annotations": sync_item7,
    "8": sync_item8,
    "active_objects": sync_item8,
    "9": sync_item9,
    "segmentation": sync_item9,
    "10": sync_item10,
    "video_splits": sync_item10,
    "11": sync_item11,
    "dynamic_scenes": sync_item11,
}


def sync_by_folder_or_name(cfg: dict, target_name: str, mode: str = "sync", workers: int = 1, dry_run: bool = False, zip_threshold: int = 30):
    """Sync a dataset component by its folder name, alias, or index."""
    norm = normalize_item_name(target_name)
    if norm in SYNC_DISPATCH:
        SYNC_DISPATCH[norm](cfg, mode=mode, workers=workers, dry_run=dry_run, zip_threshold=zip_threshold)
        return

    # If arbitrary folder name: resolve local and box targets
    client = get_box_client(cfg)
    local_path = box_sync_local_root(cfg, target_name)
    root_folder_id = box_sync_folder_id(cfg, target_name, fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)

    print(f"\n[Custom Folder Sync: '{target_name}'] Starting sync (Mode: {mode.upper()})...")
    print(f"  Local target : {local_path}")
    print(f"  Box folder ID: {root_folder_id}")

    if local_path.is_file():
        service = BoxSyncService(client=client, local_root=local_path.parent, box_root_id=root_folder_id, workers=workers, dry_run=dry_run)
        if mode in ("upload", "sync"):
            service.upload_file(local_path.name, local_path.stat().st_size, root_folder_id, local_file_path=local_path)
    else:
        service = BoxSyncService(
            client=client,
            local_root=local_path.parent,
            box_root_id=root_folder_id,
            mode=mode,
            workers=workers,
            dry_run=dry_run,
            zip_threshold=zip_threshold,
        )
        service.sync(target_rel_path=local_path.name)
    print(f"[Custom Folder Sync: '{target_name}'] Finished.")


def run_parallel_agents(cfg_path: Optional[str], mode: str = "sync", workers: int = 1, items: Optional[List[str]] = None, zip_threshold: int = 30):
    """Spawn multiple agent background processes to sync selected datasets."""
    python_bin = sys.executable
    script_path = str(Path(__file__).resolve())

    items = items or ["1", "2", "3", "4", "5", "8", "9", "10"]
    processes = {}

    print(f"\n[Orchestrator] Spawning {len(items)} sync agents (mode={mode}, workers={workers} each)...")
    for item in items:
        log_file = LOG_DIR / f"agent_item_{item}.log"
        f = open(log_file, "a", buffering=1)
        cmd = [
            python_bin, "-u", script_path,
            "--item", str(item),
            "--mode", mode,
            "--workers", str(workers),
            "--zip-threshold", str(zip_threshold),
        ]
        if cfg_path:
            cmd.extend(["--config", cfg_path])
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, close_fds=True)
        processes[item] = (proc, log_file, f)
        print(f"  • Spawned Agent for '{item}' (PID: {proc.pid}) -> logging to {log_file}")

    print("\n[Orchestrator] All agents spawned in background.")
    return processes


def main():
    parser = argparse.ArgumentParser(
        description="Config-driven and folder-name-based sync for Scene4Cast / WorldSGG datasets with Box"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to YAML configuration file (default: configs/annotation_utd.yaml or fallback)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["upload", "download", "sync"],
        default="sync",
        help="Sync mode: upload, download, or sync (bidirectional). Default: sync",
    )
    parser.add_argument(
        "--item",
        default="all",
        help=(
            "Dataset name, folder name, or item number to sync.\n"
            "Options include: active_objects, bbox_annotations_3d_obb_final, frames_annotated, "
            "gt_annotations, segmentation, video_splits, annotations, dynamic_scenes, 1-11, or all.\n"
            "Default: all"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers per sync process (default: 1)",
    )
    parser.add_argument(
        "--zip-threshold",
        type=int,
        default=30,
        help="If folder contains more than this number of files across all subfolders, zip it and transfer the archive (default: 30, 0 to disable)",
    )
    parser.add_argument(
        "--local-root",
        default=None,
        help="Override base local directory for datasets",
    )
    parser.add_argument(
        "--folder-id",
        default=None,
        help="Override target Box folder ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect transfer plan without transferring files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Audit status of all datasets locally and on Box",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.local_root:
        if "box_sync" not in cfg:
            cfg["box_sync"] = {}
        cfg["box_sync"]["local_root"] = args.local_root
    if args.folder_id:
        if "box_sync" not in cfg:
            cfg["box_sync"] = {}
        cfg["box_sync"]["root_folder_id"] = args.folder_id

    if args.check:
        check_status(cfg)
        return

    if args.item.lower() == "all":
        run_parallel_agents(args.config, mode=args.mode, workers=args.workers, zip_threshold=args.zip_threshold)
    else:
        sync_by_folder_or_name(cfg, args.item, mode=args.mode, workers=args.workers, dry_run=args.dry_run, zip_threshold=args.zip_threshold)



if __name__ == "__main__":
    main()
