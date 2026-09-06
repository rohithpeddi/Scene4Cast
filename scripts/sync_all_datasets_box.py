#!/usr/bin/env python3
"""
Unified, config-driven dataset synchronizer for Scene4Cast / WorldSGG.

Supports:
  - Modes: upload (local -> Box), download (Box -> local), sync (bidirectional)
  - Config-driven paths via YAML (--config) for portable execution on any machine or cluster
  - Individual dataset selection (--item 1, 2, 3, 4, or all)
  - Worker rate-limiting (--workers 1 by default)
  - Chunked streaming upload/download with exponential backoff retries
  - Audit and status checks (--check)

Datasets:
  1. Main 2D & 3D Annotations (ag_annotations: folder 367501057528)
  2. WorldWise Training Annotations (world4d_rel_annotations.zip: file 2449163179187)
  3. World 3D Annotations (world_annotations: folder 415504806441)
  4. Scene4Cast 3D Bounding Box Data (bbox_annotations_3d_final.zip, bbox_annotations_3d_obb_camera.zip)

Usage:
  # Check status across local disk and Box
  python scripts/sync_all_datasets_box.py --check

  # Download all datasets onto another machine using a config
  python scripts/sync_all_datasets_box.py --config configs/box_sync_template.yaml --mode download

  # Sync a single dataset (e.g. annotations) bidirectionally with 1 worker
  python scripts/sync_all_datasets_box.py --item 1 --mode sync --workers 1

  # Dry run to preview what would be transferred
  python scripts/sync_all_datasets_box.py --mode download --dry-run
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

LOG_DIR = Path("/home/rxp190007/CODE/Scene4Cast/logs/box_sync")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def check_status(cfg: Optional[dict] = None, client=None):
    """Audit the presence and sizes of all 4 items on Box and locally."""
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
    print(f"Item 1: Main 2D & 3D Annotations (ag_annotations)")
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
    print(f"Item 4: Scene4Cast 3D Bounding Box Data")
    print(f"  Local Path: {s4c_local} ({'EXISTS' if s4c_local.exists() else 'MISSING'})")
    for fn in ["bbox_annotations_3d_final.zip", "bbox_annotations_3d_obb_camera.zip"]:
        in_box = fn in root_items
        box_size = human_size(root_items[fn].size) if in_box else "N/A"
        loc_file = s4c_local / fn if s4c_local.is_dir() else s4c_local
        loc_exists = loc_file.exists()
        loc_size = human_size(loc_file.stat().st_size) if loc_exists else "N/A"
        print(f"  • {fn:35s} Box: {'✓ ' + box_size if in_box else '✗ MISSING'} | Local: {'✓ ' + loc_size if loc_exists else '✗ MISSING'}")
    print("============================================================\n")


def sync_item1(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False):
    """Sync Item 1: Main 2D & 3D Annotations Folder."""
    client = get_box_client(cfg)
    box_folder_id = box_sync_folder_id(cfg, "annotations", fallback=DEFAULT_AG_ANNOTATIONS_ID)
    local_path = box_sync_local_root(cfg, "annotations", fallback="/data/rohith/ag/annotations")

    print(f"\n[Item 1 Agent] Starting sync for Annotations (Mode: {mode.upper()})...")
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
    )
    service.sync(target_rel_path="")
    print("[Item 1 Agent] Finished successfully.")


def sync_item2(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False):
    """Sync Item 2: WorldWise Training Annotations (world4d_rel_annotations.zip)."""
    client = get_box_client(cfg)
    root_folder_id = box_sync_folder_id(cfg, "root", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_path = box_sync_local_root(cfg, "world4d_rel_annotations", fallback="/data/rohith/ag/world4d_rel_annotations.zip")
    if local_path.is_dir():
        local_path = local_path / "world4d_rel_annotations.zip"

    print(f"\n[Item 2 Agent] Starting sync for world4d_rel_annotations.zip (Mode: {mode.upper()})...")
    print(f"  Local file path : {local_path}")
    print(f"  Box folder ID   : {root_folder_id}")

    target_folder = client.folder(root_folder_id)
    items = {it.name: it for it in target_folder.get_items(fields=["id", "name", "size"])}

    if mode in ("download", "sync"):
        if "world4d_rel_annotations.zip" in items:
            box_item = items["world4d_rel_annotations.zip"]
            if not local_path.exists() or local_path.stat().st_size != box_item.size:
                print(f"[Item 2 Agent] Downloading world4d_rel_annotations.zip ({human_size(box_item.size)}) from Box...")
                if not dry_run:
                    service = BoxSyncService(client=client, local_root=local_path.parent, box_root_id=root_folder_id, workers=workers)
                    service.download_file("world4d_rel_annotations.zip", box_item.id, dest_file_path=local_path)
                    print("[Item 2 Agent] Download completed.")
            else:
                print(f"[Item 2 Agent] Local file already matches Box ({human_size(box_item.size)}).")
        else:
            print("[Item 2 Agent] Notice: world4d_rel_annotations.zip not found on Box.")

    if mode in ("upload", "sync"):
        if local_path.exists():
            file_size = local_path.stat().st_size
            if "world4d_rel_annotations.zip" not in items or items["world4d_rel_annotations.zip"].size != file_size:
                print(f"[Item 2 Agent] Uploading {local_path} ({human_size(file_size)}) to Box...")
                if not dry_run:
                    service = BoxSyncService(client=client, local_root=local_path.parent, box_root_id=root_folder_id, workers=workers)
                    service.upload_file("world4d_rel_annotations.zip", file_size, root_folder_id, local_file_path=local_path)
                    print("[Item 2 Agent] Upload completed.")
            else:
                print("[Item 2 Agent] Box file already up to date with local.")
        else:
            if mode == "upload":
                print(f"[Item 2 Agent] Error: Local file does not exist: {local_path}")

    print("[Item 2 Agent] Finished.")


def sync_item3(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False):
    """Sync Item 3: World 3D Annotations Folder (world_annotations)."""
    client = get_box_client(cfg)
    wa_folder_id = box_sync_folder_id(cfg, "world_annotations", fallback=DEFAULT_WORLD_ANNOTATIONS_ID)
    local_path = box_sync_local_root(cfg, "world_annotations", fallback="/data/rohith/ag/world_annotations")

    print(f"\n[Item 3 Agent] Starting sync for world_annotations (Mode: {mode.upper()})...")
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
    )
    service.sync(target_rel_path="")
    print("[Item 3 Agent] Finished successfully.")


def sync_item4(cfg: dict, mode: str = "sync", workers: int = 1, dry_run: bool = False):
    """Sync Item 4: Scene4Cast 3D Bounding Box Data."""
    client = get_box_client(cfg)
    root_folder_id = box_sync_folder_id(cfg, "scene4cast_data", fallback=DEFAULT_TARGET_ROOT_FOLDER_ID)
    local_path = box_sync_local_root(cfg, "scene4cast_data", fallback="/data/rohith/ag/world_annotations")

    file_mapping = get_dataset_file_mapping(cfg, "scene4cast_data") or DEFAULT_SCENE4CAST_FILES

    print(f"\n[Item 4 Agent] Starting sync for Scene4Cast Data (Mode: {mode.upper()})...")
    print(f"  Local directory : {local_path}")
    print(f"  Box folder ID   : {root_folder_id}")

    local_path.mkdir(parents=True, exist_ok=True)
    target_folder = client.folder(root_folder_id)
    items = {it.name: it for it in target_folder.get_items(fields=["id", "name", "size"])}

    for filename, fallback_id in file_mapping.items():
        loc_file = local_path / filename
        box_item = items.get(filename)
        file_id = box_item.id if box_item else fallback_id

        if mode in ("download", "sync"):
            if box_item:
                if not loc_file.exists() or loc_file.stat().st_size != box_item.size:
                    print(f"[Item 4 Agent] Downloading {filename} ({human_size(box_item.size)}) from Box...")
                    if not dry_run:
                        service = BoxSyncService(client=client, local_root=local_path, box_root_id=root_folder_id, workers=workers)
                        service.download_file(filename, box_item.id, dest_file_path=loc_file)
                        print(f"[Item 4 Agent] Downloaded {filename}.")
                else:
                    print(f"[Item 4 Agent] Local {filename} already matches Box ({human_size(box_item.size)}).")

        if mode in ("upload", "sync"):
            if loc_file.exists():
                file_size = loc_file.stat().st_size
                if not box_item or box_item.size != file_size:
                    print(f"[Item 4 Agent] Uploading {filename} ({human_size(file_size)}) to Box...")
                    if not dry_run:
                        service = BoxSyncService(client=client, local_root=local_path, box_root_id=root_folder_id, workers=workers)
                        service.upload_file(filename, file_size, root_folder_id, local_file_path=loc_file)
                        print(f"[Item 4 Agent] Uploaded {filename}.")
                else:
                    print(f"[Item 4 Agent] Box {filename} already up to date.")

    print("[Item 4 Agent] Finished.")


def run_parallel_agents(cfg_path: Optional[str], mode: str = "sync", workers: int = 1):
    """Spawn multiple agent background processes to sync each dataset."""
    python_bin = sys.executable
    script_path = str(Path(__file__).resolve())

    items = [1, 2, 3, 4]
    processes = {}

    print(f"\n[Orchestrator] Spawning {len(items)} sync agents (mode={mode}, workers={workers} each)...")
    for item in items:
        log_file = LOG_DIR / f"agent_item_{item}.log"
        f = open(log_file, "a", buffering=1)
        cmd = [python_bin, "-u", script_path, "--item", str(item), "--mode", mode, "--workers", str(workers)]
        if cfg_path:
            cmd.extend(["--config", cfg_path])
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, close_fds=True)
        processes[item] = (proc, log_file, f)
        print(f"  • Spawned Agent {item} (PID: {proc.pid}) -> logging to {log_file}")

    print("\n[Orchestrator] All agents spawned in background.")
    return processes


def main():
    parser = argparse.ArgumentParser(description="Config-driven sync for all Scene4Cast datasets with Box")
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
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Item number to sync (1, 2, 3, 4, or all). Default: all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers per sync process (default: 1)",
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

    if args.item == "1":
        sync_item1(cfg, mode=args.mode, workers=args.workers, dry_run=args.dry_run)
    elif args.item == "2":
        sync_item2(cfg, mode=args.mode, workers=args.workers, dry_run=args.dry_run)
    elif args.item == "3":
        sync_item3(cfg, mode=args.mode, workers=args.workers, dry_run=args.dry_run)
    elif args.item == "4":
        sync_item4(cfg, mode=args.mode, workers=args.workers, dry_run=args.dry_run)
    elif args.item == "all":
        run_parallel_agents(args.config, mode=args.mode, workers=args.workers)


if __name__ == "__main__":
    main()
