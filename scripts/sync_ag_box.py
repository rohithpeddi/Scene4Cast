#!/usr/bin/env python3
"""
Sync local /data/rohith/ag items with Box folder 380446756239 (WorldSGGDataset).

Features:
  - Sync an entire directory, a subfolder, or a single file with Box folder 380446756239.
  - Modes: upload, download, sync (bidirectional).
  - Skips already up-to-date files (matching size).
  - Chunked streaming upload for large files (> 20 MB).
  - Exponential backoff retries for rate limits (429) and network timeouts.
  - Multi-threaded parallel transfers.
  - Dry run mode (--dry-run).
  - Dedicated --download-annotations shortcut for downloading annotations from Box.

Usage:
  # Upload annotations folder
  python scripts/sync_ag_box.py --source annotations --mode upload

  # Sync a single file
  python scripts/sync_ag_box.py --source annotations/person_bbox.pkl --mode upload

  # Dry-run check for downloading
  python scripts/sync_ag_box.py --source annotations --mode download --dry-run

  # Download all annotations from Box folder 380446756239
  python scripts/sync_ag_box.py --download-annotations
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.box_sync_common import (
    BoxSyncService,
    box_sync_folder_id,
    box_sync_local_root,
    get_box_client,
    load_config,
    normalize_item_name,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync /data/rohith/ag items with Box folder 380446756239 (WorldSGGDataset)"
    )
    parser.add_argument(
        "--source",
        "-s",
        default="annotations",
        help="Folder or file name to sync (e.g. active_objects, bbox_annotations_3d_obb_final, frames_annotated, video_splits.json, segmentation)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["upload", "download", "sync"],
        default="upload",
        help="Sync mode: upload, download, or sync (bidirectional). Default: upload",
    )
    parser.add_argument(
        "--download-annotations",
        action="store_true",
        help="Shortcut to download annotations folder from Box folder 380446756239",
    )
    parser.add_argument(
        "--folder-id",
        default=None,
        help="Box destination folder ID (default: auto-resolved based on folder name)",
    )
    parser.add_argument(
        "--local-root",
        default=None,
        help="Local root directory (default: /data/rohith/ag)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent transfer workers (default: 1)",
    )
    parser.add_argument(
        "--zip-threshold",
        type=int,
        default=30,
        help="If folder contains more than this number of files across all subfolders, zip it and transfer the archive (default: 30, 0 to disable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show transfer plan without transferring any files",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--box-cred",
        default=None,
        help="Path to box_cred.json credentials file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debugging logs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    target_source = args.source
    mode = args.mode
    if args.download_annotations:
        target_source = "annotations"
        mode = "download"

    norm_purpose = normalize_item_name(target_source)
    root_folder_id = (cfg.get("box_sync", {}) or {}).get("root_folder_id", "380446756239")
    box_folder_id = box_sync_folder_id(cfg, norm_purpose, args.folder_id, fallback=root_folder_id)
    resolved_path = box_sync_local_root(cfg, norm_purpose, args.local_root, fallback="/data/rohith/ag")

    if resolved_path.exists():
        if resolved_path.is_file():
            local_root = resolved_path.parent
            rel_sync = resolved_path.name
        elif box_folder_id != root_folder_id:
            local_root = resolved_path
            rel_sync = ""
        else:
            local_root = resolved_path.parent
            rel_sync = resolved_path.name
    else:
        local_root = box_sync_local_root(cfg, "", args.local_root, fallback="/data/rohith/ag")
        rel_sync = target_source

    client = get_box_client(cfg, args.box_cred)
    service = BoxSyncService(
        client=client,
        local_root=local_root,
        box_root_id=box_folder_id,
        mode=mode,
        workers=args.workers,
        dry_run=args.dry_run,
        verbose=args.verbose,
        zip_threshold=args.zip_threshold,
    )

    service.sync(target_rel_path=rel_sync)


if __name__ == "__main__":
    main()
