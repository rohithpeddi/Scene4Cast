#!/usr/bin/env python3
"""
Upload dynamic scenes (/data2/rohith/ag/ag4D/dynamic_scenes) to Box folder 380446756239.

Features:
  - Mode filtering: --mode {train, test, both} to sync dynamic scenes corresponding
    only to the specified phase (determined via world4d_rel_annotations and object_bbox).
  - Split filtering: --split {04, 05, 59, 09, AD, EH, IL, MP, QT, UZ, all} or comma-separated
    combinations (e.g. --split 05,09,AD,EH) matching video ID shards.
  - Large-file streaming: uses boxsdk chunked upload for ~450 MB predictions.npz files.
  - Skips already-uploaded files with matching size.
  - Exponential backoff retry logic for transient Box API errors / rate limits.
  - Multi-threaded worker pool.
  - Dry run mode (--dry-run) to inspect matched counts and volumes before transfer.

Usage:
  # Dry-run for train split AD
  python scripts/upload_dynamic_scenes_box.py --dry-run --mode train --split AD

  # Dry-run for both train and test across 05, 09, AD, EH
  python scripts/upload_dynamic_scenes_box.py --dry-run --mode both --split 05,09,AD,EH

  # Run upload for test set on AD split with 4 workers
  python scripts/upload_dynamic_scenes_box.py --mode test --split AD --workers 4
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.box_sync_common import (
    BoxSyncService,
    STANDARD_SPLITS,
    VideoSplitResolver,
    box_sync_folder_id,
    box_sync_local_root,
    get_box_client,
    human_size,
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload dynamic scenes to Box folder 380446756239"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Filter dynamic scenes by mode: train, test, or both (default: both)",
    )
    parser.add_argument(
        "--sync-mode",
        choices=["upload", "download", "sync"],
        default="upload",
        help="Synchronization direction: upload (default), download, or sync (bidirectional)",
    )
    parser.add_argument(
        "--split",
        default="all",
        help=(
            "Filter by video shard split: 04, 05, 59, 09, AD, EH, IL, MP, QT, UZ, all, "
            "or comma-separated (e.g. 05,09,AD,EH)"
        ),
    )
    parser.add_argument(
        "--local-root",
        default=None,
        help="Local dynamic scenes directory (default: /data2/rohith/ag/ag4D/dynamic_scenes)",
    )
    parser.add_argument(
        "--folder-id",
        default=None,
        help="Box destination folder ID (default: 380446756239)",
    )
    parser.add_argument(
        "--target-subdir",
        default="dynamic_scenes",
        help="Subdirectory name inside Box destination folder (default: dynamic_scenes)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel upload workers (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect matched scenes and show plan without transferring files",
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
        help="Enable verbose output",
    )
    return parser.parse_args()


def parse_split_arg(split_str: str) -> Optional[Set[str]]:
    if not split_str or split_str.lower() in ("all", "both", "*"):
        return None
    splits = set()
    for s in split_str.replace(";", ",").split(","):
        s = s.strip()
        if s:
            splits.add(s)
    return splits or None


def main():
    args = parse_args()
    cfg = load_config(args.config)
    box_folder_id = box_sync_folder_id(cfg, "dynamic_scenes", args.folder_id, fallback="380446756239")
    local_root = box_sync_local_root(cfg, "dynamic_scenes", args.local_root, fallback="/data2/rohith/ag/ag4D/dynamic_scenes")
    splits = parse_split_arg(args.split)

    client = get_box_client(cfg, args.box_cred)
    resolver = VideoSplitResolver()

    # Search for scene directories: inside pi3_dynamic or direct
    search_base = local_root / "pi3_dynamic" if (local_root / "pi3_dynamic").is_dir() else local_root
    if not search_base.is_dir():
        print(f"Error: Directory does not exist: {search_base}", file=sys.stderr)
        sys.exit(1)

    print(f"\n==================================================================")
    print(f" Dynamic Scenes Uploader -> Box Folder [{box_folder_id}]")
    print(f" Local Root   : {local_root}")
    print(f" Search Base  : {search_base}")
    print(f" Target Subdir: {args.target_subdir}")
    print(f" Mode Filter  : {args.mode.upper()}")
    print(f" Split Filter : {', '.join(sorted(splits)) if splits else 'ALL'}")
    print(f" Workers      : {args.workers}")
    print(f" Dry Run      : {args.dry_run}")
    print(f"==================================================================\n")

    print(f"Scanning local scene folders in {search_base}...")
    matched_dirs = []
    breakdown: Dict[Tuple[str, str], int] = {}

    for entry in sorted(search_base.iterdir()):
        if not entry.is_dir():
            continue
        vid = entry.name.split("_")[0]
        v_mode = resolver.get_mode(vid)
        if args.mode != "both" and v_mode != args.mode:
            continue
        if not resolver.matches_split_filter(vid, splits):
            continue
        matched_dirs.append(entry)
        s_canon = resolver.get_canonical_split(vid)
        breakdown[(v_mode, s_canon)] = breakdown.get((v_mode, s_canon), 0) + 1

    print(f"Found {len(matched_dirs):,} matching dynamic scene folder(s).")
    if not matched_dirs:
        print("No scene directories matched criteria. Exiting.")
        return

    # Collect files
    file_entries: List[Tuple[Path, str, int]] = []
    for d in matched_dirs:
        for f in d.iterdir():
            if f.is_file():
                rel = f.relative_to(local_root).as_posix()
                if args.target_subdir:
                    rel = f"{args.target_subdir}/{rel}"
                file_entries.append((f, rel, f.stat().st_size))

    print("\n------------------------- SUMMARY -------------------------")
    print(f"Total matched folders: {len(matched_dirs):,}")
    print("Breakdown (Mode, Split):")
    for (m, s), count in sorted(breakdown.items()):
        print(f"  - Mode={m:<5} | Split={s:<4}: {count:>5} folders")
    print(f"Total files matched  : {len(file_entries):,} ({human_size(sum(s for _, _, s in file_entries))})")
    print("-----------------------------------------------------------\n")

    service = BoxSyncService(
        client=client,
        local_root=local_root,
        box_root_id=box_folder_id,
        mode=args.sync_mode,
        workers=args.workers,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if args.sync_mode in ("upload", "sync"):
        service.upload_file_map(file_entries, target_box_root_id=box_folder_id)

    if args.sync_mode in ("download", "sync"):
        target_sub_id = service.find_box_path(box_folder_id, args.target_subdir) if args.target_subdir else box_folder_id
        if target_sub_id:
            print(f"\nScanning remote Box folder for download: {args.target_subdir or box_folder_id}...")
            box_files = service.get_box_files(target_sub_id, current_rel_prefix=args.target_subdir or "")
            to_download = []
            for rel, (size, fid) in box_files.items():
                parts = Path(rel).parts
                # Check video ID from path
                vid = parts[1] if len(parts) > 1 and parts[0] == args.target_subdir else parts[0]
                v_mode = resolver.get_mode(vid)
                if args.mode != "both" and v_mode != args.mode:
                    continue
                if not resolver.matches_split_filter(vid, splits):
                    continue
                loc_p = local_root / rel
                if not loc_p.exists() or loc_p.stat().st_size != size:
                    to_download.append((rel, size, fid))

            print(f"Remote files matching criteria: {len(to_download):,} to download")
            if not args.dry_run and to_download:
                service.download_files_list(to_download)


if __name__ == "__main__":
    main()
