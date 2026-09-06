#!/usr/bin/env python3
"""
Generate Action Genome video splits JSON (video_splits.json).

Splits:
  - train: 7,584 videos
  - test:  1,750 videos
"""

import argparse
import json
import os
from typing import List, Set

from dataloader.ag_dataset import StandardAG


def _extract_video_ids(video_list: List[List[str]]) -> List[str]:
    ids: Set[str] = set()
    for frames in video_list:
        if not frames:
            continue
        video_id = frames[0].split("/")[0]
        ids.add(video_id)
    return sorted(ids)


def generate_splits(ag_root: str = "/data/rohith/ag", out_path: str = "/data/rohith/ag/video_splits.json"):
    print(f"Loading test split from {ag_root}...")
    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    test_video_ids = _extract_video_ids(test_dataset.video_list)

    print(f"Loading train split from {ag_root}...")
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    train_video_ids = _extract_video_ids(train_dataset.video_list)

    out_obj = {
        "train": train_video_ids,
        "test": test_video_ids,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"Successfully generated {out_path} (train={len(train_video_ids)}, test={len(test_video_ids)})")
    return out_obj


def main():
    parser = argparse.ArgumentParser(description="Generate Action Genome video splits JSON")
    parser.add_argument("--ag-root", default="/data/rohith/ag", help="Action Genome dataset root")
    parser.add_argument("--out", default="/data/rohith/ag/video_splits.json", help="Output path for video_splits.json")
    args = parser.parse_args()

    generate_splits(args.ag_root, args.out)


if __name__ == "__main__":
    main()
