#!/usr/bin/env python3
"""
Generate COCO-format GT annotations from the Action Genome dataset.
"""

import argparse
import json
import os
from typing import Any

import numpy as np
import torch

from dataloader.base_ag_dataset import BaseAG
from scripts.box_sync_common import load_config


def to_jsonable(v: Any) -> Any:
    """Convert tensors / numpy arrays to plain Python objects."""
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (str, int, float)) or v is None:
        return v
    return v


def clean_gt_frame_items(frame_items):
    cleaned = []
    for item in frame_items:
        new_item = {}
        for k, v in item.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                new_item[k] = to_jsonable(v)
            elif isinstance(v, (list, tuple)):
                new_item[k] = [to_jsonable(x) for x in v]
            else:
                new_item[k] = v
        cleaned.append(new_item)
    return cleaned


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate COCO-format GT annotations from the AG dataset.",
    )
    parser.add_argument("--ag-root", type=str, default="/data/rohith/ag", help="Action Genome root directory")
    parser.add_argument("--output-dir", type=str, default="/data/rohith/ag/gt_annotations",
                        help="Output dir for GT annotation JSONs")
    parser.add_argument("--phase", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    return parser.parse_args()


def main():
    args = parse_args()
    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)

    dataset = BaseAG(
        phase=args.phase,
        mode="sgdet",
        datasize="large",
        data_path=args.ag_root,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
        enable_coco_gt=True,
    )

    print(f"Dataset loaded with {len(dataset)} videos ({args.phase} phase)")

    for vid_idx in range(len(dataset)):
        frame_names = dataset.video_list[vid_idx]
        gt_video_annotations = dataset.gt_annotations[vid_idx]

        video_id = frame_names[0].split("/")[0]
        video_dir = os.path.join(output_directory, video_id)
        os.makedirs(video_dir, exist_ok=True)

        images_json = []
        annotations_json = []
        ann_id = 1

        for frame_rel in frame_names:
            image_id = len(images_json) + 1
            images_json.append({
                "id": image_id,
                "file_name": frame_rel,
            })

            boxes_xyxy, cat_ids = dataset.parse_gt_for_frame(gt_video_annotations, frame_rel)

            for b, cid in zip(boxes_xyxy, cat_ids):
                x1, y1, x2, y2 = b
                w = float(x2 - x1)
                h = float(y2 - y1)
                area = max(0.0, w) * max(0.0, h)
                annotations_json.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cid),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(area),
                    "iscrowd": 0,
                })
                ann_id += 1

        cleaned_gt_ann = [clean_gt_frame_items(fitems) for fitems in gt_video_annotations]

        with open(os.path.join(video_dir, "images.json"), "w") as f:
            json.dump(images_json, f, indent=2)

        with open(os.path.join(video_dir, "annotations.json"), "w") as f:
            json.dump(annotations_json, f, indent=2)

        with open(os.path.join(video_dir, "gt_annotations.json"), "w") as f:
            json.dump(cleaned_gt_ann, f, indent=2)

        if (vid_idx + 1) % 100 == 0 or vid_idx == len(dataset) - 1:
            print(f"[{vid_idx+1}/{len(dataset)}] processed {video_id}")

    print(f"Done. Saved to {output_directory}")


if __name__ == "__main__":
    main()
