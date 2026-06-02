#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from dataloader.ag_dataset import StandardAG
from datasets.preprocess.annotations.config_utils import (
    load_config, resolve_common, first, add_config_arg,
)
from datasets.preprocess.annotations.raw.frame_bbox_3D_base import FrameToWorldAnnotationsBase, rerun_frame_vis_final_only


class FrameToWorldAnnotations(FrameToWorldAnnotationsBase):
    pass


# --------------------------------------------------------------------------------------
# Dataset + CLI
# --------------------------------------------------------------------------------------


def load_dataset(ag_root_directory: str):
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    dataloader_train = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0,
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False,
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "World4D GT helper: "
            "(a) inspect 3D bbox annotations, "
            "(b) visualize original Pi3 outputs (points + floor + frames + camera + 3D boxes) "
            "for annotated frames."
        )
    )
    add_config_arg(parser)
    parser.add_argument("--ag_root_directory", type=str, default=None)
    parser.add_argument("--dynamic_scene_dir_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config and resolve CLI > config > default
    config = load_config(args.config)
    frame_cfg = config.get("frame_bbox_3d_gt", {})
    ag_root, dynamic_scene_dir, _ = resolve_common(args, config)
    split = first(args.split, frame_cfg.get("split"), default="04")

    frame_to_world_generator = FrameToWorldAnnotations(
        ag_root_directory=ag_root,
        dynamic_scene_dir_path=dynamic_scene_dir,
    )
    _, _, dataloader_train, dataloader_test = load_dataset(ag_root)

    frame_to_world_generator.generate_gt_world_3D_bb_annotations(dataloader=dataloader_train, split=split)
    frame_to_world_generator.generate_gt_world_3D_bb_annotations(dataloader=dataloader_test, split=split)


def main_sample():
    """
    Simple entry point to visualize original Pi3 point clouds + floor mesh
    + coordinate frames + camera frustum + 3D bounding boxes for a single video.
    Adjust `video_id` as needed.
    """
    args = parse_args()

    # Load config and resolve CLI > config > default
    config = load_config(args.config)
    ag_root, dynamic_scene_dir, _ = resolve_common(args, config)

    frame_to_world_generator = FrameToWorldAnnotations(
        ag_root_directory=ag_root,
        dynamic_scene_dir_path=dynamic_scene_dir,
    )
    video_id = "01KML.mp4"
    # frame_to_world_generator.build_frames_final_and_store(video_id=video_id, overwrite=False)
    frame_to_world_generator.visualize_final_only(video_id=video_id, app_id="World4D-FinalOnly-Sample")


if __name__ == "__main__":
    # main()
    main_sample()