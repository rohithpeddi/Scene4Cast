#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy

from tqdm import tqdm

from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"],
        help="Split for the data"
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/data/rohith/ag/",
        help="Path to the data"
    )

    return parser.parse_args()


def prepare_input(
        img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    # for f_id in range(len(pts3ds_self)):
    #     depth = depths_tosave[f_id].cpu().numpy()
    #     conf = conf_self_tosave[f_id].cpu().numpy()
    #     color = colors_tosave[f_id].cpu().numpy()
    #     c2w = cam2world_tosave[f_id].cpu().numpy()
    #     intrins = intrinsics_tosave[f_id].cpu().numpy()
    # np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
    # np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
    # iio.imwrite(
    #     os.path.join(outdir, "color", f"{f_id:06d}.png"),
    #     (color * 255).astype(np.uint8),
    # )
    # np.savez(
    #     os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
    #     pose=c2w,
    #     intrinsics=intrins,
    # )

    return pts3ds_other, colors, conf_other, cam_dict


def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(args, model, img_paths, output_video_pkl):
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    add_path_to_dust3r(args.model_path)

    from src.dust3r.inference import inference, inference_recurrent

    img_mask = [True] * len(img_paths)

    img_size = 512

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=img_size,
        revisit=1,
        update=True,
    )

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference(views, model, device)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame).")

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, args.output_dir, 1, True
    )

    output_dict = {"pts3ds": pts3ds_other, "colors": colors, "cam_dict": cam_dict}
    with open(output_video_pkl, "wb") as f:
        torch.save(output_dict, f)


def get_video_belongs_to_split(video_id):
    """
    Get the split that the video belongs to based on its ID.
    """
    first_letter = video_id[0]
    if first_letter.isdigit() and int(first_letter) < 5:
        return "04"
    elif first_letter.isdigit() and int(first_letter) >= 5:
        return "59"
    elif first_letter in "ABCD":
        return "AD"
    elif first_letter in "EFGH":
        return "EH"
    elif first_letter in "IJKL":
        return "IL"
    elif first_letter in "MNOP":
        return "MP"
    elif first_letter in "QRST":
        return "QT"
    elif first_letter in "UVWXYZ":
        return "UZ"


def main():
    from src.dust3r.model import ARCroco3DStereo

    args = parse_args()
    # For each image in the sequence, check if it exists.
    root_data_directory = args.datapath
    output_directory = os.path.join(root_data_directory, "ag4D", "cut3r")
    os.makedirs(output_directory, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    video_frame_id_list_dict_file_path = os.path.join(root_data_directory, "4d_video_frame_id_list.pkl")
    if not os.path.exists(video_frame_id_list_dict_file_path):
        print(f"Video frame ID list dictionary file not found: {video_frame_id_list_dict_file_path}")
        return

    # Load the video frame ID list dictionary from the pickle file
    import pickle
    with open(video_frame_id_list_dict_file_path, "rb") as f:
        video_frame_id_list = pickle.load(f)

    video_list_directory = os.path.join(root_data_directory, "frames_annotated")
    video_frames_directory = os.path.join(root_data_directory, "frames")
    video_dir_list = os.listdir(video_list_directory)

    for video_name in tqdm(video_dir_list):
        video_dir_path = os.path.join(video_list_directory, video_name)
        if not os.path.isdir(video_dir_path):
            print(f"Skipping {video_dir_path} as it is not a directory.")
            continue

        # Check if the video is already processed, if yest then skip it
        output_video_pkl = os.path.join(output_directory, f"{video_name[:-4]}.pkl")
        if os.path.exists(output_video_pkl):
            print(f"Output already exists for {video_name}. Skipping.")
            continue

        # Skip the video if it is not in the specified split
        video_split = get_video_belongs_to_split(video_name)
        if args.split != video_split:
            print(f"Skipping {video_name} as it does not belong to the specified split.")
            continue

        # Get the video directory path
        # Construct image paths for all the frames in the video_frame_id_list
        img_paths = []
        for frame_id in sorted(video_frame_id_list[video_name]):
            img_path = os.path.join(video_frames_directory, video_name, f"{frame_id:06d}.png")
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            img_paths.append(img_path)

        run_inference(args, model, img_paths, output_video_pkl)


def preprocess_frame_ids():
    datapath = "/data/rohith/ag/"

    filtered_frames_directory = os.path.join(datapath, "filtered_frames")
    annotated_frames_directory = os.path.join(datapath, "frames_annotated")

    # For each video we need to create a map of video_id and frame_id_list combining frames from both directories
    video_frame_id_list = {}
    video_list_directory = os.path.join(filtered_frames_directory)
    video_dir_list = os.listdir(video_list_directory)
    for video_dir in tqdm(video_dir_list):
        video_dir_path = os.path.join(video_list_directory, video_dir)
        if not os.path.isdir(video_dir_path):
            print(f"Skipping {video_dir_path} as it is not a directory.")
            continue

        # Get the list of frame IDs for the current video
        frame_id_list = sorted(os.listdir(video_dir_path))
        # Convert frame_00249.jpg ---> 250
        frame_id_list = [int(frame_id.split("_")[1].split(".")[0]) + 1 for frame_id in frame_id_list]
        video_frame_id_list[video_dir] = frame_id_list

    # Now, we need to combine the frame IDs from both directories
    video_list_directory = os.path.join(annotated_frames_directory)
    video_dir_list = os.listdir(video_list_directory)
    for video_dir in tqdm(video_dir_list):
        video_dir_path = os.path.join(video_list_directory, video_dir)
        if not os.path.isdir(video_dir_path):
            print(f"Skipping {video_dir_path} as it is not a directory.")
            continue

        # Get the list of frame IDs for the current video
        frame_id_list = sorted(os.listdir(video_dir_path))
        # convert 000249.png ---> 249
        frame_id_list = [int(frame_id.split(".")[0]) for frame_id in frame_id_list]
        if video_dir in video_frame_id_list:
            previous_list_size = len(video_frame_id_list[video_dir])
            video_frame_id_list[video_dir].extend(frame_id_list)
            completed_list_size = len(video_frame_id_list[video_dir])
            print(f"Video {video_dir} frame ID list size increased from {previous_list_size} to {completed_list_size}")
        else:
            video_frame_id_list[video_dir] = frame_id_list

    # Store the video_frame_id_list in a pickle file
    import pickle
    with open(os.path.join(datapath, "4d_video_frame_id_list.pkl"), "wb") as f:
        pickle.dump(video_frame_id_list, f)


if __name__ == "__main__":
    main()
    # preprocess_frame_ids()
