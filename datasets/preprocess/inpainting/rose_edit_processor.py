import numpy as np
from PIL import Image

import cv2
import os
import torch
import pickle
import matplotlib.pyplot as plt


class RoseEditProcessor:

    def __init__(
            self,
            data_directory,
            sam2_directory,
            monst3r_directory,
            mask_prefix="mask_"
    ):
        self.pipeline = None
        self.data_directory = data_directory
        self.sam2_directory = sam2_directory
        self.monst3r_directory = monst3r_directory
        self.mask_video_directory = os.path.join(self.data_directory, "mask_videos")
        self.sampled_video_directory = os.path.join(self.data_directory, "sampled_videos")
        self.mask_prefix = mask_prefix

        os.makedirs(self.mask_video_directory, exist_ok=True)
        os.makedirs(self.sampled_video_directory, exist_ok=True)

        self.debug = False

        # Load the video_id_frame_id_list.pkl file that contains the list of (video_id, frame_id) tuples.
        video_id_frame_id_list_pkl_file_path = os.path.join(self.data_directory, "4d_video_frame_id_list.pkl")
        if os.path.exists(video_id_frame_id_list_pkl_file_path):
            with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
                self.video_id_frame_id_list = pickle.load(f)

    def get_sam2_masks(self, video_id):
        frame_ids = self.video_id_frame_id_list[video_id]
        frame_id_list = sorted(list(np.unique(frame_ids)))

        sam_video_mask_dir = os.path.join(self.sam2_directory, video_id, "mask")
        frame_files = sorted([f for f in os.listdir(sam_video_mask_dir) if f.endswith('.png')])
        frame_ids = [os.path.splitext(f)[0] for f in frame_files]

        sam_masks_dict = {}
        for frame_file, frame_id in zip(frame_files, frame_ids):
            mask_path = os.path.join(sam_video_mask_dir, frame_file)
            mask_image = Image.open(mask_path).convert("RGB")
            mask_image_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0
            # Change it into binary mask if any pixel value is greater than 0, set it to 1.0 else 0.0
            mask_image_tensor = (mask_image_tensor > 0).float()
            sam_masks_dict[int(frame_id)] = mask_image_tensor

        sam_mask_keys = list(sam_masks_dict.keys())
        sam_mask_keys.sort()
        print(f"[{video_id}] Loaded {len(sam_mask_keys)} SAM2 masks: {sam_mask_keys[:5]} ... {sam_mask_keys[-5:]}")

        sam2_masked_frames = []
        for frame_id in frame_id_list:
            sam_mask = sam_masks_dict[frame_id]
            # Load the original frame
            original_frame_path = os.path.join(self.data_directory, "frames", video_id, f"{frame_id:06d}.png")
            original_frame = Image.open(original_frame_path).convert("RGB")
            original_frame_tensor = torch.from_numpy(np.array(original_frame)).float() / 255.0

            # Ensure the mask is binary
            binary_mask = (sam_mask > 0.5).float()

            # Create the masked image by setting the masked region to white (1.0)
            masked_frame_tensor = original_frame_tensor * (1 - binary_mask) + binary_mask * 1.0
            masked_frame = Image.fromarray((masked_frame_tensor.numpy() * 255).astype(np.uint8))
            sam2_masked_frames.append(masked_frame)

            if self.debug:
                # Show (a) Binary SAM2 mask (b) Original frame (c) Masked frame
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(sam_mask)
                axs[0].set_title(f"SAM2 Mask Frame {frame_id}")
                axs[1].imshow(original_frame)
                axs[1].set_title(f"Original Frame {frame_id}")
                axs[2].imshow(masked_frame)
                axs[2].set_title(f"SAM2 Masked Frame {frame_id}")
                for ax in axs:
                    ax.axis('off')
                plt.show()
        return sam2_masked_frames

    def get_sampled_frames(self, video_id):
        frame_ids = self.video_id_frame_id_list[video_id]
        frame_id_list = sorted(list(np.unique(frame_ids)))

        sampled_frames = []
        for frame_id in frame_id_list:
            # Load the original frame
            original_frame_path = os.path.join(self.data_directory, "frames", video_id, f"{frame_id:06d}.png")
            original_frame = Image.open(original_frame_path).convert("RGB")
            sampled_frames.append(original_frame)
        return sampled_frames

    def store_mask_videos(self, video_id, mask_frames):
        mask_video_path = os.path.join(self.mask_video_directory, video_id)
        # Convert the mask frames which is a list of PIL images to a video and save it
        height, width = mask_frames[0].size[1], mask_frames[0].size[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(mask_video_path, fourcc, 30, (width, height))
        for frame in mask_frames:
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"[{video_id}] Saved mask video to {mask_video_path}")

    def store_sampled_videos(self, video_id, sampled_frames):
        sampled_video_path = os.path.join(self.sampled_video_directory, video_id)
        # Convert the sampled frames which is a list of PIL images to a video and save it
        height, width = sampled_frames[0].size[1], sampled_frames[0].size[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(sampled_video_path, fourcc, 30, (width, height))
        for frame in sampled_frames:
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"[{video_id}] Saved sampled video to {sampled_video_path}")

    def process(self, video_id):
        print(f"Processing video_id: {video_id}")

        # Get SAM2 masks
        sam2_mask_frames = self.get_sam2_masks(video_id)
        # Store the SAM2 masks as a video
        self.store_mask_videos(video_id, sam2_mask_frames)

        # Get sampled original frames
        sampled_frames = self.get_sampled_frames(video_id)
        # Store the sampled original frames as a video
        self.store_sampled_videos(video_id, sampled_frames)


def main():
    sam2_directory = "/data/rohith/ag/ag4D/uni4D/sam2"
    monst3r_directory = "/data/rohith/ag/ag4D/monst3r"
    monst3r_mask_prefix = "dynamic_mask_"
    data_directory = "/data/rohith/ag/"

    video_id_list = ["00T1E.mp4"]

    processor = RoseEditProcessor(
        data_directory=data_directory,
        sam2_directory=sam2_directory,
        monst3r_directory=monst3r_directory,
        mask_prefix=monst3r_mask_prefix
    )

    for video_id in video_id_list:
        processor.process(video_id)


if __name__ == "__main__":
    main()
