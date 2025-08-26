import numpy as np
from PIL import Image

import os
import torch
import pickle
from diffusers import QwenImageEditPipeline


class QwenFrameEdit:
    def __init__(
            self,
            data_directory,
            sam2_directory,
            monst3r_directory,
            output_directory,
            mask_prefix="mask_"
    ):
        self.pipeline = None
        self.data_directory = data_directory
        self.sam2_directory = sam2_directory
        self.monst3r_directory = monst3r_directory
        self.output_directory = output_directory
        self.mask_prefix = mask_prefix
        os.makedirs(self.output_directory, exist_ok=True)

        self.debug = True
        self.load_model()

        # Load the video_id_frame_id_list.pkl file that contains the list of (video_id, frame_id) tuples.
        video_id_frame_id_list_pkl_file_path = os.path.join(self.data_directory, "4d_video_frame_id_list.pkl")
        if os.path.exists(video_id_frame_id_list_pkl_file_path):
            with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
                self.video_id_frame_id_list = pickle.load(f)

    def load_model(self):
        self.pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
        print("pipeline loaded")
        self.pipeline.to(torch.bfloat16)
        self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(disable=None)

    def construct_video_prompt(self, video_id):
        num_frames = os.listdir(os.path.join(self.data_directory, video_id, "frames"))
        prompt = "Fill the masked region of the image with realistic and contextually appropriate content."
        return [prompt] * len(num_frames)

    def construct_masked_frames(self, video_id):
        # 1. Get the SAM2 segmentation mask for each frame
        sam_video_mask_dir = os.path.join(self.sam2_directory, video_id, "mask")
        # Load all the masks from the mask directory and store them in a tensor.
        # Assume the masks are named as 000001.png, 000002.png
        frame_files = sorted(os.listdir(sam_video_mask_dir))
        frame_ids = [os.path.splitext(f)[0] for f in frame_files]

        # Sam frame id is 1-indexed, convert it to 0-indexed.
        sam_masks_dict = {}
        for frame_file, frame_id in zip(frame_files, frame_ids):
            mask_path = os.path.join(sam_video_mask_dir, frame_file)
            mask_image = Image.open(mask_path).convert("RGB")
            mask_image_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0
            sam_masks_dict[int(frame_id) - 1] = mask_image_tensor

        # 2. Get the Monst3R dynamic mask.
        # Monst3R masks are named as dynamic_mask_0.png, dynamic_mask_1.png
        # Monst3R frame id is 0-indexed.
        monst3r_video_mask_dir = os.path.join(self.monst3r_directory, video_id)
        monst3r_mask_files = sorted([f for f in os.listdir(monst3r_video_mask_dir) if f.startswith(self.mask_prefix)])
        monst3r_masks_dict = {}
        for mask_file in monst3r_mask_files:
            frame_id = int(mask_file[len(self.mask_prefix):].split('.')[0])
            mask_path = os.path.join(monst3r_video_mask_dir, mask_file)
            mask_image = Image.open(mask_path).convert("RGB")
            mask_image_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0
            monst3r_masks_dict[frame_id] = mask_image_tensor

        # 3. Combine the individual masks to get the final mask.
        frame_id_list = self.video_id_frame_id_list[video_id]
        combined_masks = []
        for frame_id in frame_id_list:
            sam_mask = sam_masks_dict[frame_id]
            monst3r_mask = monst3r_masks_dict[frame_id]
            combined_mask = torch.clamp(sam_mask + monst3r_mask, 0, 1)
            combined_masks.append(combined_mask)

        # 4. In debug mode, visualize the individual masks and the combined mask in a grid.
        if self.debug:
            import matplotlib.pyplot as plt
            for i, frame_id in enumerate(frame_id_list):
                sam_mask = sam_masks_dict[frame_id]
                monst3r_mask = monst3r_masks_dict[frame_id]
                combined_mask = combined_masks[i]

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(sam_mask)
                axs[0].set_title(f"SAM Mask Frame {frame_id}")
                axs[1].imshow(monst3r_mask)
                axs[1].set_title(f"Monst3R Mask Frame {frame_id}")
                axs[2].imshow(combined_mask)
                axs[2].set_title(f"Combined Mask Frame {frame_id}")
                for ax in axs:
                    ax.axis('off')
                plt.show()

        # 5. Use the combined mask and overlay it on the original frame to get the masked image.
        masked_frames = []
        for i, frame_id in enumerate(frame_id_list):
            original_frame_path = os.path.join(self.data_directory, "frames", video_id, f"{frame_id:06d}.png")
            original_frame = Image.open(original_frame_path).convert("RGB")
            original_frame_tensor = torch.from_numpy(np.array(original_frame)).float() / 255.0

            combined_mask = combined_masks[i]
            # Ensure the mask is binary
            binary_mask = (combined_mask > 0.5).float()

            # Create the masked image by setting the masked region to white (1.0)
            masked_frame_tensor = original_frame_tensor * (1 - binary_mask) + binary_mask * 1.0
            masked_frame = Image.fromarray((masked_frame_tensor.numpy() * 255).astype(np.uint8))
            masked_frames.append(masked_frame)

        return masked_frames

    def edit_video(self, video_id):
        # 1. Construct the masked image that should be filled by the Qwen Image Model
        masked_frames = self.construct_masked_frames(video_id)
        # 2. Construct the prompts for each frame of the video.
        prompts = self.construct_video_prompt(video_id)
        # 3. Use the QwenImageEditPipeline to edit the masked image with the given prompt.
        edited_frames = []
        for masked_frame, prompt in zip(masked_frames, prompts):
            inputs = {
                "image": masked_frame,
                "prompt": prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50,
            }
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                output_image = output.images[0]
                edited_frames.append(output_image)

        # 4. Save the edited frames to the output directory.
        output_video_dir = os.path.join(self.output_directory, video_id)
        os.makedirs(output_video_dir, exist_ok=True)

        for i, edited_frame in enumerate(edited_frames):
            edited_frame.save(os.path.join(output_video_dir, f"{i:06d}.png"))


def main():
    sam2_directory = "/data/rohith/ag/ag4D/uni4D/sam2/00T1E.mp4/mask/"
    monst3r_directory = "/data/rohith/ag/ag4D/monst3r/00T1E.mp4"
    monst3r_mask_prefix = "dynamic_mask_"
    data_directory = "/data/rohith/ag/"
    output_directory = "/data/rohith/ag/inpainting/qwen/"

    video_id_list = ["00T1E.mp4"]

    processor = QwenFrameEdit(
        data_directory=data_directory,
        sam2_directory=sam2_directory,
        monst3r_directory=monst3r_directory,
        output_directory=output_directory,
        mask_prefix=monst3r_mask_prefix
    )

    for video_id in video_id_list:
        processor.edit_video(video_id)


if __name__ == "__main__":
    main()
