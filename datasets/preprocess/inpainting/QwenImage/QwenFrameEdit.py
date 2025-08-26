import numpy as np
from PIL import Image

import os
import torch

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
        # 3. Combine the individual masks to get the final mask.

        # 4. In debug mode, visualize the individual masks and the combined mask in a grid.
        # 5. Use the combined mask and overlay it on the original frame to get the masked image.
        # 6. Get the prompt for the corresponding frame id and video id.

        return []

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
    pass


if __name__ == "__main__":
    main()
