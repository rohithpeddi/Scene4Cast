import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
import argparse
from typing import Literal
import json
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXInpaintPipeline,
    CogVideoXDualInpaintPipeline,
    CogVideoXSelfGuidanceInpaintPipeline,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    CogVideoXSFTInpaintPipeline,
    CogVideoXTransformer3DInpaintModel,
    CogVideoXI2VDualInpaintPipeline,
    CogVideoXI2VDualInpaintAnyLPipeline,
    FluxFillPipeline
)
import cv2
from openai import OpenAI
from diffusers.utils import export_to_video, load_image, load_video
from PIL import Image

vlm_model = OpenAI()

def _visualize_video(pipe, mask_background, original_video, video, masks):
    
    original_video = pipe.video_processor.preprocess_video(original_video, height=video.shape[1], width=video.shape[2])
    masks = pipe.masked_video_processor.preprocess_video(masks, height=video.shape[1], width=video.shape[2])
    
    if mask_background:
        masked_video = original_video * (masks >= 0.5)
    else:
        masked_video = original_video * (masks < 0.5)
    
    original_video = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
    masked_video = pipe.video_processor.postprocess_video(video=masked_video, output_type="np")[0]
    
    masks = masks.squeeze(0).squeeze(0).numpy()
    masks = masks[..., np.newaxis].repeat(3, axis=-1)

    video_ = concatenate_images_horizontally(
        [original_video, masked_video, masks, video],
    )
    return video_

def concatenate_images_horizontally(images_list, output_type="np"):
    '''
    Concatenate three lists of images horizontally.
    Args:
        images_list: List[List[Image.Image]] or List[List[np.ndarray]]
    Returns:
        List[Image.Image] or List[np.ndarray]
    '''
    concatenated_images = []

    length = len(images_list[0])
    for i in range(length):
        tmp_tuple = ()
        for item in images_list:
            tmp_tuple += (np.array(item[i]), )

        # Concatenate arrays horizontally
        concatenated_img = np.concatenate(tmp_tuple, axis=1)

        # Convert back to PIL Image
        if output_type == "pil":
            concatenated_img = Image.fromarray(concatenated_img)
        elif output_type == "np":
            pass
        else:
            raise NotImplementedError
        concatenated_images.append(concatenated_img)
    return concatenated_images

def read_video_with_mask(video_path, masks, mask_id, skip_frames_start=0, skip_frames_end=-1, mask_background=False, fps=0):
    '''
    read the video and masks, and return the video, masked video and binary masks
    Args:
        video_path: str, the path of the video
        masks: np.ndarray, the masks of the video
        mask_id: int, the id of the mask
        skip_frames_start: int, the number of frames to skip at the beginning
        skip_frames_end: int, the number of frames to skip at the end
    Returns:
        video: List[Image.Image], the video (RGB)
        masked_video: List[Image.Image], the masked video (RGB)
        binary_masks: List[Image.Image], the binary masks (RGB)
    '''

    video = load_video(video_path)[skip_frames_start:skip_frames_end]
    mask = masks[skip_frames_start:skip_frames_end]
    # read fps
    if fps == 0:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
    masked_video = []
    binary_masks = []
    for frame, frame_mask in zip(video, mask):
        frame_array = np.array(frame)
        
        black_frame = np.zeros_like(frame_array)
        
        binary_mask = (frame_mask == mask_id)
        
        binary_mask_expanded = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
        
        masked_frame = np.where(binary_mask_expanded, black_frame, frame_array)
        masked_video.append(Image.fromarray(masked_frame.astype(np.uint8)).convert("RGB"))
        
        if mask_background:
            binary_mask_image = np.where(binary_mask, 0, 255).astype(np.uint8)
        else:
            binary_mask_image = np.where(binary_mask, 255, 0).astype(np.uint8)
        binary_masks.append(Image.fromarray(binary_mask_image).convert("RGB"))
    video = [item.convert("RGB") for item in video]
    return video, masked_video, binary_masks, fps

def video_editing_prompt(prompt, instruction, llm_model, masked_image=None, target_img_caption=True):

    if prompt is None or instruction is None:
        raise ValueError("prompt is None or instruction is None")

    system_prompt = """You are a video description editing expert. You need to edit the original video description based on the user's editing instruction.
    Requirements:
    1. Keep the description coherent and natural
    2. Ensure the modification conforms to the user's instruction, strengthen the editing objects corresponding to the editing commands
    3. Retain important details in the original description that are not affected by the instruction
    """
    
    user_prompt = f"""Original video description: {prompt}
    Editing instruction: {instruction}
    Please edit the video description based on the editing instruction. Only return the edited description, no other words."""

    response = vlm_model.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    prompt = response.choices[0].message.content


    if target_img_caption:
        if masked_image is None:
            raise ValueError("masked_image is None when target_img_caption=True")
            
        import base64
        from io import BytesIO
        buffered = BytesIO()
        masked_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        system_prompt ="""You are an expert in visual scene understanding and image editing. Your task is to:
        1. Analyze the visible (unmasked) elements in the provided image
        2. Consider the editing instruction carefully
        3. Generate a precise description of how the edited content should look, assuming the editing instruction is already applied to the original image"""
        
        user_prompt = f"""Based on the visible elements in this image and the editing instruction, generate a concise description for image inpainting. 
        Editing instruction: {instruction}
        Requirements:
        1. Focus ONLY on describing the target edited region (masked area), assuming the editing instruction is already applied to the original image
        2. Maintain visual consistency with unmasked elements
        3. Use specific, descriptive language (colors, textures, shapes)
        4. Avoid mentioning the mask or editing process itself
        5. Keep the description under 20 words
        6. Ensure the description aligns with the editing instruction

        Example:
        Editing instruction: "Change the sky to sunset"
        Good: "Vibrant orange and pink sunset sky with scattered wispy clouds"
        Bad: "A masked region showing sunset colors"

        Only return the description, no explanations or additional text."""

        # Call OpenAI vision API with image
        response = vlm_model.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,  # Add some creativity while maintaining coherence
            max_tokens=50     # Limit response length for conciseness
        )
    else:
        # image inpainting prompt from the new video caption
        system_prompt = "You are an expert in image description. Based on the given video description and editing instruction, please generate a concise description for the first static frame, focusing on the most important visual elements."
        
        user_prompt = f"""Video description (after editing): {prompt}, editing instruction: {instruction}
        Please generate a static description for the first frame. Requirements:
        1. Keep the description concise and precise
        2. Only describe edited visual elements
        3. Avoid using any dynamic or temporal-related words
        4. Must explicitly reflect the visual changes requested in the editing instruction
        5. Emphasize and highlight the edited content as the focus of the description

        Example:
        Editing instruction: "Change the sea water to green"
        Good description: "Green seawater washing against the shore"

        Only return the description, no other words."""
        
        response = vlm_model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
    image_inpainting_prompt = response.choices[0].message.content

    return prompt, image_inpainting_prompt

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"], 
    seed: int = 42,
    # inpainting
    inpainting_mask_meta: str = None,
    inpainting_sample_id: int = None,
    inpainting_branch: str = None,
    inpainting_frames: int = None,
    mask_background: bool = False,
    add_first: bool = False,
    first_frame_gt: bool = False,
    replace_gt: bool = False,
    mask_add: bool = False,
    down_sample_fps: int = 8,
    prev_clip_weight: float = 0.0,
    img_inpainting_model: str = None,
    video_editing_instruction: str = None,
    llm_model: str = None,
    long_video: bool = False,
    dilate_size: int = -1,
    id_pool_resample_learnable_path: str = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    # inpainting
    - inpainting_mask_meta (str): The path of the inpainting mask meta data.
    - inpainting_sample_id (int): The id of the inpainting sample.
    - inpainting_branch (str): The path of the inpainting branch.
    - inpainting_frames (int): The number of frames to generate for inpainting.
    - mask_background (bool): Whether to mask the background.
    - add_first (bool): Whether to add the first frame.
    - first_frame_gt (bool): Whether to use the first frame as the ground truth.
    - replace_gt (bool): Whether to replace the ground truth.
    - mask_add (bool): Whether to add the mask.
    - down_sample_fps (int): The down sample fps.
    """

    image = None
    video = None

    if generate_type == "i2v_inpainting":
        meta_data = pd.read_csv(inpainting_mask_meta).iloc[inpainting_sample_id, :]
        video_base_name = meta_data['path'].split(".")[0]
        if ".0.mp4" in meta_data['path']:
            video_path = os.path.join(image_or_video_path, video_base_name[:-3], f'{video_base_name}.0.mp4')
            mask_frames_path = os.path.join("../data/video_inpainting/videovo", video_base_name, "all_masks.npz")
        elif ".mp4" in meta_data['path']:
            video_path = os.path.join(image_or_video_path.replace("videovo", "pexels/pexels"), video_base_name[:9], f'{video_base_name}.mp4')
            mask_frames_path = os.path.join("../data/video_inpainting/pexels", video_base_name, "all_masks.npz")
        else:
            raise NotImplementedError
        fps = meta_data['fps']
        mask_id = meta_data['mask_id']
        start_frame = meta_data['start_frame']
        end_frame = meta_data['end_frame']
        all_masks = np.load(mask_frames_path)["arr_0"]
        prompt = meta_data['caption']

       
        print("-"*100)
        print(f"video_path: {video_path}; mask_id: {mask_id}; start_frame: {start_frame}; end_frame: {end_frame}; mask_shape: {all_masks.shape}")
        print("-"*100)

        video, masked_video, binary_masks, fps = read_video_with_mask(video_path, skip_frames_start=start_frame, skip_frames_end=end_frame, masks=all_masks, mask_id=mask_id, mask_background=mask_background, fps=fps)

        if inpainting_branch:
            print(f"Using the provided inpainting branch: {inpainting_branch}")
            branch = CogvideoXBranchModel.from_pretrained(inpainting_branch, torch_dtype=dtype).to(dtype=dtype).cuda()
            if id_pool_resample_learnable_path is None:
                pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
                    model_path,
                    branch=branch,
                    torch_dtype=dtype,
                )
            else:
                print(f"Loading the id pool resample learnable from: {id_pool_resample_learnable_path}")
                # load the transformer
                transformer = CogVideoXTransformer3DModel.from_pretrained(
                    model_path,
                    subfolder="transformer",
                    torch_dtype=dtype,
                    id_pool_resample_learnable=True,
                ).to(dtype=dtype).cuda()

                pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
                    model_path,
                    branch=branch,
                    transformer=transformer,
                    torch_dtype=dtype,
                )
            
                pipe.load_lora_weights(
                    id_pool_resample_learnable_path, 
                    weight_name="pytorch_lora_weights.safetensors", 
                    adapter_name="test_1",
                    target_modules=["transformer"]
                    )

                list_adapters_component_wise = pipe.get_list_adapters()
                print(f"list_adapters_component_wise: {list_adapters_component_wise}")
        else:
            print("No inpainting branch provided, using the default branch...")
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=dtype,
            ).to(dtype=dtype).cuda()
            branch = CogvideoXBranchModel.from_transformer(
                transformer=transformer,
                num_layers=1,
                attention_head_dim=transformer.config.attention_head_dim,
                num_attention_heads=transformer.config.num_attention_heads,
                load_weights_from_transformer=True
            ).to(dtype=dtype).cuda()

            pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
                model_path,
                branch=branch,
                transformer=transformer,
                torch_dtype=dtype,
                # device_map="balanced",
            )
        pipe.text_encoder.requires_grad_(False)
        pipe.transformer.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        pipe.branch.requires_grad_(False)

        if img_inpainting_model:
            if dilate_size > 0:
                for i in range(len(binary_masks)):
                    mask = cv2.dilate(np.array(binary_masks[i]), np.ones((dilate_size, dilate_size)))
                    mask = mask.astype(np.uint8)
                    mask = Image.fromarray(mask)
                    binary_masks[i] = mask

            image = video[0]
            mask = binary_masks[0]

            image_array = np.array(image)
            mask_array = np.array(mask)
            foreground_mask = (mask_array == 255)
            masked_image = np.where(foreground_mask, image_array, 0)
            masked_image = Image.fromarray(masked_image.astype(np.uint8))

            original_prompt = prompt
            prompt, image_inpainting_prompt = video_editing_prompt(prompt, video_editing_instruction, llm_model, masked_image=masked_image, target_img_caption=True)

            print("-"*100)
            print(f"original_prompt: {original_prompt}")
            print("-"*100)
            print(f"prompt: {prompt}")
            print("-"*100)
            print(f"image_inpainting_prompt: {image_inpainting_prompt}")
            print("-"*100)
            print(f"video_editing_instruction: {video_editing_instruction}")
            print("-"*100)


            pipe_img_inpainting = FluxFillPipeline.from_pretrained(img_inpainting_model, torch_dtype=torch.bfloat16).to("cuda")
            image_inpainting = pipe_img_inpainting(
                prompt=image_inpainting_prompt,
                image=image,
                mask_image=mask,
                height=image.size[1],
                width=image.size[0],
                guidance_scale=30,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            image.save(os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_flux_i_img.png"))
            mask.save(os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_flux_i_mask.png"))
            image_inpainting.save(os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_flux_o_img.png"))
            masked_image.save(os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path)}_gt_o_img.png"))
            gt_video_first_frame = video[0]
            video[0] = image_inpainting
            masked_video[0] = image_inpainting

            del pipe_img_inpainting
            torch.cuda.empty_cache()

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to("cuda")

    if generate_type == "i2v_inpainting":
        frames = inpainting_frames if inpainting_frames else 49
        down_sample_fps = fps if down_sample_fps == 0 else down_sample_fps
        video, masked_video, binary_masks = video[::int(fps//down_sample_fps)], masked_video[::int(fps//down_sample_fps)], binary_masks[::int(fps//down_sample_fps)]
        
        if not long_video:
            video, masked_video, binary_masks = video[:frames], masked_video[:frames], binary_masks[:frames]
        
        if len(video) < frames:
            raise ValueError(f"video length is less than {frames}, using {len(video) - len(video) % 4 + 1} frames...")
            
        if first_frame_gt:
            gt_mask_first_frame = binary_masks[0]
            if mask_background:
                binary_masks[0] = Image.fromarray(np.ones_like(np.array(binary_masks[0])) * 255).convert("RGB")
            else:
                binary_masks[0] = Image.fromarray(np.zeros_like(np.array(binary_masks[0]))).convert("RGB")
        image = masked_video[0]

        inpaint_outputs = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            video=masked_video,
            masks=binary_masks,
            strength=1.0,
            replace_gt=replace_gt,
            mask_add=mask_add,
            stride= int(frames - down_sample_fps),
            prev_clip_weight=prev_clip_weight,
            id_pool_resample_learnable=True if id_pool_resample_learnable_path is not None else False,
            output_type="np"
        ).frames[0]
        video_generate = inpaint_outputs
        binary_masks[0] = gt_mask_first_frame
        video[0] = gt_video_first_frame
        round_video = _visualize_video(pipe, mask_background, video[:len(video_generate)], video_generate, binary_masks[:len(video_generate)])
        export_to_video(round_video, output_path.replace(".mp4", f"_fps{down_sample_fps}.mp4"), fps=8)

        with open(output_path.replace(".mp4", f".json"), "w") as f:
            prompt_dict = {
                "Original_video_caption": original_prompt,
                "Edited_video_caption": prompt,
                "Edited_image_caption": image_inpainting_prompt,
                "Editing_instruction": video_editing_instruction,
            }
            json.dump(prompt_dict, f, indent=4, ensure_ascii=False)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v', 'inpainting')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--inpainting_branch", type=str, default=None, help="The path of the inpainting branch")
    parser.add_argument("--inpainting_mask_meta", type=str, default=None, help="The path of the inpainting mask meta")
    parser.add_argument("--inpainting_sample_id", type=int, default=None, help="The id of the inpainting sample")
    parser.add_argument("--inpainting_frames", type=int, default=None, help="The number of frames to generate")
    parser.add_argument(
        "--mask_background",
        action='store_true',
        help="Enable mask_background feature. Default is False.",
    )
    parser.add_argument(
        "--add_first",
        action='store_true',
        help="Enable add_first feature. Default is False.",
    )
    parser.add_argument(
        "--first_frame_gt",
        action='store_true',
        help="Enable first_frame_gt feature. Default is False.",
    )
    parser.add_argument(
        "--replace_gt",
        action='store_true',
        help="Enable replace_gt feature. Default is False.",
    )
    parser.add_argument(
        "--mask_add",
        action='store_true',
        help="Enable mask_add feature. Default is False.",
    )
    parser.add_argument(
        "--down_sample_fps",
        type=int,
        default=0,
        help="The down sample fps for the video. Default is 8.",
    )
    parser.add_argument(
        "--prev_clip_weight",
        type=float,
        default=0.0,
        help="The weight for prev_clip. Default is 0.0.",
    )
    parser.add_argument(
        "--img_inpainting_model",
        type=str,
        default=None,
        help="The path of the image inpainting model. Default is None.",
    )
    parser.add_argument(
        "--video_editing_instruction",
        type=str,
        default=None,
        help="The instruction for video editing. Default is None.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="The LLM model for video editing. Default is None.",
    )
    parser.add_argument(
        "--long_video",
        action='store_true',
        help="Enable long_video feature. Default is False.",
    )
    parser.add_argument(
        "--dilate_size",
        type=int,
        default=-1,
        help="The dilate size for the mask. Default is -1.",
    )
    parser.add_argument(
        "--id_pool_resample_learnable_path",
        type=str,
        default=None,
        help="The path of the id_pool_resample_learnable. Default is None.",
    )
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        inpainting_mask_meta=args.inpainting_mask_meta,
        inpainting_sample_id=args.inpainting_sample_id,
        inpainting_branch=args.inpainting_branch,
        inpainting_frames=args.inpainting_frames,
        mask_background=args.mask_background,
        add_first=args.add_first,
        first_frame_gt=args.first_frame_gt,
        replace_gt=args.replace_gt,
        mask_add=args.mask_add,
        down_sample_fps=args.down_sample_fps,
        prev_clip_weight=args.prev_clip_weight,
        img_inpainting_model=args.img_inpainting_model,
        video_editing_instruction=args.video_editing_instruction,
        llm_model=args.llm_model,
        long_video=args.long_video,
        dilate_size=args.dilate_size,
        id_pool_resample_learnable_path=args.id_pool_resample_learnable_path,
    )
