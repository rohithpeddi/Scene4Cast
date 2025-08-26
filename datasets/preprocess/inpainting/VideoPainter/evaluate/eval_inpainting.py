import os
import argparse
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from time import time
from typing import Literal
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXI2VDualInpaintAnyLPipeline,
    CogvideoXBranchModel,
    CogVideoXTransformer3DModel,
    FluxFillPipeline
)
from diffusers.utils import export_to_video, load_image, load_video
from openai import OpenAI
from safetensors import safe_open
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from dataset import DavisTestDataset, OurTestDataset
from metrics import (
    MetricsCalculator,
    calculate_i3d_activations,
    calculate_vfid,
    init_i3d_model
)


def _visualize_video(pipe, original_video, video, masks):
    
    original_video = pipe.video_processor.preprocess_video(original_video, height=video.shape[1], width=video.shape[2])
    masks = pipe.masked_video_processor.preprocess_video(masks, height=video.shape[1], width=video.shape[2])
    
    masked_video = original_video * (masks < 0.5)
    
    original_video = pipe.video_processor.postprocess_video(video=original_video, output_type="np")[0]
    masked_video = pipe.video_processor.postprocess_video(video=masked_video, output_type="np")[0]
    
    masks = masks.squeeze(0).squeeze(0).numpy()
    masks = masks[..., np.newaxis].repeat(3, axis=-1)

    video_ = concatenate_images_horizontally(
        [original_video, masked_video, masks, video],
    )
    return video_, original_video, masks

def concatenate_images_horizontally(images_list, output_type="np"):

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

def main_worker(args):
    args.size = (args.width, args.height)
    w, h = args.size    
    assert (args.dataset == 'davis') or args.dataset == 'VPBench', f"{args.dataset} dataset is not supported"

    if args.dataset == 'davis':
        test_dataset = DavisTestDataset(vars(args))
    elif args.dataset == 'VPBench':
        test_dataset = OurTestDataset(vars(args))
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set up metrics calculator
    metrics_calculator = MetricsCalculator(device)
    i3d_model = init_i3d_model('../ckpt/i3d_rgb_imagenet.pt')  

    time_all = []

    project_name = "Eval"
    result_path = os.path.join(args.save_addr, f'{project_name}_{args.dataset}_dilate_{args.dilate_size}')
    if args.replace_gt:
        result_path = os.path.join(args.save_addr, f'{project_name}_{args.dataset}_dilate_{args.dilate_size}_w_replace_gt')
    else:
        result_path = os.path.join(args.save_addr, f'{project_name}_{args.dataset}_dilate_{args.dilate_size}_wo_replace_gt')
    if args.id_adapter_resample_learnable_path is not None:
        result_path = result_path + "_w_id_resample"
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
        
    # Add FID_Score column to DataFrame
    results_df = pd.DataFrame(columns=[
        'Video_Index', 'Total_Videos', 'Video_Name',
        # Global metrics
        'PSNR', 'SSIM', 'LPIPS', 'MSE', 'MAE',
        'CLIP_Score', 'Temporal_Consistency', 'FID_Score',
        # Masked region metrics
        'Masked_PSNR', 'Masked_SSIM', 'Masked_LPIPS',
        'Masked_MSE', 'Masked_MAE',
        'Masked_CLIP_Score', 'Masked_Region_CLIP_Score',
        'Masked_Temporal_Consistency',
        # Caption related
        'Video_Caption', 'Masked_Target_Caption'
    ])
    
    # Initialize global metrics lists
    total_my_psnr = []
    total_my_ssim = []
    total_my_lpips = []
    total_my_mse = []
    total_my_mae = []
    total_clip_score = []
    total_temporal_consistency = []
    
    # Initialize masked region metrics lists
    total_masked_psnr = []
    total_masked_ssim = []
    total_masked_lpips = []
    total_masked_mse = []
    total_masked_mae = []
    total_masked_clip_score = []
    total_masked_region_clip_score = []
    total_masked_temporal_consistency = []

    # Initialize I3D features for FID calculation
    output_i3d_activations = []
    real_i3d_activations = []
      

    branch = CogvideoXBranchModel.from_pretrained(args.inpainting_branch, torch_dtype=torch.bfloat16).to(device)
    if args.id_adapter_resample_learnable_path is None:
        pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
            args.model_path,
            branch=branch,
            torch_dtype=torch.bfloat16,
            # device_map="balanced",
        )
    else:
        print(f"Loading the id pool resample learnable from: {args.id_adapter_resample_learnable_path}")
        # load the transformer
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            id_pool_resample_learnable=True,
        ).to(device)

        pipe = CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
            args.model_path,
            branch=branch,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            # device_map="balanced",
        )
    
        pipe.load_lora_weights(
            args.id_adapter_resample_learnable_path, 
            weight_name="pytorch_lora_weights.safetensors", 
            adapter_name="test_1",
            target_modules=["transformer"]
            )
        # pipe.fuse_lora(lora_scale=1 / lora_rank)

        list_adapters_component_wise = pipe.get_list_adapters()
        print(f"list_adapters_component_wise: {list_adapters_component_wise}")

    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)

    caption_df = None
    if args.caption_path:
        caption_df = pd.read_csv(args.caption_path)
        print(f"Successfully read caption file: {args.caption_path}")
    
    for index, items in enumerate(test_loader):
        torch.cuda.empty_cache()

        # frames [1, T, C, H, W]
        # masks [1, T, 1, H, W]
        # video_name [1]
        # frames_PIL list for T [H, W, C]
        frames, masks, flows_f, flows_b, video_name, frames_PIL, video_PIL, masks_PIL, masked_video_PIL, fps = items
        video_PIL = [Image.fromarray(frame.numpy()[0].astype(np.uint8)).convert('RGB') for frame in video_PIL]
        masks_PIL = [Image.fromarray(mask.numpy()[0].astype(np.uint8)).convert('RGB') for mask in masks_PIL] 
        masked_video_PIL = [Image.fromarray(masked.numpy()[0].astype(np.uint8)).convert('RGB') for masked in masked_video_PIL]
        args.down_sample_fps = fps if args.down_sample_fps == 0 else args.down_sample_fps
        video_PIL, masked_video_PIL, masks_PIL = video_PIL[::int(fps//args.down_sample_fps)], masked_video_PIL[::int(fps//args.down_sample_fps)], masks_PIL[::int(fps//args.down_sample_fps)]
        frames, masks, frames_PIL = frames[:, ::int(fps//args.down_sample_fps)], masks[:, ::int(fps//args.down_sample_fps)], frames_PIL[::int(fps//args.down_sample_fps)]
        if "long" in args.caption_path:
            args.max_video_length = len(video_PIL)
        video_name = video_name[0]
        if args.max_video_length is not None and frames.size(1) > args.max_video_length:
            pre_length = frames.size(1)
            frames = frames[:, :args.max_video_length]
            masks = masks[:, :args.max_video_length]
            frames_PIL = frames_PIL[:args.max_video_length]
            video_PIL = video_PIL[:args.max_video_length]
            masked_video_PIL = masked_video_PIL[:args.max_video_length]
            masks_PIL = masks_PIL[:args.max_video_length]
            print(f'Video length exceeds limit, truncated from {pre_length} to {args.max_video_length} frames')
        print(f'Processing: {video_name}')
        print(f"frames shape: {frames.shape}, {frames.max()}, {frames.min()}")
        print(f"masks shape: {masks.shape}, {masks.max()}, {masks.min()}")
        print(f"frames_PIL shape: {len(frames_PIL)}-{frames_PIL[0].shape}, {frames_PIL[0].max()}, {frames_PIL[0].min()}")
        print(f"video_PIL shape: {np.array(video_PIL).shape}, {np.array(video_PIL).max()}, {np.array(video_PIL).min()}")
        print(f"masks_PIL shape: {np.array(masks_PIL).shape}, {np.array(masks_PIL).max()}, {np.array(masks_PIL).min()}")
        print(f"masked_video_PIL shape: {np.array(masked_video_PIL).shape}, {np.array(masked_video_PIL).max()}, {np.array(masked_video_PIL).min()}")

        video_length = frames.size(1)
        frames, masks = frames.contiguous(), masks.contiguous()

        masked_frames = frames * (1 - masks)
        

        save_frame_path = os.path.join(result_path, video_name)
        save_comp_path = os.path.join(save_frame_path, 'comp_frames')
        if os.path.exists(save_comp_path):
            existing_frames = len([f for f in os.listdir(save_comp_path) if f.endswith('.png')])
            if existing_frames >= args.max_video_length:
                print(f"Skipping {video_name} - {existing_frames} frames already exist (target: {args.max_video_length})")
                
                # Read existing generated frames for metric calculation
                comp_frames = []
                for i in range(args.max_video_length):
                    frame_path = os.path.join(save_comp_path, f'{i:05d}.png')
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        comp_frames.append(frame)
                comp_frames = np.stack(comp_frames)
            else:
                comp_frames = []
        else:
            comp_frames = []

        time_start = time()
        if caption_df is not None:
            if args.dataset == 'davis':
                video_row = caption_df[caption_df['Video_Index'] == index + 1].iloc[0]
                video_inpainting_prompt = video_row['Video_Caption']
                image_inpainting_prompt = video_row['Masked_Target_Caption']
            elif args.dataset == 'VPBench':
                video_row = caption_df[caption_df['path'] == video_name].iloc[0]
                video_inpainting_prompt = video_row['video_inpainting_prompt']
                image_inpainting_prompt = video_row['image_inpainting_prompt']
            print(f"Reading caption-{str(index + 1)}:")
        else:
            video_inpainting_prompt, image_inpainting_prompt = metrics_calculator.video_editing_prompt(masked_frames.squeeze(0)[0], frames.squeeze(0))
            print("Using model to generate caption:")

        torch.cuda.empty_cache()
        print(f"Video Caption: {video_inpainting_prompt}")
        print(f"Masked Target Caption: {image_inpainting_prompt}")

        if len(comp_frames) < args.max_video_length:

            if args.dilate_size > 0:
                print(f"Dilating the mask with size {args.dilate_size}...")
                for i in range(len(masks_PIL)):
                    mask = cv2.dilate(np.array(masks_PIL[i]), np.ones((args.dilate_size, args.dilate_size)))
                    mask = mask.astype(np.uint8)
                    mask = Image.fromarray(mask)
                    masks_PIL[i] = mask

            image = video_PIL[0]
            mask = masks_PIL[0]

            image_array = np.array(image)
            mask_array = np.array(mask)
            foreground_mask = (mask_array == 255)
            masked_image = np.where(foreground_mask, image_array, 0)
            masked_image = Image.fromarray(masked_image.astype(np.uint8))


            pipe_img_inpainting = FluxFillPipeline.from_pretrained(args.img_inpainting_model, torch_dtype=torch.bfloat16).to(device)
            image_inpainting = pipe_img_inpainting(
                prompt=image_inpainting_prompt,
                image=image,
                mask_image=mask,
                height=image.size[1],
                width=image.size[0],
                guidance_scale=30,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(args.seed)
            ).images[0]
            if not os.path.exists(os.path.join(result_path, video_name)):
                os.makedirs(os.path.join(result_path, video_name), exist_ok=True)
            image_inpainting.save(os.path.join(result_path, video_name, "frame_1_flux.png"))
            masked_image.save(os.path.join(result_path, video_name, "frame_1_gt.png"))
            gt_video_first_frame = video_PIL[0]
            video_PIL[0] = image_inpainting
            masked_video_PIL[0] = image_inpainting

            del pipe_img_inpainting
            torch.cuda.empty_cache()

            
            if not args.long_video:
                video_PIL, masked_video_PIL, masks_PIL = video_PIL[:args.max_video_length], masked_video_PIL[:args.max_video_length], masks_PIL[:args.max_video_length]
            
            if len(video_PIL) < args.max_video_length:
                raise NotImplementedError(f"video length is less than {args.max_video_length}, using {len(video_PIL) - len(video_PIL) % 4 + 1} frames...")
                
            if args.first_frame_gt:
                gt_mask_first_frame = masks_PIL[0]
                masks_PIL[0] = Image.fromarray(np.zeros_like(np.array(masks_PIL[0]))).convert("RGB")
            image = masked_video_PIL[0]
            # image.save(output_path.replace(".mp4", f"_first_frame_0.png"))
            inpaint_outputs = pipe(
                prompt=video_inpainting_prompt,
                image=image,
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                use_dynamic_cfg=True,
                guidance_scale=6.0,
                generator=torch.Generator().manual_seed(args.seed),
                video=masked_video_PIL,
                masks=masks_PIL,
                strength=1.0,
                replace_gt=args.replace_gt,
                mask_add=args.mask_add,
                stride= 49,
                prev_clip_weight=0.0,
                id_pool_resample_learnable=True if args.id_adapter_resample_learnable_path is not None else False,
                output_type="np"
            ).frames[0]
            video_generate = inpaint_outputs
            masks_PIL[0] = gt_mask_first_frame
            video_PIL[0] = gt_video_first_frame
        else:
            video_generate = comp_frames / 255.0
        round_video, ori_frames, masks = _visualize_video(pipe, video_PIL[:len(video_generate)], video_generate, masks_PIL[:len(video_generate)])
        export_to_video(round_video, os.path.join(result_path, video_name, "output.mp4"), fps=args.down_sample_fps)
        print(f"video_generate: {type(video_generate)}, {video_generate.shape}, {video_generate[0].max()}, {video_generate[0].min()};")
        comp_frames = video_generate
        ori_frames = ori_frames * 255
        comp_frames = comp_frames * 255
        ori_frames, comp_frames, masks = ori_frames.astype(np.uint8), comp_frames.astype(np.uint8), masks.astype(np.uint8)
        
        if args.replace_gt:
            for i in range(len(comp_frames)):
                comp_frames[i] = comp_frames[i] * masks[i] + ori_frames[i] * (1 - masks[i])
            
        print(f"ori_frames: {type(ori_frames)} {ori_frames.shape}, {ori_frames[0].max()}, {ori_frames[0].min()};")
        print(f"comp_frames: {type(comp_frames)}, {comp_frames.shape}, {comp_frames[0].max()}, {comp_frames[0].min()};")
        print(f"masks: {type(masks)}, {masks.shape}, {masks[0].max()}, {masks[0].min()};")

        time_i = time() - time_start
        time_i = time_i*1.0/video_length
        time_all.append(time_i)

        # calculate metrics
        cur_my_psnr = []
        cur_my_ssim = []
        cur_my_lpips = []
        cur_my_mse = []
        cur_my_mae = []
        cur_clip_score = []
        
        cur_masked_psnr = []
        cur_masked_ssim = []
        cur_masked_lpips = []
        cur_masked_mse = []
        cur_masked_mae = []
        cur_masked_clip_score = []
        cur_masked_region_clip_score = []

        comp_PIL = []
        frames_PIL = []
        for ori, comp, mask in zip(ori_frames, comp_frames, masks):
            my_psnr = metrics_calculator.calculate_psnr(ori, comp)
            my_ssim = metrics_calculator.calculate_ssim(ori, comp)
            my_lpips = metrics_calculator.calculate_lpips(ori, comp)
            my_mse = metrics_calculator.calculate_mse(ori, comp)
            my_mae = metrics_calculator.calculate_mae(ori, comp)
            clip_score = metrics_calculator.calculate_clip_similarity(img=comp, txt=video_inpainting_prompt)

            mask_np = np.array(mask) # [H, W, 1]
            masked_psnr = metrics_calculator.calculate_psnr(ori, comp, mask_gt=1-mask_np)
            masked_ssim = metrics_calculator.calculate_ssim(ori, comp, mask_gt=1-mask_np)
            masked_lpips = metrics_calculator.calculate_lpips(ori, comp, mask_gt=1-mask_np)
            masked_mse = metrics_calculator.calculate_mse(ori, comp, mask_gt=1-mask_np)
            masked_mae = metrics_calculator.calculate_mae(ori, comp, mask_gt=1-mask_np)
            masked_clip_score = metrics_calculator.calculate_clip_similarity(img=comp, txt=video_inpainting_prompt, mask=mask_np)
            masked_region_clip_score = metrics_calculator.calculate_clip_similarity(img=comp, txt=image_inpainting_prompt, mask=mask_np)

            cur_my_psnr.append(my_psnr)
            cur_my_ssim.append(my_ssim)
            cur_my_lpips.append(my_lpips)
            cur_my_mse.append(my_mse)
            cur_my_mae.append(my_mae)
            cur_clip_score.append(clip_score)

            cur_masked_psnr.append(masked_psnr)
            cur_masked_ssim.append(masked_ssim)
            cur_masked_lpips.append(masked_lpips)
            cur_masked_mse.append(masked_mse)
            cur_masked_mae.append(masked_mae)
            cur_masked_clip_score.append(masked_clip_score)
            cur_masked_region_clip_score.append(masked_region_clip_score)

            total_my_psnr.append(my_psnr)
            total_my_ssim.append(my_ssim)
            total_my_lpips.append(my_lpips)
            total_my_mse.append(my_mse)
            total_my_mae.append(my_mae)
            total_clip_score.append(clip_score)

            total_masked_psnr.append(masked_psnr)
            total_masked_ssim.append(masked_ssim)
            total_masked_lpips.append(masked_lpips)
            total_masked_mse.append(masked_mse)
            total_masked_mae.append(masked_mae)
            total_masked_clip_score.append(masked_clip_score)
            total_masked_region_clip_score.append(masked_region_clip_score)

            frames_PIL.append(Image.fromarray(ori.astype(np.uint8)))
            comp_PIL.append(Image.fromarray(comp.astype(np.uint8)))

        # saving i3d activations
        frames_i3d, comp_i3d = calculate_i3d_activations(frames_PIL,
                                                        comp_PIL,
                                                        i3d_model,
                                                        device=device)
        real_i3d_activations.append(frames_i3d)
        output_i3d_activations.append(comp_i3d)

        cur_my_psnr = sum(cur_my_psnr) / len(cur_my_psnr)
        cur_my_ssim = sum(cur_my_ssim) / len(cur_my_ssim)
        cur_my_lpips = sum(cur_my_lpips) / len(cur_my_lpips)
        cur_my_mse = sum(cur_my_mse) / len(cur_my_mse)
        cur_my_mae = sum(cur_my_mae) / len(cur_my_mae)
        cur_clip_score = sum(cur_clip_score) / len(cur_clip_score)

        avg_my_psnr = sum(total_my_psnr) / len(total_my_psnr)
        avg_my_ssim = sum(total_my_ssim) / len(total_my_ssim)
        avg_my_lpips = sum(total_my_lpips) / len(total_my_lpips)
        avg_my_mse = sum(total_my_mse) / len(total_my_mse)
        avg_my_mae = sum(total_my_mae) / len(total_my_mae)
        avg_clip_score = sum(total_clip_score) / len(total_clip_score)

        cur_masked_psnr = sum(cur_masked_psnr) / len(cur_masked_psnr)
        cur_masked_ssim = sum(cur_masked_ssim) / len(cur_masked_ssim)
        cur_masked_lpips = sum(cur_masked_lpips) / len(cur_masked_lpips)
        cur_masked_mse = sum(cur_masked_mse) / len(cur_masked_mse)
        cur_masked_mae = sum(cur_masked_mae) / len(cur_masked_mae)
        cur_masked_clip_score = sum(cur_masked_clip_score) / len(cur_masked_clip_score)
        cur_masked_region_clip_score = sum(cur_masked_region_clip_score) / len(cur_masked_region_clip_score)

        avg_masked_psnr = sum(total_masked_psnr) / len(total_masked_psnr)
        avg_masked_ssim = sum(total_masked_ssim) / len(total_masked_ssim)
        avg_masked_lpips = sum(total_masked_lpips) / len(total_masked_lpips)
        avg_masked_mse = sum(total_masked_mse) / len(total_masked_mse)
        avg_masked_mae = sum(total_masked_mae) / len(total_masked_mae)
        avg_masked_clip_score = sum(total_masked_clip_score) / len(total_masked_clip_score)
        avg_masked_region_clip_score = sum(total_masked_region_clip_score) / len(total_masked_region_clip_score)

        comp_frames = np.stack(comp_frames)
        temporal_consistency = metrics_calculator.calculate_temporal_consistency(comp_frames)
        masked_temporal_consistency = metrics_calculator.calculate_temporal_consistency(comp_frames, masks)
        
        total_temporal_consistency.append(temporal_consistency)
        total_masked_temporal_consistency.append(masked_temporal_consistency)

        avg_time = sum(time_all) / len(time_all)
        print(f'\n[{index+1:3}/{len(test_loader)}] Video Name: {str(video_name):25}')
        
        print('\nGlobal Metrics:')
        print(f'{"Metric":<15} {"Current":<12} {"Average":<12}')
        print('-' * 40)
        print(f'PSNR{":":>11} {cur_my_psnr:.4f}      {avg_my_psnr:.4f}')
        print(f'SSIM{":":>11} {cur_my_ssim:.4f}      {avg_my_ssim:.4f}')
        print(f'LPIPS{":":>10} {cur_my_lpips:.4f}      {avg_my_lpips:.4f}')
        print(f'MSE{":":>12} {cur_my_mse:.4f}      {avg_my_mse:.4f}')
        print(f'MAE{":":>12} {cur_my_mae:.4f}      {avg_my_mae:.4f}')
        print(f'CLIP Score{":":>8} {cur_clip_score:.4f}      {avg_clip_score:.4f}')
        print(f'Temporal Consistency{":":>8} {temporal_consistency:.4f}      {sum(total_temporal_consistency)/len(total_temporal_consistency):.4f}')

        print('\nMasked Region Metrics:')
        print(f'{"Metric":<15} {"Current":<12} {"Average":<12}')
        print('-' * 40)
        print(f'PSNR{":":>11} {cur_masked_psnr:.4f}      {avg_masked_psnr:.4f}')
        print(f'SSIM{":":>11} {cur_masked_ssim:.4f}      {avg_masked_ssim:.4f}')
        print(f'LPIPS{":":>10} {cur_masked_lpips:.4f}      {avg_masked_lpips:.4f}')
        print(f'MSE{":":>12} {cur_masked_mse:.4f}      {avg_masked_mse:.4f}')
        print(f'MAE{":":>12} {cur_masked_mae:.4f}      {avg_masked_mae:.4f}')
        print(f'CLIP Score{":":>8} {cur_masked_clip_score:.4f}      {avg_masked_clip_score:.4f}')
        print(f'Masked Region CLIP Score{":":>8} {cur_masked_region_clip_score:.4f}      {avg_masked_region_clip_score:.4f}')
        print(f'Temporal Consistency{":":>8} {masked_temporal_consistency:.4f}      {sum(total_masked_temporal_consistency)/len(total_masked_temporal_consistency):.4f}')
        
        print(f'\nProcessing Time: {time_i:.4f} seconds/frame | Average Time: {avg_time:.4f} seconds/frame')
        print('=' * 80)

        results_df.loc[len(results_df)] = [
            f'{index+1:3}',
            len(test_loader),
            video_name,
            f'{cur_my_psnr:.4f}',
            f'{cur_my_ssim:.4f}',
            f'{cur_my_lpips:.4f}',
            f'{cur_my_mse:.4f}',
            f'{cur_my_mae:.4f}',
            f'{cur_clip_score:.4f}',
            f'{temporal_consistency:.4f}',
            '',
            f'{cur_masked_psnr:.4f}',
            f'{cur_masked_ssim:.4f}',
            f'{cur_masked_lpips:.4f}',
            f'{cur_masked_mse:.4f}',
            f'{cur_masked_mae:.4f}',
            f'{cur_masked_clip_score:.4f}',
            f'{cur_masked_region_clip_score:.4f}',
            f'{masked_temporal_consistency:.4f}',
            video_inpainting_prompt,
            image_inpainting_prompt
        ]

        # Save results for warping error evaluation
        if args.save_results:
            save_frame_path = os.path.join(result_path, video_name)
            save_comp_path = os.path.join(save_frame_path, 'comp_frames')  # Completed frames
            save_ori_path = os.path.join(save_frame_path, 'ori_frames')    # Original frames
            save_mask_path = os.path.join(save_frame_path, 'masks')        # Masks

            for path in [save_comp_path, save_ori_path, save_mask_path]:
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

            for i, (comp, ori, mask) in enumerate(zip(comp_frames, ori_frames, masks)):
                # Save completed frame
                cv2.imwrite(
                    os.path.join(save_comp_path, f'{i:05d}.png'),
                    cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                
                # Save original frame
                cv2.imwrite(
                    os.path.join(save_ori_path, f'{i:05d}.png'),
                    cv2.cvtColor(ori.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
                
                # Save mask (convert to 0-255 grayscale)
                mask_np = mask  # [1, H, W]
                mask_np = (mask_np * 255).astype(np.uint8)  # Convert to 0-255 range
                cv2.imwrite(
                    os.path.join(save_mask_path, f'{i:05d}.png'),
                    mask_np
                )

    # Calculate global average metrics
    avg_my_psnr = sum(total_my_psnr) / len(total_my_psnr)
    avg_my_ssim = sum(total_my_ssim) / len(total_my_ssim)
    avg_my_lpips = sum(total_my_lpips) / len(total_my_lpips)
    avg_my_mse = sum(total_my_mse) / len(total_my_mse)
    avg_my_mae = sum(total_my_mae) / len(total_my_mae)
    avg_clip_score = sum(total_clip_score) / len(total_clip_score)
    avg_temporal_consistency = sum(total_temporal_consistency) / len(total_temporal_consistency)
    
    # Calculate masked region average metrics
    avg_masked_psnr = sum(total_masked_psnr) / len(total_masked_psnr)
    avg_masked_ssim = sum(total_masked_ssim) / len(total_masked_ssim)
    avg_masked_lpips = sum(total_masked_lpips) / len(total_masked_lpips)
    avg_masked_mse = sum(total_masked_mse) / len(total_masked_mse)
    avg_masked_mae = sum(total_masked_mae) / len(total_masked_mae)
    avg_masked_clip_score = sum(total_masked_clip_score) / len(total_masked_clip_score)
    avg_masked_region_clip_score = sum(total_masked_region_clip_score) / len(total_masked_region_clip_score)
    avg_masked_temporal_consistency = sum(total_masked_temporal_consistency) / len(total_masked_temporal_consistency)
    
    # Calculate FID score
    fid_score = calculate_vfid(real_i3d_activations, output_i3d_activations)

    # Print all metrics
    print('Finish evaluation... Average Frame VFID: '
        f'{fid_score:.3f} | Time: {avg_time:.4f}')
    print(f'Global Metrics:')
    print(f'PSNR: {avg_my_psnr:.4f}')
    print(f'SSIM: {avg_my_ssim:.4f}')
    print(f'LPIPS: {avg_my_lpips:.4f}')
    print(f'MSE: {avg_my_mse:.4f}')
    print(f'MAE: {avg_my_mae:.4f}')
    print(f'CLIP Score: {avg_clip_score:.4f}')
    print(f'Temporal Consistency: {avg_temporal_consistency:.4f}')
    
    print(f'\nMasked Region Metrics:')
    print(f'PSNR: {avg_masked_psnr:.4f}')
    print(f'SSIM: {avg_masked_ssim:.4f}')
    print(f'LPIPS: {avg_masked_lpips:.4f}')
    print(f'MSE: {avg_masked_mse:.4f}')
    print(f'MAE: {avg_masked_mae:.4f}')
    print(f'CLIP Score: {avg_masked_clip_score:.4f}')
    print(f'Masked Region CLIP Score: {avg_masked_region_clip_score:.4f}')
    print(f'Temporal Consistency: {avg_masked_temporal_consistency:.4f}')

    results_df.loc[len(results_df)] = [
        'Average',
        '',
        '',
        f'{avg_my_psnr:.4f}',
        f'{avg_my_ssim:.4f}',
        f'{avg_my_lpips:.4f}',
        f'{avg_my_mse:.4f}',
        f'{avg_my_mae:.4f}',
        f'{avg_clip_score:.4f}',
        f'{avg_temporal_consistency:.4f}',
        f'{fid_score:.4f}',
        f'{avg_masked_psnr:.4f}',
        f'{avg_masked_ssim:.4f}',
        f'{avg_masked_lpips:.4f}',
        f'{avg_masked_mse:.4f}',
        f'{avg_masked_mae:.4f}',
        f'{avg_masked_clip_score:.4f}',
        f'{avg_masked_region_clip_score:.4f}',
        f'{avg_masked_temporal_consistency:.4f}',
        '',
        ''
    ]

    csv_path = os.path.join(result_path, f"{args.dataset}_metrics.csv")
    results_df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--dataset', default='davis', type=str)
    parser.add_argument('--video_root', default='dataset_root', type=str)
    parser.add_argument('--mask_root', default='mask_root', type=str)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--caption_path', type=str, default='', 
                       help='Path to the caption CSV file')
    parser.add_argument('--max_video_length', type=int, default=None,
                       help='Maximum number of frames to process per video')
    parser.add_argument('--save_addr', type=str, default='', 
                       help='Path to save the results')

    # inpainting branch
    parser.add_argument('--inpainting_branch', type=str, default=None,
                       help='Path to the inpainting branch')
    parser.add_argument('--id_adapter_resample_learnable_path', type=str, default=None,
                       help='Path to the id pool resample learnable')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the model')
    parser.add_argument('--lora_rank', type=int, default=1,
                       help='Path to the id pool resample learnable')
    parser.add_argument('--img_inpainting_model', type=str, default=None,
                       help='Path to the id pool resample learnable')
    parser.add_argument('--dilate_size', type=int, default=0,
                       help='Dilate size for the mask')
    parser.add_argument('--down_sample_fps', type=int, default=0,
                       help='Down sample fps for the video')
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
        "--long_video",
        action='store_true',
        help="Enable long_video feature. Default is False.",
    )
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main_worker(args)