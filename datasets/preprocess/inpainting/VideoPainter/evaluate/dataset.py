import os
import json
import random

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from file_client import FileClient
from img_util import imfrombytes
from utils import (Stack, ToTorchFormatTensor)

from diffusers.utils import export_to_video, load_image, load_video

def read_video_with_mask(video_path, masks, mask_id, size=None, start_frame=0, end_frame=49):
    video = load_video(video_path)[start_frame:end_frame]
    mask = masks[start_frame:end_frame]
        
    masked_video = []
    binary_masks = []
    final_video = []
    for frame, frame_mask in tqdm(zip(video, mask), desc="Reading video and masks"):
        frame_array = np.array(frame)
        final_video.append(Image.fromarray(frame_array.astype(np.uint8)).convert("RGB"))

        black_frame = np.zeros_like(frame_array)
        
        binary_mask = (frame_mask == mask_id)
        
        binary_mask_expanded = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)

        masked_frame = np.where(binary_mask_expanded, black_frame, frame_array)
        masked_video.append(Image.fromarray(masked_frame.astype(np.uint8)).convert("RGB"))

        binary_mask_image = np.where(binary_mask_expanded, 255, 0).astype(np.uint8)
        binary_masks.append(Image.fromarray(binary_mask_image).convert('L'))
    
    return final_video, masked_video, binary_masks

class DavisTestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args['size']

        self.video_root = args['video_root']
        self.mask_root = args['mask_root']

        self.video_names = sorted(os.listdir(self.mask_root))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            self.video_dict[v] = v_len
            self.frame_dict[v] = frame_list

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        selected_index = list(range(self.video_dict[video_name]))

        frames = []
        masks = []
        masked_video = []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            frame_path = os.path.join(self.video_root, video_name, frame_list[idx])

            img_bytes = self.file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)

            mask_path = os.path.join(self.mask_root, video_name, str(idx).zfill(5) + '.png')
            mask = Image.open(mask_path).resize(self.size, Image.NEAREST).convert('L')

            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)

            m = cv2.dilate(m,
                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                           iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

            m = np.repeat(m[:, :, np.newaxis], 3, axis=2)
            black_frame = np.zeros_like(np.array(img))
            masked_frame = np.where(m, black_frame, np.array(img))
            masked_video.append(Image.fromarray(masked_frame.astype(np.uint8)).convert("RGB"))

        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)

        video_PIL, masks_PIL, masked_video_PIL = frames, [item.convert('RGB') for item in masks], masked_video
        video_PIL, masks_PIL, masked_video_PIL = [np.array(f).astype(np.uint8) for f in video_PIL], [np.array(f).astype(np.uint8) for f in masks_PIL], [np.array(f).astype(np.uint8) for f in masked_video_PIL]

        fps=8

        return frame_tensors, mask_tensors, 'None', 'None', video_name, frames_PIL, video_PIL, masks_PIL, masked_video_PIL, fps

class OurTestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args['size']

        self.video_root = args['video_root']
        self.mask_root = args['mask_root']
        self.caption_path = args['caption_path']
        self.dataset_df = pd.read_csv(self.caption_path)

        self.video_names = self.dataset_df['path'].values[:]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        meta_data = self.dataset_df[self.dataset_df['path'] == video_name].iloc[0]
        mask_id = int(meta_data['mask_id'])
        fps = int(meta_data['fps'])
        start_frame = int(meta_data['start_frame'])
        end_frame = int(meta_data['end_frame'])

        if ".0.mp4" in video_name:
            video_base_name = video_name.split(".")[0]
            video_path = os.path.join(self.video_root, video_base_name[:-3], f'{video_base_name}.0.mp4')
            mask_frames_path = os.path.join(f"{self.mask_root}/videovo", video_base_name, "all_masks.npz")
        elif ".mp4" in video_name:
            video_base_name = video_name.split(".")[0]
            video_path = os.path.join(self.video_root.replace("videovo", "pexels/pexels"), video_base_name[:9], f'{video_base_name}.mp4')
            mask_frames_path = os.path.join(f"{self.mask_root}/pexels", video_base_name, "all_masks.npz")
        else:
            raise NotImplementedError

        all_masks = np.load(mask_frames_path)["arr_0"]

        frames, masked_video, masks = read_video_with_mask(video_path, masks=all_masks, mask_id=mask_id, size=self.size, start_frame=start_frame, end_frame=end_frame)

        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)

        video_PIL, masks_PIL, masked_video_PIL = frames, [item.convert('RGB') for item in masks], masked_video
        video_PIL, masks_PIL, masked_video_PIL = [np.array(f).astype(np.uint8) for f in video_PIL], [np.array(f).astype(np.uint8) for f in masks_PIL], [np.array(f).astype(np.uint8) for f in masked_video_PIL]

        return frame_tensors, mask_tensors, 'None', 'None', video_name, frames_PIL, video_PIL, masks_PIL, masked_video_PIL, fps