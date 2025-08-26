# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import PIL.Image
from PIL import Image, ImageFilter, ImageOps
import torch
from transformers import T5EncoderModel, T5Tokenizer

from ...image_processor import PipelineImageInput
from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders import CogVideoXLoraLoaderMixin
from ...models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from ...models.branch_cogvideox import CogvideoXBranchModel
from ...models.embeddings import get_3d_rotary_pos_embed
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from .pipeline_output import CogVideoXPipelineOutput




logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import CogVideoXPipeline
        >>> from diffusers.utils import export_to_video

        >>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
        >>> prompt = (
        ...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
        ...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
        ...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
        ...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
        ...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
        ...     "atmosphere of this unique musical performance."
        ... )
        >>> video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```
"""


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
        
class CogVideoXI2VDualInpaintAnyLPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        branch ([`CogvideoXBranchModel`]):
            A branch model to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
        branch: CogvideoXBranchModel,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler, branch=branch
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.masked_video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            # logger.warning(
            #     "The following part of your input was truncated because `max_sequence_length` is set to "
            #     f" {max_sequence_length} tokens: {removed_text}"
            # )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None, 
        image: torch.Tensor = None,
        video=None, 
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_video_latents=False,
    ):
        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if len(image.shape) == 4:
            image = image.unsqueeze(2)  # [B, C, F, H, W]

            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
                ]
            else:
                image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]

            image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            image_latents = self.vae.config.scaling_factor * image_latents
        elif len(image.shape) == 5:
            image_latents = image
        else:
            raise ValueError(f"image shape is not valid: {image.shape}")
        padding_shape = (
            batch_size,
            num_frames - 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        if (video is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of video + Noise."
                "However, either the video or the noise timestep has not been provided."
            )
        if return_video_latents or (latents is None and not is_strength_max):
            video = video.to(device=device, dtype=dtype)

            if video.shape[-3] == 16:
                video_latents = video
            else:
                video_latents = self._encode_vae_video(video=video, generator=generator)
            video_latents = video_latents.repeat(batch_size // video_latents.shape[0], 1, 1, 1, 1)
        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(video_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents, image_latents)

        if return_noise:
            outputs += (noise,)

        if return_video_latents:
            outputs += (video_latents,)

        return outputs

    def _encode_vae_video(self, video: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            video_latents = [
                retrieve_latents(self.vae.encode(video[i : i + 1]), generator=generator[i])
                for i in range(video.shape[0])
            ]
            video_latents = torch.cat(video_latents, dim=0)
        else:
            video_latents = retrieve_latents(self.vae.encode(video), generator=generator)

        video_latents = self.vae.config.scaling_factor * video_latents

        return video_latents.permute(0, 2, 1, 3, 4)

    def prepare_mask_latents(
        self, mask, masked_video, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=((mask.shape[-3] - 1) // self.vae_scale_factor_temporal + 1, height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_video = masked_video.to(device=device, dtype=dtype)

        if masked_video.shape[1] == 4:
            masked_video_latents = masked_video
        else:
            masked_video_latents = self._encode_vae_video(masked_video, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1, 1)
        if masked_video_latents.shape[0] < batch_size:
            if not batch_size % masked_video_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_video_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_video_latents = masked_video_latents.repeat(batch_size // masked_video_latents.shape[0], 1, 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_video_latents = (
            torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_video_latents = masked_video_latents.to(device=device, dtype=dtype)
        return mask, masked_video_latents
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        return frames
    
    # Copied from diffusers.pipelines.animatediff.pipeline_animatediff_video2video.AnimateDiffVideoToVideoPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, timesteps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        image,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        video: List[PIL.Image.Image] = None,
        masks: List[PIL.Image.Image] = None,
        masked_video_latents: torch.Tensor = None,
        strength: float = 1.0,
        control_mode: Optional[Union[int, List[int]]] = None,
        conditioning_scale: Union[float, List[float]] = 1.0,
        mask_background: Optional[bool] = False,
        add_first: Optional[bool] = False,
        wo_text: Optional[bool] = False,
        id_pool_resample_learnable: Optional[bool] = False,
        mask_add: Optional[bool] = False,
        replace_gt:Optional[bool] = False,
        stride: int = 24,
        prev_clip_weight: Optional[float] = 0.0,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(video, (list, tuple)):
            total_frames = len(video)
        else:
            total_frames = video.shape[1]
        
        n_windows = (total_frames - num_frames) // stride + 1

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        # 1. Check inputs
        self.check_inputs(
            image,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, timesteps=timesteps, strength=strength, device=device
        )
        self._num_timesteps = len(timesteps)
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        
        if stride < num_frames:
            num_frame_latents = ((num_frames - 1) // self.vae_scale_factor_temporal + 1) * n_windows - (n_windows - 1) * ((num_frames - stride) // self.vae_scale_factor_temporal + 1)
        elif stride == num_frames:
            num_frame_latents = ((num_frames - 1) // self.vae_scale_factor_temporal) * n_windows + 1
        else:
            raise ValueError(f"stride: {stride}, num_frames: {num_frames}")
        latent_channels = self.transformer.config.in_channels // 2
        frame_counts = torch.zeros(num_frame_latents, device=device)
        frame_accumulator = torch.zeros(
            (batch_size, num_frame_latents, latent_channels,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial),
            device=device,
            dtype=prompt_embeds.dtype
        )

        prev_window_states_dict = {}
        for window_idx in range(n_windows):
            start_frame = window_idx * stride
            end_frame = start_frame + num_frames

            if isinstance(video, (list, tuple)):
                window_video = video[start_frame:end_frame]
            else:
                window_video = video[:, start_frame:end_frame]

            if isinstance(masks, (list, tuple)):
                window_mask = masks[start_frame:end_frame]
            else:
                window_mask = masks[:, start_frame:end_frame]

            # 5. Prepare latents
            window_video = self.video_processor.preprocess_video(window_video, height=height, width=width)
            window_video = window_video.to(dtype=torch.float32)
            if window_idx == 0:
                image_ = self.video_processor.preprocess(image, height=height, width=width).to(
                        device, dtype=prompt_embeds.dtype
                    )
            else:
                if -int((num_frames - stride) // self.vae_scale_factor_temporal) < 0:
                    image_ = latents[:, -int((num_frames - stride) // self.vae_scale_factor_temporal) - 1:-int((num_frames - stride) // self.vae_scale_factor_temporal) + 1 - 1]
                    # print(f"image was set by the frame {-int((num_frames - stride) // self.vae_scale_factor_temporal) - 1} of the previous window latents, num_frames: {num_frames}, stride: {stride}")
                else:
                    image_ = latents[:, -1:]
                    # print(f"image was set by the last frame of the previous window latents")
            latents_outputs = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                latent_channels,
                num_frames,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents=None, # must be None for all windows
                video=window_video,
                image=image_,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
                return_noise=True,
                return_video_latents=latent_channels == 16,
            )

            if latent_channels == 16:
                latents, image_latents, noise, video_latents = latents_outputs
            else:
                latents, image_latents, noise = latents_outputs

            # 7. Prepare mask latents
            mask_condition = self.masked_video_processor.preprocess_video(window_mask, height=height, width=width)
            if masked_video_latents is not None:
                masked_video_latents = None
            if masked_video_latents is None:
                if mask_background:
                    masked_video = window_video * (mask_condition >= 0.5)
                else:
                    masked_video = window_video * (mask_condition < 0.5)
            else:
                masked_video = masked_video_latents[:, start_frame:end_frame]

            mask, masked_video_latents = self.prepare_mask_latents(
                mask_condition,
                masked_video,
                batch_size * num_videos_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )
            mask = mask.permute(0, 2, 1, 3, 4).repeat(1, 1, latent_channels, 1, 1)

            # 8. Prepare extra step kwargs
            image_rotary_emb = (
                self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
                if self.transformer.config.use_rotary_positional_embeddings
                else None
            )

            # 9. Denoising loop
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            old_pred_original_sample = None

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    latent_video_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_video_input = self.scheduler.scale_model_input(latent_video_input, t)
                    
                    latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                    latent_model_input = torch.cat([latent_video_input, latent_image_input], dim=2)
                    timestep = t.expand(latent_model_input.shape[0])

                    # branch block
                    latent_branch_input = torch.cat([masked_video_latents, mask[:, :, :1, :, :]], dim=-3)
                    
                    branch_block_samples = self.branch(
                        hidden_states=latent_video_input,
                        encoder_hidden_states=prompt_embeds,
                        branch_cond=latent_branch_input,
                        branch_mode=control_mode,
                        conditioning_scale=conditioning_scale,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        wo_text=wo_text,
                        return_dict=False,
                    )[0]

                    branch_block_samples = [block_sample.to(dtype=prompt_embeds.dtype) for block_sample in branch_block_samples]

                    current_attention_kwargs = attention_kwargs.copy() if attention_kwargs else {}
                    if window_idx > 0:
                        current_attention_kwargs["prev_hidden_states"] = prev_window_states_dict[0]
                        current_attention_kwargs["prev_clip_weight"] = prev_clip_weight
                        current_attention_kwargs["prev_resample_mask"] = prev_resample_mask
                    noise_pred, hidden_states_list, prev_resample_mask = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        branch_block_samples=branch_block_samples,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=current_attention_kwargs,
                        add_first=add_first,
                        branch_block_masks=mask[:, :, :1, :, :] if mask_add else None,
                        id_pool_resample_learnable=id_pool_resample_learnable,
                        return_hidden_states=True,
                        return_resample_mask=True,
                        return_dict=False,
                    )
                    noise_pred = noise_pred.float()
                    if window_idx < n_windows - 1:
                        if t.item() == timesteps[-1]:
                            prev_window_states_dict[0] = {
                                layer_idx: hidden_states.detach() 
                                for layer_idx, hidden_states in enumerate(hidden_states_list)
                            }
                            prev_resample_mask = prev_resample_mask.detach()

                    # perform guidance
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    latents = latents.to(prompt_embeds.dtype)

                    # inpainting
                    latents = latents.to(device)

                    if replace_gt:
                        init_latents_proper = video_latents.to(device)
                        if do_classifier_free_guidance:
                            init_mask, _ = mask.chunk(2)
                        else:
                            init_mask = mask
                        init_mask = init_mask.to(device)

                        if i < len(timesteps) - 1:
                            noise_timestep = timesteps[i + 1]
                            init_latents_proper = self.scheduler.add_noise(
                                init_latents_proper.to(device), noise.to(device), torch.tensor([noise_timestep]).to(device)
                            ).to(device)

                        if mask_background:
                            latents = init_mask * init_latents_proper + (1 - init_mask) * latents
                        else:
                            latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                    # call the callback, if provided
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                        mask = callback_outputs.pop("mask", mask)
                        masked_video_latents = callback_outputs.pop("masked_video_latents", masked_video_latents)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            for i in range(latents.shape[1]):
                compressed_start_frame = window_idx * latents.shape[1]
                if window_idx > 0 and stride < num_frames:
                    compressed_start_frame -= (int((num_frames - stride) // self.vae_scale_factor_temporal) + 1) * window_idx
                elif window_idx > 0 and stride == num_frames:
                    compressed_start_frame -= window_idx
                elif window_idx ==0:
                    compressed_start_frame = 0
                else:
                    raise ValueError(f"stride: {stride}, num_frames: {num_frames}")
                compressed_frame_idx = compressed_start_frame + i
                frame_accumulator[:, compressed_frame_idx] += latents[:, i]
                frame_counts[compressed_frame_idx] += 1


        for i in range(num_frame_latents):
            if frame_counts[i] > 0:
                frame_accumulator[:, i] /= frame_counts[i]

        if not output_type == "latent":
            video = self.decode_latents(frame_accumulator)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = frame_accumulator

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)