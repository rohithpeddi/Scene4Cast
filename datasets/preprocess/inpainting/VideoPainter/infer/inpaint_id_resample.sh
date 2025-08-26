#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

model_path="../ckpt/CogVideoX-5b-I2V"
num_inference_steps=50
guidance_scale=6.0
num_videos_per_prompt=1
dtype="bfloat16"

inpainting_branches=(
    ../ckpt/VideoPainter/checkpoints/branch
)

id_adapter_resample_learnable_path=../ckpt/VideoPainterID/checkpoints

lora_rank=256

inpainting_sample_ids=($(seq 0 100000))

inpainting_frames=49
down_sample_fps=8
overlap_frames=0
image_or_video_path="../data/videovo/raw_video"

data_kind="test"
if [ "$data_kind" == "test" ]; then
    inpainting_mask_meta="../data/pexels_videovo_test_dataset.csv"
elif [ "$data_kind" == "val" ]; then
    inpainting_mask_meta="../data/pexels_videovo_val_dataset.csv"
elif [ "$data_kind" == "train" ]; then
    inpainting_mask_meta="../data/pexels_videovo_train_dataset.csv"
else
    echo "data_kind must be test or train or val"
    exit 1
fi

prev_clip_weight=0.5

img_inpainting_model="../ckpt/flux_inp"

llm_model="gpt-4o"

dilate_size=32

output_base_path="./inp_IP_fps${down_sample_fps}_dilate_${dilate_size}_${data_kind}"

if [ "$llm_model" != "None" ]; then
    output_base_path="${output_base_path}_${llm_model}"
fi

if [ "$overlap_frames" != "0" ]; then
    output_base_path="${output_base_path}_overlap_${overlap_frames}"
fi

if [ ! -d "$output_base_path" ]; then
    sudo mkdir -p "$output_base_path"
    sudo chmod -R 777 "$output_base_path"
fi

for inpainting_branch in "${inpainting_branches[@]}"; do
    for inpainting_sample_id in "${inpainting_sample_ids[@]}"; do

        output_path="${output_base_path}/${inpainting_sample_id}.mp4"
        
        python inpaint.py \
            --prompt "$prompt" \
            --model_path "$model_path" \
            --inpainting_branch "$inpainting_branch" \
            --output_path "$output_path" \
            --num_inference_steps "$num_inference_steps" \
            --guidance_scale "$guidance_scale" \
            --num_videos_per_prompt "$num_videos_per_prompt" \
            --dtype "$dtype" \
            --generate_type "i2v_inpainting" \
            --inpainting_mask_meta "$inpainting_mask_meta" \
            --inpainting_sample_id "$inpainting_sample_id" \
            --inpainting_frames "$inpainting_frames" \
            --image_or_video_path "$image_or_video_path" \
            --first_frame_gt \
            --replace_gt \
            --mask_add \
            --down_sample_fps $down_sample_fps \
            --overlap_frames $overlap_frames \
            --prev_clip_weight $prev_clip_weight \
            --img_inpainting_model $img_inpainting_model \
            --llm_model $llm_model \
            --dilate_size $dilate_size \
            --id_adapter_resample_learnable_path $id_adapter_resample_learnable_path \
            --lora_rank $lora_rank \
            --long_video
    done
done
