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

inpainting_sample_ids=(79)
video_editing_instructions=(
    "Change the lamp into a small tree."
)

inpainting_frames=49
down_sample_fps=8
image_or_video_path="../data/videovo/raw_video"
inpainting_mask_meta="../data/pexels_videovo_test_dataset.csv"
prev_clip_weight=0.0

img_inpainting_model="../ckpt/flux_inp"

llm_model="gpt-4o"

dilate_size=48

output_base_path="./edit_fps${down_sample_fps}_dilate_${dilate_size}"

if [ "$llm_model" != "None" ]; then
    output_base_path="${output_base_path}_${llm_model}"
fi

if [ ! -d "$output_base_path" ]; then
    mkdir -p "$output_base_path"
fi

for inpainting_branch in "${inpainting_branches[@]}"; do
    for ((i=0; i<${#inpainting_sample_ids[@]}; i++)); do
        inpainting_sample_id="${inpainting_sample_ids[i]}"
        video_editing_instruction="${video_editing_instructions[i]}"
        
        output_path="${output_base_path}/${inpainting_sample_id}_${inpainting_frames}_prev_clip_weight_${prev_clip_weight}_${video_editing_instruction}.mp4"
        
        python edit.py \
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
            --prev_clip_weight $prev_clip_weight \
            --img_inpainting_model $img_inpainting_model \
            --video_editing_instruction "$video_editing_instruction" \
            --llm_model $llm_model \
            --dilate_size $dilate_size \
            --lora_rank $lora_rank
    done
done
