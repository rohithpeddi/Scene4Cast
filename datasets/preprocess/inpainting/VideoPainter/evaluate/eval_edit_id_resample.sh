export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

model_path="../ckpt/CogVideoX-5b-I2V"
inpainting_branch=../ckpt/VideoPainter/checkpoints/branch
id_adapter_resample_learnable_path=../ckpt/VideoPainterID/checkpoints
img_inpainting_model="../ckpt/flux_inp"

down_sample_fps=8

declare -A configs=(
    ["standard"]="
        dataset_name=VPBench
        video_root=../data/videovo/raw_video
        mask_root=../data/video_inpainting
        caption_path=../data/our_video_edit.csv
        save_addr=edit"
    ["anyl"]="
        dataset_name=VPBench
        video_root=../data/videovo/raw_video
        mask_root=../data/video_inpainting
        caption_path=../data/our_video_edit_long.csv
        save_addr=edit_anyl"
)

run_evaluation() {
    local dilate_size=$1
    local replace_gt=$2
    local config_type=$3
    
    eval "${configs[$config_type]}"
    
    echo "Running evaluation for ${config_type}"
    echo "dataset: ${dataset_name}"
    echo "dilate_size: ${dilate_size}, replace_gt: ${replace_gt}"
    echo "video_root: ${video_root}"
    echo "mask_root: ${mask_root}"
    echo "caption_path: ${caption_path}"
    echo "save_addr: ${save_addr}"

    sudo mkdir -p $save_addr
    sudo chmod 777 $save_addr
    
    local max_length=49
    if [ "$config_type" = "anyl" ]; then
        max_length=9999
    fi
    
    python eval_edit.py \
        --dataset $dataset_name \
        --video_root $video_root \
        --mask_root $mask_root \
        --caption_path $caption_path \
        --max_video_length $max_length \
        --save_addr $save_addr \
        --save_results \
        --model_path $model_path \
        --inpainting_branch $inpainting_branch \
        --img_inpainting_model $img_inpainting_model \
        --dilate_size $dilate_size \
        --down_sample_fps $down_sample_fps \
        --first_frame_gt \
        --mask_add \
        --id_adapter_resample_learnable_path $id_adapter_resample_learnable_path \
        $([ "$replace_gt" = "true" ] && echo "--replace_gt") \
        $([ "$config_type" = "anyl" ] && echo "--long_video")
}

for config_type in "standard" "anyl"; do
    echo "Starting evaluations for config: $config_type"
    for dilate in 48 24 0; do
        for replace in true false; do
            run_evaluation $dilate "$replace" "$config_type"
        done
    done
done



