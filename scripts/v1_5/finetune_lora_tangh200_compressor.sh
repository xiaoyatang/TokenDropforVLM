#!/bin/bash
#SBATCH --job-name=compressor3-2-pos-reorder_ftLora
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=32
#SBATCH --mem=520G 
#SBATCH --time=24:00:00
#SBATCH --output=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp5/%x_%j.out
#SBATCH --error=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp5/%x_%j.err

source ~/.bashrc

conda activate llava_dsleaf

cd /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA
PORT=$((29500 + RANDOM % 10000))
echo "Using MASTER_PORT=$PORT"

deepspeed --master_port=$PORT llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/Comp5/ccompressor3-2-pos-reorder_llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Comp5/compressor3-6-pos_llava-v1.5-13b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --compressor_config ./tokenDropConfig/compressor3-6-pos.yaml \
    --compressor_lr 2e-5 


# default: --group_by_modality_length True