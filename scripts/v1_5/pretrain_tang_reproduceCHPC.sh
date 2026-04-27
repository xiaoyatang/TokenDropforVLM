#!/bin/bash
#SBATCH --job-name=compressor2chpc_pretrain
#SBATCH --partition=rai-gpu-grn 
#SBATCH --qos=rai-gpu-grn-short 
#SBATCH --account=rai
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=12 
#SBATCH --mem=320G #80G    
#SBATCH --time=7:00:00
#SBATCH --output=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp2-2/%x_%j.out
#SBATCH --error=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp2-2/%x_%j.err

source /uufs/sci.utah.edu/home/xiaoya/.bashrc
source /uufs/sci.utah.edu/home/xiaoya/anaconda3/etc/profile.d/conda.sh
conda activate llava

cd /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA
PORT=$((29500 + RANDOM % 10000))
echo "Using MASTER_PORT=$PORT"

deepspeed --master_port $PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/Comp2-2/compress2chpc_llava-v1.5-13b-pretrain \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --compressor_config ./tokenDropConfig/compressor2.yaml


# run1
    # --num_train_epochs 1 \
    # --per_device_train_batch_size 32 \
    # --per_device_eval_batch_size 4 \
    # --gradient_accumulation_steps 2 \
    # --evaluation_strategy "no" \
    # --save_strategy "steps" \
    # --save_steps 24000 \
    # --save_total_limit 1 \
    # --learning_rate 1e-3 \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \
    # --lr_scheduler_type "cosine" \
    # --logging_steps 1 \
    # --tf32 True \
    # --model_max_length 2048 \
    # --gradient_checkpointing True \
    # --dataloader_num_workers 4 \
    # --lazy_preprocess True \
    # --report_to wandb \
    # --compressor_config ./tokenDropConfig/compressor.yaml