#!/bin/bash
#SBATCH --job-name=lva_l40compress_concaSMB_pretrain
#SBATCH --partition=tolga-lab,general-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodelist=pegasus
#SBATCH --cpus-per-task=32 
#SBATCH --mem=60G #80G/320G    
#SBATCH --time=24:00:00
#SBATCH --output=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/%x_%j.out
#SBATCH --error=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/%x_%j.err

source ~/.bashrc

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
    --output_dir ./checkpoints/compressL40ConcaSMB_llava-v1.5-13b-reloadtest \
    --max_steps 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --compressor_config ./tokenDropConfig/compressor.yaml \
    --pretrain_mm_mlp_adapter /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/compressL40ConcaSMB_llava-v1.5-13b-pretrain/mm_projector.bin
    # --save_steps 24000 \
    # --num_train_epochs 1 \