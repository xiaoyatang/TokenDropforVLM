#!/bin/bash
#SBATCH --job-name=mergecompressor1-pos2
#SBATCH --partition=tolga-lab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=spartacus-3
#SBATCH --cpus-per-task=8 
#SBATCH --mem=40G #128G    
#SBATCH --time=12:00:00
#SBATCH --output=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp5/%x_%j.out
#SBATCH --error=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp5/%x_%j.err

source ~/.bashrc

conda activate llava

cd /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA

python ./scripts/merge_lora_weights.py \
    --model-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/Comp5/compressor1-pos2_llava-v1.5-13b-lora \
    --model-base lmsys/vicuna-13b-v1.5 \
    --save-model-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/Comp5/compressor1-pos2_llava-v1.5-13b-lora/compressor1-pos2_llava-v1.5-13b-merged/
    # --model-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/compressConcaSMB_llava-v1.5-13b-lora/
    # --save-model-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/compressConcaSMB_llava-v1.5-13b-lora/compressConcaSMB_llava-v1.5-13b-merged/
    
    # --model-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/llava-v1.5-13b-lora-run2/
    # --save-model-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/llava-v1.5-13b-lora-run2/llava-v1.5-13b-merged/   