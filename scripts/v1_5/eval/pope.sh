#!/bin/bash
#SBATCH --job-name=evalpopecompress5-2
#SBATCH --partition=tolga-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=spartacus-3
#SBATCH --mem=40G #128G    
#SBATCH --time=12:00:00
#SBATCH --output=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp5/%x_%j.out
#SBATCH --error=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp5/%x_%j.err

source ~/.bashrc

conda activate llava

cd /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/Comp5/compress5-2_0.001llava-v1.5-13b-lora/compress5-2_0.001_llava-v1.5-13b-merged \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/compress5-2_llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/compress5-2_llava-v1.5-13b.jsonl
