#!/bin/bash
#SBATCH --job-name=evalmmbenchcompress5-2
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

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/Comp5/compress5-2_0.001llava-v1.5-13b-lora/compress5-2_0.001_llava-v1.5-13b-merged \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/compress5-2_llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment compress5-2_llava-v1.5-13b
