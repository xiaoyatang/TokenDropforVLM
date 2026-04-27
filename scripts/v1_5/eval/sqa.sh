#!/bin/bash
#SBATCH --job-name=evalsqacompress5-2
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

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/Comp5/compress5-2_0.001llava-v1.5-13b-lora/compress5-2_0.001_llava-v1.5-13b-merged \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/compress5-2_llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/compress5-2_llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/compress5-2_llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/compress5-2_llava-v1.5-13b_result.json
