#!/bin/bash
#SBATCH --job-name=evalMME
#SBATCH --partition=tolga-lab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=spartacus-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G #128G    
#SBATCH --time=12:00:00
#SBATCH --output=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/eval_lora_baseline/%x_%j.out
#SBATCH --error=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/eval_lora_baseline/%x_%j.err

source ~/.bashrc

conda activate llava_dsleaf


python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b
