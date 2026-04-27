#!/bin/bash
#SBATCH --job-name=evalllavabenchcompress5-2
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

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/Comp5/compress5-2_0.001llava-v1.5-13b-lora/compress5-2_0.001_llava-v1.5-13b-merged \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/compress5-2_llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/compress5-2_llava-v1.5-13b.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/compress5-2_llava-v1.5-13b.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/compress5-2_llava-v1.5-13b.jsonl
