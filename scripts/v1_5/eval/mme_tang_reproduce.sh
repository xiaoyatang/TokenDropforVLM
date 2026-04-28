#!/bin/bash
#SBATCH --job-name=evalMMEcompressor3-2-pos-reorder
#SBATCH --partition=tolga-lab
#SBATCH --gres=gpu:1
#SBATCH --nodelist=chimera
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G #128G    
#SBATCH --time=24:00:00
#SBATCH --output=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp5/%x_%j.out
#SBATCH --error=/uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/logs/Comp5/%x_%j.err

source ~/.bashrc

conda activate llava

cd /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/Comp5/compressor3-2-pos-reorder_llava-v1.5-13b-lora/compressor3-2-pos-reorder_llava-v1.5-13b-merged \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/compressor3-2-pos-reorder_llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment compressor3-2-pos-reorder_llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/compressor3-2-pos-reorder_llava-v1.5-13b



# --model-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/llava-v1.5-13b-lora-run2/llava-v1.5-13b-merged/ \
# --answers-file ./playground/data/eval/MME/answers/baseline_lora_run2.jsonl \
# python convert_answer_to_mme.py --experiment baseline_lora_run2
# python calculation.py --results_dir answers/baseline_lora_run2


# --model-path /uufs/sci.utah.edu/projects/DeepLearning/Xiaoya_GSV/LLaVA/checkpoints/compressConcaSMB_llava-v1.5-13b-lora/compressConcaSMB_llava-v1.5-13b-merged \
# --answers-file ./playground/data/eval/MME/answers/compressConcaSMB_Pretrain3-2-lora.jsonl \
# python convert_answer_to_mme.py --experiment compressConcaSMB_Pretrain3-2-lora
# python calculation.py --results_dir answers/compressConcaSMB_Pretrain3-2-lora