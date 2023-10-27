#!/bin/bash

PROJECT_PATH="/home/reutemann/Dino-Video-Summarization-Transformer"
EXP_NAME="le_001"
DATASET="kinetics400"
DATA_PATH="/graphics/scratch/datasets/MSVD/YouTubeClips"
CHECKPOINT="/home/reutemann/Dino-Video-Summarization-Transformer/checkpoints/kinetics400_vitb_ssl_finetuned_3x3_224_student/checkpoint.pth"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_similarity_dino.py \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 0.001 \
  --batch_size_per_gpu 8 \
  --num_workers 4 \
  --num_labels 400 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/eval/$EXP_NAME" \
  --opts \
  DATA.PATH_TO_DATA_DIR "/home/reutemann/Dino-Video-Summarization-Transformer/MSVD" \
  DATA.PATH_PREFIX "${DATA_PATH}" \
  DATA.USE_FLOW False
  