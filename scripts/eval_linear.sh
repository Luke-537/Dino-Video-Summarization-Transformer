#!/bin/bash

PROJECT_PATH="/home/reutemann/Dino-Video-Summarization-Transformer"
EXP_NAME="le_002"
DATASET="kinetics400"
DATA_PATH="/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized"
CHECKPOINT="/home/reutemann/Dino-Video-Summarization-Transformer/checkpoints/model_k400_pretrained/kinetics400_vitb_ssl.pth"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=5
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  eval_linear.py \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 2 \
  --lr 0.001 \
  --batch_size_per_gpu 16 \
  --num_workers 4 \
  --num_labels 400 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/eval/$EXP_NAME" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}/annotations" \
  DATA.PATH_PREFIX "${DATA_PATH}/test" \
  DATA.USE_FLOW False
