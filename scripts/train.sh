#!/bin/bash

PROJECT_PATH="/home/reutemann/Dino-Video-Summarization-Transformer"
DATA_PATH="/graphics/scratch2/students/reutemann/kinetics-dataset/k400_resized/annotations"
EXP_NAME="kinetics400_vitb_ssl_finetuned_8x4_224"

cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  train_ssl.py \
  --arch "timesformer" \
  --batch_size_per_gpu 1 \
  --data_path "${DATA_PATH}" \
  --output_dir "checkpoints/$EXP_NAME" \
  --epochs 1 \
  --warmup_epochs 0 \
  --cfg "models/configs/Kinetics/TimeSformer_divST_custom_224.yaml" \
  --pretrained_rgb "/home/reutemann/Dino-Video-Summarization-Transformer/checkpoints/model_k400_pretrained/kinetics400_vitb_ssl.pth" \
  --opts \
  MODEL.TWO_STREAM False \
  MODEL.TWO_TOKEN False \
  DATA.NO_FLOW_AUG False \
  DATA.USE_FLOW False \
  DATA.RAND_CONV False \
  DATA.NO_SPATIAL False

