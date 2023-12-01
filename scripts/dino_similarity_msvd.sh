#!/bin/bash

PROJECT_PATH="/home/reutemann/Dino-Video-Summarization-Transformer"
DATASET="kinetics400"
DATA_PATH="/graphics/scratch/datasets/MSVD/YouTubeClips"
CHECKPOINT="/home/reutemann/Dino-Video-Summarization-Transformer/checkpoints/model_k400_pretrained/kinetics400_vitb_ssl.pth"

cd "$PROJECT_PATH" || exit

export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port="$RANDOM" \
  dino_similarity.py \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 20 \
  --lr 0.001 \
  --batch_size_per_gpu 4 \
  --num_workers 4 \
  --num_labels 400 \
  --dataset "$DATASET" \
  --cfg "models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml" \
  --output_dir "checkpoints/eval/$EXP_NAME" \
  --opts \
  DATA.PATH_TO_DATA_DIR "/home/reutemann/Dino-Video-Summarization-Transformer/MSVD" \
  DATA.PATH_PREFIX "${DATA_PATH}" \
  DATA.USE_FLOW False
  
#--cfg "models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml" \
#--cfg "models/configs/Kinetics/TimeSformer_divST_custom_224.yaml" \
#CHECKPOINT="/home/reutemann/Dino-Video-Summarization-Transformer/checkpoints/model_k400_pretrained/kinetics400_vitb_ssl.pth"
#CHECKPOINT="/home/reutemann/Dino-Video-Summarization-Transformer/checkpoints/kinetics400_vitb_ssl_finetuned_lower_learning_rate_8x2_224/checkpoint.pth"