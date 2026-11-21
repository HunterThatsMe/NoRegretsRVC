#!/usr/bin/env bash
set -euo pipefail

# 1. Preprocess dataset
python core.py preprocess \
  --model_name "NoRegretsVoice" --dataset_path "/kaggle/input/datasets/your-username/NoRegretsVoice" \
  --sample_rate 32000 --cpu_cores 4 \
  --cut_preprocess "Automatic" --chunk_len 3 --overlap_len 0.3 \
  --noise_reduction False --process_effects False

# 2. Extract features (rmvpe + contentvec)
python core.py extract \
  --model_name "NoRegretsVoice" --f0_method rmvpe \
  --sample_rate 32000 --embedder_model contentvec \
  --gpu 0-1 --cpu_cores 4 --include_mutes 0

# 3. Train HiFi-GAN RVC model (fp32)
python core.py train \
  --model_name "NoRegretsVoice" --optimizer AdamW --vocoder "HiFi-GAN" \
  --save_every_epoch 5 --total_epoch 200 --batch_size 2 \
  --sample_rate 32000 --gpu 0-1 --fp16 False \
  --save_only_latest True --save_every_weights True \
  --pretrained True --checkpointing False \
  --cache_data_in_gpu False

# 4. Generate Auto feature index
python core.py index --model_name "NoRegretsVoice" --index_algorithm Auto
