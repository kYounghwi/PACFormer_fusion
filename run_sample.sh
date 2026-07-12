#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

python main.py \
  --mode train \
  --root_path data/AUSGRID_SAMPLE \
  --pv_csv pv.csv \
  --nwp_backend memmap \
  --nwp_memmap_dir nwp \
  --nwp_time_shift 0 \
  --stations_csv_path stations.csv \
  --seq_len 288 \
  --pred_len 96 \
  --batch_size 32 \
  --train_epochs 50 \
  --total_epochs 50 \
  --learning_rate 1e-4 \
  --lradj dual \
  --d_model 128 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 1 \
  --d_layers 1 \
  --num_groups 25 \
  --patch_len 16 \
  --stride 8 \
  --padding_patch end \
  --dropout 0.05 \
  --cube_patch 4 1 1 \
  --nwp_vit_layers 2 \
  --q_event_mode event \
  --output_attention \
  --loss mse \
  --weight_decay 0 \
  --grad_clip 1.0 \
  --metric_threshold 1e-3 \
  --seed 42 \
  --checkpoints results \
  --run_name AUSGRID_SAMPLE_ORIGINAL_96_50ep
