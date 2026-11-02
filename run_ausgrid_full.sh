#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

FUSION_MODE="${1:-}"
if [[ "$FUSION_MODE" != "decay" && "$FUSION_MODE" != "swiglu" ]]; then
  echo "Usage: bash run_ausgrid_full.sh {decay|swiglu} [adam|muon] [auto|cpu|cuda:N]"
  exit 2
fi
OPTIMIZER="${2:-adam}"
if [[ "$OPTIMIZER" != "adam" && "$OPTIMIZER" != "muon" ]]; then
  echo "Optimizer must be adam or muon."
  exit 2
fi
DEVICE="${3:-auto}"
OPTIMIZER_SUFFIX=""
if [[ "$OPTIMIZER" == "muon" ]]; then
  OPTIMIZER_SUFFIX="_MUON"
fi

# Override when only selected horizons are needed, for example:
# HORIZONS="48 96" bash run_ausgrid_full.sh decay
read -r -a HORIZON_LIST <<< "${HORIZONS:-48 96 168 240}"

for HORIZON in "${HORIZON_LIST[@]}"; do
  case "$HORIZON" in
    48) CUBE_T=6 ;;
    96) CUBE_T=12 ;;
    168|240) CUBE_T=15 ;;
    *) echo "Unsupported horizon: $HORIZON"; exit 2 ;;
  esac
  python main.py \
    --mode train \
    --root_path data/AUSGRID \
    --pv_csv pv.csv \
    --nwp_backend memmap \
    --nwp_memmap_dir nwp \
    --nwp_time_shift 0 \
    --nwp_history_len -1 \
    --nwp_future_len -1 \
    --stations_csv_path stations.csv \
    --seq_len 288 \
    --pred_len "$HORIZON" \
    --batch_size 32 \
    --train_epochs 50 \
    --total_epochs 50 \
    --early_stopping \
    --patience 5 \
    --learning_rate 1e-4 \
    --lradj dual \
    --d_model 128 \
    --d_ff 256 \
    --n_heads 4 \
    --e_layers 1 \
    --d_layers 1 \
    --num_groups 25 \
    --sites_per_group 5 \
    --diversity_weight 0.9 \
    --patch_len 16 \
    --stride 8 \
    --padding_patch end \
    --dropout 0.05 \
    --cube_patch "$CUBE_T" 1 1 \
    --nwp_vit_layers 2 \
    --q_event_mode event \
    --fusion_mode "$FUSION_MODE" \
    --optimizer "$OPTIMIZER" \
    --device "$DEVICE" \
    --output_attention \
    --loss mse \
    --weight_decay 0 \
    --grad_clip 1.0 \
    --metric_threshold 1e-3 \
    --seed 42 \
    --checkpoints results \
    --run_name "AUSGRID_FULL_H${HORIZON}_HISTNWP_STRICT_${FUSION_MODE}${OPTIMIZER_SUFFIX}"
done
