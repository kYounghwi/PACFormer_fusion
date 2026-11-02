@echo off
setlocal EnableDelayedExpansion

set "FUSION_MODE=%~1"
if /I "%FUSION_MODE%"=="decay" goto run
if /I "%FUSION_MODE%"=="swiglu" goto run
echo Usage: run_ausgrid_full.bat ^{decay^|swiglu^} [adam^|muon]
exit /b 2

:run
set "OPTIMIZER=%~2"
if "%OPTIMIZER%"=="" set "OPTIMIZER=adam"
if /I not "%OPTIMIZER%"=="adam" if /I not "%OPTIMIZER%"=="muon" (
  echo Optimizer must be adam or muon.
  exit /b 2
)
set "OPTIMIZER_SUFFIX="
if /I "%OPTIMIZER%"=="muon" set "OPTIMIZER_SUFFIX=_MUON"
cd /d "%~dp0"
for %%H in (48 96 168 240) do (
  if %%H==48 set "CUBE_T=6"
  if %%H==96 set "CUBE_T=12"
  if %%H==168 set "CUBE_T=15"
  if %%H==240 set "CUBE_T=15"
  echo [AUSGRID FULL] horizon=%%H fusion=%FUSION_MODE% optimizer=%OPTIMIZER%
  python main.py ^
    --mode train ^
    --root_path data/AUSGRID ^
    --pv_csv pv.csv ^
    --nwp_backend memmap ^
    --nwp_memmap_dir nwp ^
    --nwp_time_shift 0 ^
    --nwp_history_len -1 ^
    --nwp_future_len -1 ^
    --stations_csv_path stations.csv ^
    --seq_len 288 ^
    --pred_len %%H ^
    --batch_size 32 ^
    --train_epochs 50 ^
    --total_epochs 50 ^
    --early_stopping ^
    --patience 5 ^
    --learning_rate 1e-4 ^
    --lradj dual ^
    --d_model 128 ^
    --d_ff 256 ^
    --n_heads 4 ^
    --e_layers 1 ^
    --d_layers 1 ^
    --num_groups 25 ^
    --sites_per_group 5 ^
    --diversity_weight 0.9 ^
    --patch_len 16 ^
    --stride 8 ^
    --padding_patch end ^
    --dropout 0.05 ^
    --cube_patch !CUBE_T! 1 1 ^
    --nwp_vit_layers 2 ^
    --q_event_mode event ^
    --fusion_mode %FUSION_MODE% ^
    --optimizer %OPTIMIZER% ^
    --output_attention ^
    --loss mse ^
    --weight_decay 0 ^
    --grad_clip 1.0 ^
    --metric_threshold 1e-3 ^
    --seed 42 ^
    --checkpoints results ^
    --run_name AUSGRID_FULL_H%%H_HISTNWP_STRICT_%FUSION_MODE%%OPTIMIZER_SUFFIX%
  if errorlevel 1 exit /b 1
)

endlocal
