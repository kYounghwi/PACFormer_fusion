# PACFormer PV-NWP Fusion

PyTorch implementation of PACFormer for multi-site medium-range PV forecasting with asymmetric PV-NWP fusion.

## Installation

The project was verified with Python 3.10.19, CUDA 12.8, and the package versions in `requirements.txt`.

```bash
conda activate test
pip install -r requirements.txt
```

Only `train` and `test` execution modes are provided. Training validates after every epoch, saves `last.pt` and the best validation checkpoint `best.pt`, then automatically evaluates the test split using `best.pt`.

## Included AUSGRID sample

`data/AUSGRID_SAMPLE` contains 1,000 contiguous hourly PV rows, all 299 station coordinates, and the exactly corresponding 1,000 NWP rows.

```text
data/AUSGRID_SAMPLE/
  pv.csv
  stations.csv
  nwp/
    manifest.json
    data.float16.memmap
    variables.npy
    lat.npy
    lon.npy
    pv_time.npy
    era5_utc_time.npy
```

NWP statistics are generated from the sample's training period on the first run and saved as `data/AUSGRID_SAMPLE/nwp/era5_nwp_stats_train.npz`. Later runs reuse that file.

## Train and test the sample

From Git Bash with the `test` Conda environment active:

```bash
bash run_ausgrid_sample.sh
```

From PowerShell without activating the environment first:

```powershell
.\run_ausgrid_sample.ps1
```

The Bash script requires Git Bash, WSL, or another Bash installation. The PowerShell script uses `C:\Users\USER\anaconda3\envs\test` through `conda run` and works on the current machine without Bash.

The script reproduces the original AUSGRID 96-step hyperparameters: `seq_len=288`, `pred_len=96`, `batch_size=32`, `d_model=128`, `d_ff=512`, four heads, one PV encoder/decoder layer, two NWP ViT layers, patch length 16, stride 8, NWP cube patch `(4,1,1)`, and `output_attention=True`. It performs 50 epochs without early stopping and then tests `results/AUSGRID_SAMPLE_ORIGINAL_96_50ep/best.pt`.

## Test an existing checkpoint

```bash
python main.py \
  --mode test \
  --root_path data/AUSGRID_SAMPLE \
  --pv_csv pv.csv \
  --nwp_backend memmap \
  --nwp_memmap_dir nwp \
  --stations_csv_path stations.csv \
  --batch_size 32 \
  --checkpoint results/AUSGRID_SAMPLE_ORIGINAL_96_50ep/best.pt
```

The checkpoint restores model and forecasting hyperparameters. Runtime data paths and batch size are taken from the test command.

## Evaluation

Metrics are calculated in normalized PV space. Positions whose inverse-normalized target is at or below `1e-3` are excluded. MAE, MSE, and RMSE are first calculated independently for each site and then averaged with equal weight across sites containing at least one valid target. Test runs save the aggregate metrics to `test_metrics.json` and the per-site metrics to `test_site_metrics.json` and `test_site_metrics.npz`. The default dual learning-rate schedule applies exponential decay to the PV-side parameters and warmup-cosine decay to `nwp_branch`.
