# Propagation-Aware Multi-Site PV Context Fusion with Imperfect Numerical Weather Prediction for Medium-Range PV Forecasting

Official PyTorch implementation of the PACFormer-based PV-NWP information fusion framework.

<p align="center">
  <br>
  <strong>Framework overview figure</strong><br>
  <em>The model overview will be added upon publication.</em>
  <br><br>
</p>

## Overview

Medium-range photovoltaic (PV) forecasting across geographically distributed sites requires the integration of complementary but imperfect information. Historical PV observations describe site-level generation states and multi-site spatio-temporal context, whereas numerical weather prediction (NWP) provides future meteorological priors that may contain regional bias and limited local detail.

PACFormer addresses this problem through two components:

- **PV context encoder:** a dual-branch Transformer that combines globally shared inter-site variability with dynamic group-level lead-lag propagation context.
- **Asymmetric PV-NWP fusion:** an NWP-based future prior is preserved as the base representation, while the PACFormer-derived PV context provides condition-dependent refinement.

Across six multi-site datasets, the framework improves the mean squared error by 14.27% relative to the average of the three strongest competing baselines. Additional experiments evaluate robustness under high-variability conditions, imperfect NWP corruption, and limited training data.

## Setup

The implementation was verified with Python 3.10.19, CUDA 12.8, and the package versions listed in `requirements.txt`.

```bash
git clone https://github.com/kYounghwi/PACFormer_fusion.git
cd PACFormer_fusion

conda create -n pacformer python=3.10.19 -y
conda activate pacformer
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.8 build of PyTorch. For a different CUDA version or CPU-only execution, install the corresponding PyTorch build before installing the remaining dependencies.

## Run

Training, validation, and testing are managed through `main.py`. During training, the model is validated after every epoch. The latest checkpoint and the checkpoint with the best validation MAE are saved as `last.pt` and `best.pt`, respectively. After training, `best.pt` is automatically evaluated on the test split.

```bash
python main.py \
  --mode train \
  --root_path PATH_TO_DATA \
  --pv_csv pv.csv \
  --nwp_backend memmap \
  --nwp_memmap_dir nwp \
  --stations_csv_path stations.csv \
  --seq_len 288 \
  --pred_len 96 \
  --batch_size 32 \
  --train_epochs 50 \
  --learning_rate 1e-4 \
  --d_model 128 \
  --d_ff 512 \
  --n_heads 4 \
  --num_groups 25 \
  --patch_len 16 \
  --stride 8 \
  --checkpoints results \
  --run_name PACFormer
```

Use `python main.py --help` for the complete list of data, model, optimization, and evaluation options.

To evaluate an existing checkpoint:

```bash
python main.py \
  --mode test \
  --root_path PATH_TO_DATA \
  --pv_csv pv.csv \
  --nwp_backend memmap \
  --nwp_memmap_dir nwp \
  --stations_csv_path stations.csv \
  --batch_size 32 \
  --checkpoint results/PACFormer/best.pt
```

Model and forecasting hyperparameters are restored from the checkpoint. Runtime data paths, batch size, and worker settings are taken from the test command.

## AUSGRID Sample

The repository includes a small executable sample under `data/AUSGRID_SAMPLE`. It contains 1,000 contiguous hourly PV observations from 299 sites, their coordinates, and the corresponding ERA5-based NWP fields.

Run the complete train-validation-test workflow with:

```bash
bash run_sample.sh
```

The script uses the experimental 96-step configuration:

| Parameter | Value |
|---|---:|
| Input / prediction length | 288 / 96 |
| Batch size / epochs | 32 / 50 |
| Initial learning rate | `1e-4` |
| Model / FFN dimension | 128 / 512 |
| Attention heads | 4 |
| Representative groups | 25 |
| Patch length / stride | 16 / 8 |
| NWP cube patch | `(4, 1, 1)` |
| NWP ViT layers | 2 |
| Event query mode | `event` |

NWP normalization statistics are generated from the training split on the first run and subsequently reused.

## Evaluation

Metrics are calculated in normalized PV space after excluding positions whose inverse-normalized target is at or below `1e-3`. MAE, MSE, and RMSE are first calculated independently for each site and then averaged equally across sites with at least one valid target.

Test outputs are saved in the checkpoint directory:

- `test_metrics.json`: aggregate site-averaged metrics
- `test_site_metrics.json` and `test_site_metrics.npz`: per-site metrics
- `test_prediction.npy` and `test_target.npy`: inverse-transformed predictions and targets

## Repository Layout

```text
PACFormer_fusion/
|-- PACFormer/
|   |-- exp.py                    # Training, validation, and testing
|   `-- modules/
|       |-- PACFormer.py          # PACFormer and asymmetric fusion model
|       |-- tst_backbone.py       # Group-propagation branch
|       |-- nwp_branch.py         # Gridded NWP encoder
|       |-- attention.py          # Attention layers
|       |-- embedding.py          # PV and positional embeddings
|       |-- masking.py            # Propagation and shifted-window masks
|       `-- transformer.py        # Transformer building blocks
|-- src/
|   |-- data_factory.py           # Dataset and DataLoader construction
|   |-- data_loader.py            # PV-NWP loading and alignment
|   |-- metrics.py                # Site-averaged evaluation metrics
|   |-- timefeatures.py           # Temporal covariates
|   `-- tools.py                  # Learning-rate scheduling utilities
|-- data/AUSGRID_SAMPLE/          # Executable PV-NWP sample
|-- main.py                       # Command-line entry point
|-- run_sample.sh                 # AUSGRID sample experiment
`-- requirements.txt
```

## Citation

The manuscript is currently under review. Please use the following provisional citation; publication details will be updated after acceptance.

```bibtex
@unpublished{kim2026pacformer,
  title  = {Propagation-Aware Multi-Site PV Context Fusion with Imperfect Numerical Weather Prediction for Medium-Range PV Forecasting},
  author = {Kim, Younghwi and others},
  year   = {2026},
  note   = {Manuscript under review}
}
```
