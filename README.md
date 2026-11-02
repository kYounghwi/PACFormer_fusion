# Propagation-Aware Multi-Site PV Context Fusion with Imperfect Numerical Weather Prediction for Medium-Range PV Forecasting

Official PyTorch implementation of the PACFormer-based PV-NWP information fusion framework.

<p align="center">
  <br>
<img width="1757" height="1378" alt="Image" src="https://github.com/user-attachments/assets/41376d61-234f-43d0-9482-596bc53065ca" />
</p>

## Overview

Medium-range photovoltaic (PV) forecasting across geographically distributed sites requires the integration of complementary but imperfect information. Historical PV observations describe site-level generation states and multi-site spatio-temporal context, whereas numerical weather prediction (NWP) provides future meteorological priors that may contain regional bias and limited local detail.

PACFormer addresses this problem through two components:

- **PV context encoder:** a dual-branch Transformer specialized to capture shared variability and lead–lag propagation across dynamic site groups by incorporating the intrinsic characteristics of PV generation signals.
- **Asymmetric PV-NWP fusion:** an NWP-based future prior is preserved as the base representation, while the PACFormer-derived PV context provides condition-dependent refinement.

Across six multi-site datasets, the framework improves the mean squared error by 14.27% relative to the average of the three strongest competing baselines. Additional experiments evaluate robustness under high-variability conditions, imperfect NWP corruption, and limited training data.

## Setup


```bash
git clone https://github.com/kYounghwi/PACFormer_fusion.git
cd PACFormer_fusion

conda create -n pacformer python=3.10.19 -y
conda activate pacformer
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.8 build of PyTorch. For a different CUDA version or CPU-only execution, install the corresponding PyTorch build before installing the remaining dependencies.

## Run

On Windows, select one fusion mode and run the corresponding batch command from the repository root:

```bat
run_ausgrid_full.bat decay adam cuda:0
```

or

```bat
run_ausgrid_full.bat swiglu adam cuda:0
```

Adam remains the default optimizer. Muon is enabled only when it is explicitly requested:

```bat
run_ausgrid_full.bat swiglu muon cuda:0
```

The NWP-only Pangu baseline is run with:

```bat
run_pangu_ausgrid_full.bat adam cuda:0
```

The final argument selects the execution device (`auto`, `cpu`, `cuda:0`, `cuda:1`, etc.); omitting it uses `auto`, which selects `cuda:0` when CUDA is available and otherwise uses the CPU. The batch files execute the complete train-validation-test workflow for the 48-, 96-, 168-, and 240-hour horizons. The latest checkpoint and the checkpoint with the best validation nMAE are saved as `last.pt` and `best.pt`, respectively, and `best.pt` is automatically evaluated on the test split after training. In Muon mode, eligible two-dimensional hidden-layer matrices use Muon, while biases, normalization parameters, embeddings, and output heads use AdamW.


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
|-- data/AUSGRID/                 # Full AUSGRID PV and ERA5 inputs
|-- main.py                       # Command-line entry point
|-- run_ausgrid_full.bat          # Full Windows train-validation-test workflow
|-- run_pangu_ausgrid_full.bat    # Full Pangu baseline workflow
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
