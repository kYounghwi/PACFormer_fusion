import argparse
import random
from pathlib import Path

import numpy as np
import torch

import PACFormer.exp as experiment
from src.data_factory import build_data_bundle


def get_args():
    parser = argparse.ArgumentParser(description="PACFormer PV-NWP forecasting")
    parser.add_argument("--model", default="pacformer", choices=["pacformer", "pangu"])
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--root_path", default="data")
    parser.add_argument("--pv_csv", default="pv.csv")
    parser.add_argument("--nwp_csv", default="nwp.csv")
    parser.add_argument(
        "--nwp_backend",
        required=True,
        choices=["csv", "memmap", "icon_archive"],
        help=(
            "csv: legacy exact-origin fields; memmap: ERA5-style valid-time array; "
            "icon_archive: choose the latest issued forecast cycle at every hourly "
            "PV forecast origin."
        ),
    )
    parser.add_argument("--nwp_memmap_dir", default="")
    parser.add_argument("--nwp_time_shift", type=int, default=0)
    parser.add_argument(
        "--nwp_history_len",
        type=int,
        default=-1,
        help=(
            "Historical NWP hours prepended to future NWP. -1 uses "
            "min(pred_len, 120), matching the HistNWP setting."
        ),
    )
    parser.add_argument(
        "--nwp_future_len",
        type=int,
        default=-1,
        help=(
            "Future NWP hours supplied to the model. -1 uses "
            "min(pred_len, 120), so H168/H240 never use NWP beyond lead 120."
        ),
    )
    parser.add_argument("--nwp_stats_path", default="")
    parser.add_argument("--nwp_stats_overwrite", action="store_true")
    parser.add_argument("--nwp_stats_chunk_time", type=int, default=512)
    parser.add_argument("--stations_csv_path", default="stations.csv")
    parser.add_argument("--date_col", default="date")
    parser.add_argument("--pv_columns", default="")
    parser.add_argument("--nwp_origin_col", default="origin_time")
    parser.add_argument("--nwp_lead_col", default="lead_idx")
    parser.add_argument("--nwp_grid_y_col", default="grid_y")
    parser.add_argument("--nwp_grid_x_col", default="grid_x")
    parser.add_argument("--nwp_variables", default="")

    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_scale", action="store_false", dest="scale")
    parser.add_argument("--no_scale_nwp", action="store_false", dest="scale_nwp")
    parser.set_defaults(scale=True, scale_nwp=True)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_groups", type=int, default=3)
    parser.add_argument(
        "--sites_per_group",
        type=int,
        default=0,
        help=(
            "Top-k sites retained per candidate group. 0 preserves the original "
            "rule floor(sqrt(num_groups))."
        ),
    )
    parser.add_argument(
        "--diversity_weight",
        type=float,
        default=0.9,
        help=(
            "Sequential-diversity weight in [0,1]; 1 emphasizes diversity "
            "and 0 emphasizes sharp membership. 0.9 reproduces the original "
            "0.1 sharpness / 0.9 diversity mixture."
        ),
    )
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--padding_patch", default="end", choices=["end", "none"])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--norm", default="LayerNorm")
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument(
        "--q_event_mode", default="event", choices=["event", "original", "add"]
    )
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--cube_patch", type=int, nargs=3, default=[2, 4, 4])
    parser.add_argument("--nwp_vit_layers", type=int, default=1)
    parser.add_argument("--pooling", action="store_true")
    parser.add_argument("--nwp_use_stats", action="store_true")
    parser.add_argument(
        "--fusion_mode",
        required=True,
        choices=["decay", "swiglu"],
        help=(
            "Required. 'decay' keeps an explicit NWP output prior and adds a "
            "horizon-attenuated PV correction; it is usually more stable with "
            "limited data or long horizons. 'swiglu' learns a nonlinear latent "
            "residual around the decoded NWP token; it is more expressive when "
            "sufficient data support cross-modal interactions. Neither mode is "
            "universally superior, so report the selected mode."
        ),
    )
    parser.add_argument(
        "--fusion_d_ff",
        type=int,
        default=0,
        help="SwiGLU hidden width; 0 reuses --d_ff.",
    )

    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lradj", default="dual", choices=["type1", "type2", "type3", "cosine", "dual"]
    )
    parser.add_argument("--total_epochs", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss", default="mse", choices=["mse", "mae"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--metric_threshold", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoints", default="results")
    parser.add_argument("--run_name", default="PACFormer")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_args(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.cube_patch = tuple(args.cube_patch)
    if args.nwp_history_len < 0:
        args.nwp_history_len = min(args.pred_len, 120)
    if args.nwp_history_len < 0:
        raise ValueError("nwp_history_len must be non-negative or -1 for auto.")
    if args.nwp_future_len < 0:
        args.nwp_future_len = min(args.pred_len, 120)
    if args.nwp_future_len <= 0 or args.nwp_future_len > 120:
        raise ValueError("nwp_future_len must be in [1, 120] or -1 for auto.")
    if args.sites_per_group < 0:
        raise ValueError("sites_per_group must be non-negative.")
    if not 0.0 <= args.diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be in [0, 1].")
    if args.fusion_d_ff <= 0:
        args.fusion_d_ff = args.d_ff
    if args.nwp_backend == "memmap" and not args.nwp_memmap_dir:
        raise ValueError("--nwp_memmap_dir is required for --nwp_backend memmap.")
    args.padding_patch = None if args.padding_patch == "none" else args.padding_patch
    if args.total_epochs <= 0:
        args.total_epochs = args.train_epochs
    return args


def restore_test_config(args):
    if args.mode != "test":
        return args
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in test mode.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config", {})
    saved_fusion = saved_config.get("fusion_mode")
    if saved_fusion is not None and saved_fusion != args.fusion_mode:
        raise ValueError(
            "--fusion_mode must match the checkpoint: "
            f"requested={args.fusion_mode!r}, checkpoint={saved_fusion!r}."
        )
    runtime_keys = {
        "mode",
        "root_path",
        "pv_csv",
        "nwp_csv",
        "nwp_backend",
        "nwp_memmap_dir",
        "stations_csv_path",
        "checkpoint",
        "batch_size",
        "num_workers",
        "device",
        "checkpoints",
        "run_name",
    }
    for key, value in saved_config.items():
        if key not in runtime_keys and hasattr(args, key):
            setattr(args, key, value)
    args.cube_patch = tuple(args.cube_patch)
    return args


def main():

    args = restore_test_config(prepare_args(get_args()))
    print(f"Device: {args.device}")

    data = build_data_bundle(args)
    model = experiment.build_model(args)

    if args.mode == "train":
        checkpoint = experiment.train(args, model, data)
    else:
        checkpoint = Path(args.checkpoint)

    experiment.test(args, model, data, checkpoint)


if __name__ == "__main__":
    main()
