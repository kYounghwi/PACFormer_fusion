import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run the full AUSGRID PACFormer grid."
    )
    parser.add_argument(
        "--fusion-mode", required=True, choices=("decay", "swiglu")
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=(48, 96, 168, 240)
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parent
    temporal_cube = {48: 6, 96: 12, 168: 15, 240: 15}
    for index, horizon in enumerate(args.horizons, start=1):
        if horizon not in temporal_cube:
            raise ValueError(
                f"Unsupported horizon {horizon}; choose from {sorted(temporal_cube)}."
            )
        print(
            f"[AUSGRID FULL {index}/{len(args.horizons)}] "
            f"horizon={horizon} fusion={args.fusion_mode}",
            flush=True,
        )
        command = [
            sys.executable,
            "main.py",
            "--mode", "train",
            "--root_path", "data/AUSGRID",
            "--pv_csv", "pv.csv",
            "--nwp_backend", "memmap",
            "--nwp_memmap_dir", "nwp",
            "--nwp_time_shift", "0",
            "--nwp_history_len", "-1",
            "--nwp_future_len", "-1",
            "--stations_csv_path", "stations.csv",
            "--seq_len", "288",
            "--pred_len", str(horizon),
            "--batch_size", "32",
            "--train_epochs", "50",
            "--total_epochs", "50",
            "--early_stopping",
            "--patience", "5",
            "--learning_rate", "1e-4",
            "--lradj", "dual",
            "--d_model", "128",
            "--d_ff", "256",
            "--n_heads", "4",
            "--e_layers", "1",
            "--d_layers", "1",
            "--num_groups", "25",
            "--sites_per_group", "5",
            "--diversity_weight", "0.9",
            "--patch_len", "16",
            "--stride", "8",
            "--padding_patch", "end",
            "--dropout", "0.05",
            "--cube_patch", str(temporal_cube[horizon]), "1", "1",
            "--nwp_vit_layers", "2",
            "--q_event_mode", "event",
            "--fusion_mode", args.fusion_mode,
            "--output_attention",
            "--loss", "mse",
            "--weight_decay", "0",
            "--grad_clip", "1.0",
            "--metric_threshold", "1e-3",
            "--seed", "42",
            "--checkpoints", "results",
            "--run_name",
            f"AUSGRID_FULL_H{horizon}_HISTNWP_STRICT_{args.fusion_mode}",
        ]
        subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
