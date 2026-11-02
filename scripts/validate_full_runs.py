import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as entrypoint
from PACFormer import exp
from src.data_factory import build_data_bundle


EXPERIMENTS = (
    ("RN_PV", "data/RN_PV_FULL", 0, "icon_nwp_stats_train.npz", 28),
    ("US_UPV", "data/US_UPV_FULL", 8, "era5_nwp_stats_train.npz", 27),
)


def make_args(dataset, root, shift, stats_file, groups, horizon, model):
    time_patch = 2 if horizon == 48 else 4
    sys.argv = [
        "main.py",
        "--model",
        model,
        "--root_path",
        root,
        "--pv_csv",
        "pv.csv",
        "--nwp_backend",
        "memmap",
        "--nwp_memmap_dir",
        "nwp",
        "--nwp_time_shift",
        str(shift),
        "--nwp_history_len",
        "-1",
        "--nwp_future_len",
        "-1",
        "--nwp_stats_path",
        f"{root}/nwp/{stats_file}",
        "--stations_csv_path",
        "stations.csv",
        "--seq_len",
        "288",
        "--pred_len",
        str(horizon),
        "--batch_size",
        "1",
        "--d_model",
        "128",
        "--d_ff",
        "512",
        "--n_heads",
        "4",
        "--num_groups",
        str(groups),
        "--patch_len",
        "16",
        "--stride",
        "8",
        "--cube_patch",
        str(time_patch),
        "4",
        "4",
        "--nwp_vit_layers",
        "2",
        "--q_event_mode",
        "event",
        "--fusion_mode",
        "decay",
        "--seed",
        "42",
    ]
    args = entrypoint.prepare_args(entrypoint.get_args())
    args.run_name = f"validate_{model}_{dataset}_{horizon}"
    return args


def main():
    for dataset, root, shift, stats_file, groups in EXPERIMENTS:
        for horizon in (48, 96):
            data = None
            for model_name in ("pacformer", "pangu"):
                args = make_args(
                    dataset, root, shift, stats_file, groups, horizon, model_name
                )
                if data is None:
                    data = build_data_bundle(args)
                    print(
                        dataset,
                        horizon,
                        "samples",
                        len(data.train_dataset),
                        len(data.val_dataset),
                        len(data.test_dataset),
                    )
                else:
                    args.enc_in = len(data.site_columns)
                    args.c_out = len(data.site_columns)
                    sample_nwp = data.train_dataset[0][-1]
                    args.grid_h = sample_nwp.shape[0]
                    args.grid_w = sample_nwp.shape[1]
                    args.time_len = sample_nwp.shape[2]
                    args.nwp_n_vars = sample_nwp.shape[3]
                    args.spatial_resolution = (args.grid_h, args.grid_w)
                    args.stations_csv_path = str(Path(root) / "stations.csv")
                model = exp.build_model(args).eval()
                batch = next(iter(data.train_loader))
                with torch.no_grad():
                    output = model(
                        batch[0].to(args.device),
                        batch[2].to(args.device),
                        batch[3].to(args.device),
                        batch[4].to(args.device),
                        batch[5].to(args.device),
                    )
                print(model_name, tuple(batch[5].shape), tuple(output.shape))
                del model, batch, output
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
