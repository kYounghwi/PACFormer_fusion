from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .data_loader import (
    NWPMemmapStore,
    PVNWPDataset,
    StandardScaler,
    read_nwp_csv,
    read_pv_csv,
)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: PVNWPDataset
    val_dataset: PVNWPDataset
    test_dataset: PVNWPDataset
    pv_scaler: StandardScaler
    nwp_scaler: StandardScaler
    site_columns: list
    nwp_variables: list


def _resolve(root_path, file_path):
    path = Path(file_path)
    return path if path.is_absolute() else Path(root_path) / path


def _parse_columns(value):
    if not value:
        return None
    return [column.strip() for column in value.split(",") if column.strip()]


def _validate_station_csv(path, site_columns):
    frame = pd.read_csv(path)
    lower = {column.lower(): column for column in frame.columns}
    if not any(name in lower for name in ("lat", "latitude", "y", "y_lat")):
        raise ValueError("Station CSV must contain a latitude column.")
    if not any(name in lower for name in ("lon", "longitude", "lng", "x", "x_lon")):
        raise ValueError("Station CSV must contain a longitude column.")
    if len(frame) != len(site_columns):
        raise ValueError(
            f"Station CSV has {len(frame)} rows, but PV CSV has {len(site_columns)} sites."
        )
    site_column = lower.get("site", lower.get("id"))
    if site_column is not None:
        station_order = frame[site_column].astype(str).tolist()
        if station_order != [str(column) for column in site_columns]:
            raise ValueError(
                "Station CSV site order must match the PV columns exactly."
            )


def _split_sample_starts(dates, nwp_fields, seq_len, pred_len, train_ratio, val_ratio):
    total = len(dates)
    train_end = int(total * train_ratio)
    test_size = int(total * (1.0 - train_ratio - val_ratio))
    val_end = total - test_size
    if (
        train_end <= seq_len + pred_len
        or val_end <= train_end
        or total <= val_end + pred_len
    ):
        raise ValueError(
            "Dataset is too short for the requested windows and split ratios."
        )

    split_bounds = {
        "train": (seq_len, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total),
    }
    starts = {}
    for split, (target_first, target_limit) in split_bounds.items():
        candidates = []
        for target_start in range(target_first, target_limit - pred_len + 1):
            start = target_start - seq_len
            origin = pd.Timestamp(dates[target_start])
            if origin in nwp_fields:
                candidates.append(start)
        if not candidates:
            raise ValueError(f"No valid {split} samples have matching NWP origins.")
        starts[split] = np.asarray(candidates, dtype=np.int64)
    return starts, train_end


def _scale_nwp_fields(fields, train_origins, enabled=True):
    variables = next(iter(fields.values())).shape[-1]
    if enabled:
        train_values = np.concatenate(
            [fields[origin].reshape(-1, variables) for origin in train_origins], axis=0
        )
        scaler = StandardScaler.fit(train_values, axis=0)
        scaled = {origin: scaler.transform(field) for origin, field in fields.items()}
    else:
        scaler = StandardScaler(
            mean=np.zeros(variables, dtype=np.float32),
            std=np.ones(variables, dtype=np.float32),
        )
        scaled = {origin: field.astype(np.float32) for origin, field in fields.items()}
    return scaled, scaler


def build_data_bundle(args):
    pv_path = _resolve(args.root_path, args.pv_csv)
    station_path = _resolve(args.root_path, args.stations_csv_path)

    dates, pv_values, site_columns = read_pv_csv(
        pv_path,
        date_col=args.date_col,
        pv_columns=_parse_columns(args.pv_columns),
    )
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between zero and one.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than one.")
    train_end = int(len(dates) * args.train_ratio)

    if args.nwp_backend == "memmap":
        memmap_dir = _resolve(args.root_path, args.nwp_memmap_dir)
        nwp_fields = NWPMemmapStore(
            memmap_dir,
            dates,
            args.pred_len,
            time_shift=args.nwp_time_shift,
            scale=args.scale_nwp,
            train_time_end=train_end,
            stats_path=args.nwp_stats_path or None,
            stats_overwrite=args.nwp_stats_overwrite,
            stats_chunk_time=args.nwp_stats_chunk_time,
        )
        nwp_variables = nwp_fields.variables
        nwp_shape = nwp_fields.shape
    else:
        nwp_path = _resolve(args.root_path, args.nwp_csv)
        nwp_fields, nwp_variables, nwp_shape = read_nwp_csv(
            nwp_path,
            origin_col=args.nwp_origin_col,
            lead_col=args.nwp_lead_col,
            grid_y_col=args.nwp_grid_y_col,
            grid_x_col=args.nwp_grid_x_col,
            variable_columns=_parse_columns(args.nwp_variables),
        )
    _validate_station_csv(station_path, site_columns)

    if args.num_groups > len(site_columns):
        raise ValueError("num_groups cannot exceed the number of PV sites.")
    if args.d_model % args.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads.")
    if args.patch_len > args.seq_len:
        raise ValueError("patch_len cannot exceed seq_len.")
    if any(
        patch > size
        for patch, size in zip(
            args.cube_patch, (nwp_shape[2], nwp_shape[0], nwp_shape[1])
        )
    ):
        raise ValueError("cube_patch cannot exceed the NWP time or spatial dimensions.")

    starts, train_end = _split_sample_starts(
        dates,
        nwp_fields,
        args.seq_len,
        args.pred_len,
        args.train_ratio,
        args.val_ratio,
    )

    if args.scale:
        pv_scaler = StandardScaler.fit(pv_values[:train_end], axis=0)
        pv_values = pv_scaler.transform(pv_values)
    else:
        pv_scaler = StandardScaler(
            mean=np.zeros(len(site_columns), dtype=np.float32),
            std=np.ones(len(site_columns), dtype=np.float32),
        )

    if args.nwp_backend == "memmap":
        nwp_scaler = nwp_fields.scaler
    else:
        train_origins = [
            pd.Timestamp(dates[start + args.seq_len]) for start in starts["train"]
        ]
        nwp_fields, nwp_scaler = _scale_nwp_fields(
            nwp_fields, train_origins, args.scale_nwp
        )

    args.enc_in = len(site_columns)
    args.c_out = len(site_columns)
    args.grid_h, args.grid_w, args.time_len, args.nwp_n_vars = nwp_shape
    args.spatial_resolution = (args.grid_h, args.grid_w)
    args.stations_csv_path = str(station_path)

    datasets = {
        split: PVNWPDataset(
            dates,
            pv_values,
            nwp_fields,
            starts[split],
            args.seq_len,
            args.pred_len,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=args.device.type == "cuda",
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }

    return DataBundle(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        pv_scaler=pv_scaler,
        nwp_scaler=nwp_scaler,
        site_columns=site_columns,
        nwp_variables=nwp_variables,
    )
