from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .data_loader import (
    ICONForecastArchiveStore,
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
    site_capacity: np.ndarray
    capacity_column: str


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


def _load_site_capacity(path, site_columns):
    """Load installed capacity without a data-derived peak fallback."""
    frame = pd.read_csv(path)
    lower = {str(column).lower(): column for column in frame.columns}
    candidates = (
        "capacity",
        "capacity_kw",
        "installed_capacity",
        "installed_capacity_kw",
        "kwp",
    )
    column = next((lower[name] for name in candidates if name in lower), None)
    if column is None:
        raise ValueError(
            "Station CSV must contain installed capacity in one of: "
            "capacity, capacity_kw, installed_capacity, "
            "installed_capacity_kw, kWp."
        )
    capacity = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if len(capacity) != len(site_columns):
        raise ValueError("Capacity row count must match the PV site count.")
    invalid = ~np.isfinite(capacity) | (capacity <= 0)
    if np.any(invalid):
        bad = np.flatnonzero(invalid)[:10].tolist()
        raise ValueError(f"Invalid installed capacity at station rows {bad}.")
    return capacity, str(column)


def _split_sample_starts(
    dates, pv_values, nwp_fields, seq_len, pred_len, train_ratio, val_ratio
):
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
            pv_window = pv_values[start : target_start + pred_len]
            if np.isfinite(pv_window).all() and origin in nwp_fields:
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


def _archive_scaler(fields, train_cutoff, enabled=True):
    variables = next(iter(fields.values())).shape[-1]
    if not enabled:
        return StandardScaler(
            mean=np.zeros(variables, dtype=np.float32),
            std=np.ones(variables, dtype=np.float32),
        )
    issued_in_train = [
        field for initialization, field in fields.items()
        if pd.Timestamp(initialization) <= pd.Timestamp(train_cutoff)
    ]
    if not issued_in_train:
        raise ValueError("No ICON initialization is available in the training period.")
    values = np.concatenate(
        [field.reshape(-1, variables) for field in issued_in_train], axis=0
    )
    return StandardScaler.fit(values, axis=0)


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
    history_len = int(args.nwp_history_len)
    future_len = int(args.nwp_future_len)

    if args.nwp_backend == "memmap":
        memmap_dir = _resolve(args.root_path, args.nwp_memmap_dir)
        nwp_fields = NWPMemmapStore(
            memmap_dir,
            dates,
            future_len,
            history_len=history_len,
            time_shift=args.nwp_time_shift,
            scale=args.scale_nwp,
            train_time_end=train_end,
            stats_path=args.nwp_stats_path or None,
            stats_overwrite=args.nwp_stats_overwrite,
            stats_chunk_time=args.nwp_stats_chunk_time,
        )
        nwp_variables = nwp_fields.variables
        nwp_shape = nwp_fields.shape
    elif args.nwp_backend == "icon_archive":
        nwp_path = _resolve(args.root_path, args.nwp_csv)
        archive_fields, nwp_variables, _archive_shape = read_nwp_csv(
            nwp_path,
            origin_col=args.nwp_origin_col,
            lead_col=args.nwp_lead_col,
            grid_y_col=args.nwp_grid_y_col,
            grid_x_col=args.nwp_grid_x_col,
            variable_columns=_parse_columns(args.nwp_variables),
        )
        nwp_scaler = _archive_scaler(
            archive_fields, dates[max(0, train_end - 1)], args.scale_nwp
        )
        nwp_fields = ICONForecastArchiveStore(
            archive_fields,
            future_len=future_len,
            history_len=history_len,
            time_shift=args.nwp_time_shift,
            scaler=nwp_scaler,
        )
        nwp_shape = nwp_fields.shape
    else:
        if history_len:
            raise ValueError(
                "Legacy exact-origin CSV fields do not define historical NWP. "
                "Use --nwp_backend memmap, implement the ICON history hook, or "
                "set --nwp_history_len 0."
            )
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
    site_capacity, capacity_column = _load_site_capacity(station_path, site_columns)

    if args.num_groups > len(site_columns):
        raise ValueError("num_groups cannot exceed the number of PV sites.")
    if args.sites_per_group > len(site_columns):
        raise ValueError("sites_per_group cannot exceed the number of PV sites.")
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
        pv_values,
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

    if args.nwp_backend in {"memmap", "icon_archive"}:
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
        site_capacity=site_capacity,
        capacity_column=capacity_column,
    )
