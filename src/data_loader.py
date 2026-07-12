import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .timefeatures import time_features


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values, axis=0, eps=1e-6):
        values = np.asarray(values, dtype=np.float64)
        mean = np.nanmean(values, axis=axis, keepdims=False)
        std = np.nanstd(values, axis=axis, keepdims=False)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values):
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values):
        return values * self.std + self.mean


def read_pv_csv(path, date_col="date", pv_columns=None):
    frame = pd.read_csv(path)
    if date_col not in frame.columns:
        raise ValueError(f"PV CSV must contain a '{date_col}' column.")

    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col).drop_duplicates(date_col).reset_index(drop=True)
    columns = (
        list(pv_columns) if pv_columns else [c for c in frame.columns if c != date_col]
    )
    if not columns:
        raise ValueError("PV CSV does not contain any site columns.")

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"PV columns not found: {missing}")

    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = values.ffill().bfill()
    if values.isna().any().any():
        raise ValueError("PV CSV contains values that cannot be imputed.")

    dates = pd.DatetimeIndex(frame[date_col])
    return dates, values.to_numpy(dtype=np.float64), columns


def read_nwp_csv(
    path,
    origin_col="origin_time",
    lead_col="lead_idx",
    grid_y_col="grid_y",
    grid_x_col="grid_x",
    variable_columns=None,
):
    frame = pd.read_csv(path)
    index_columns = [origin_col, lead_col, grid_y_col, grid_x_col]
    missing = [c for c in index_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"NWP CSV is missing index columns: {missing}")

    variables = (
        list(variable_columns)
        if variable_columns
        else [c for c in frame.columns if c not in index_columns]
    )
    if not variables:
        raise ValueError("NWP CSV does not contain meteorological variables.")

    frame[origin_col] = pd.to_datetime(frame[origin_col])
    for column in [lead_col, grid_y_col, grid_x_col]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame[variables] = frame[variables].apply(pd.to_numeric, errors="coerce")
    if frame[variables].isna().any().any():
        raise ValueError("NWP CSV contains missing or non-numeric weather values.")
    if frame.duplicated(index_columns).any():
        raise ValueError("NWP CSV contains duplicate origin/lead/grid coordinates.")

    time_len = int(frame[lead_col].max()) + 1
    grid_h = int(frame[grid_y_col].max()) + 1
    grid_w = int(frame[grid_x_col].max()) + 1
    expected = time_len * grid_h * grid_w

    fields = {}
    for origin, group in frame.groupby(origin_col, sort=True):
        if len(group) != expected:
            raise ValueError(
                f"NWP origin {origin} has {len(group)} cells; expected {expected} "
                f"for shape ({grid_h}, {grid_w}, {time_len}, {len(variables)})."
            )
        field = np.empty((grid_h, grid_w, time_len, len(variables)), dtype=np.float32)
        y = group[grid_y_col].to_numpy(dtype=int)
        x = group[grid_x_col].to_numpy(dtype=int)
        lead = group[lead_col].to_numpy(dtype=int)
        field[y, x, lead, :] = group[variables].to_numpy(dtype=np.float32)
        fields[pd.Timestamp(origin)] = field

    return fields, variables, (grid_h, grid_w, time_len, len(variables))


class NWPMemmapStore:
    """Lazy future-NWP windows backed by a time-major memmap."""

    def __init__(
        self,
        directory,
        dates,
        time_len,
        time_shift=0,
        scale=True,
        scale_eps=1e-6,
        train_time_end=None,
        stats_path=None,
        stats_overwrite=False,
        stats_chunk_time=512,
    ):
        directory = Path(directory)
        with (directory / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)

        shape = tuple(int(value) for value in manifest["shape"])
        self.values = np.memmap(
            directory / manifest.get("data_file", "data.float16.memmap"),
            mode="r",
            dtype=np.float16,
            shape=shape,
        )
        if len(dates) != shape[0]:
            raise ValueError(
                f"PV length and NWP memmap length differ: {len(dates)} != {shape[0]}."
            )

        self.dates = pd.DatetimeIndex(dates)
        self.index_by_time = {
            pd.Timestamp(timestamp): index for index, timestamp in enumerate(self.dates)
        }
        self.time_len = int(time_len)
        self.time_shift = int(time_shift)
        self.scale_eps = float(scale_eps)
        self.variables = (
            np.load(directory / "variables.npy", allow_pickle=True).astype(str).tolist()
        )

        channels = shape[-1]
        if scale:
            stats_path = (
                Path(stats_path)
                if stats_path is not None
                else directory / "era5_nwp_stats_train.npz"
            )
            if stats_path.exists() and not stats_overwrite:
                stats = np.load(stats_path)
                mean = stats["mean"].astype(np.float32)
                std = stats["std"].astype(np.float32)
            else:
                if train_time_end is None:
                    raise ValueError(
                        "train_time_end is required when NWP statistics must be computed."
                    )
                mean, std = self._compute_stats(
                    train_time_end, stats_chunk_time, channels
                )
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    stats_path,
                    mean=mean,
                    std=std,
                    variables=np.asarray(self.variables),
                )
        else:
            mean = np.zeros(channels, dtype=np.float32)
            std = np.ones(channels, dtype=np.float32)
        self.scaler = StandardScaler(mean=mean, std=std)
        self.shape = (shape[2], shape[1], self.time_len, channels)

    def _compute_stats(self, train_time_end, chunk_time, channels):
        end = int(max(1, min(train_time_end, self.values.shape[0])))
        sums = np.zeros(channels, dtype=np.float64)
        squared_sums = np.zeros(channels, dtype=np.float64)
        counts = np.zeros(channels, dtype=np.int64)

        for start in range(0, end, max(1, int(chunk_time))):
            values = np.asarray(
                self.values[start : min(start + chunk_time, end)], dtype=np.float32
            ).reshape(-1, channels)
            finite = np.isfinite(values)
            values = np.where(finite, values, 0.0)
            sums += values.sum(axis=0, dtype=np.float64)
            squared_sums += (values * values).sum(axis=0, dtype=np.float64)
            counts += finite.sum(axis=0)

        mean = sums / np.maximum(counts, 1)
        variance = squared_sums / np.maximum(counts, 1) - mean * mean
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std < self.scale_eps, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    def _slice_bounds(self, origin):
        index = self.index_by_time.get(pd.Timestamp(origin))
        if index is None:
            return None
        start = index + self.time_shift
        end = start + self.time_len
        if start < 0 or end > self.values.shape[0]:
            return None
        return start, end

    def __contains__(self, origin):
        return self._slice_bounds(origin) is not None

    def __getitem__(self, origin):
        bounds = self._slice_bounds(origin)
        if bounds is None:
            raise KeyError(f"No complete NWP window is available for {origin}.")
        start, end = bounds
        field = np.asarray(self.values[start:end], dtype=np.float32)
        np.subtract(field, self.scaler.mean, out=field)
        np.divide(field, self.scaler.std + self.scale_eps, out=field)
        return np.transpose(field, (2, 1, 0, 3))


class PVNWPDataset(Dataset):
    """Sliding-window multi-site PV dataset with future gridded NWP fields."""

    def __init__(self, dates, pv_values, nwp_fields, sample_starts, seq_len, pred_len):
        self.dates = pd.DatetimeIndex(dates)
        self.pv_values = np.asarray(pv_values, dtype=np.float32)
        self.nwp_fields = nwp_fields
        self.sample_starts = np.asarray(sample_starts, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.calendar = time_features(self.dates)

    def __len__(self):
        return len(self.sample_starts)

    def __getitem__(self, item):
        start = int(self.sample_starts[item])
        target_start = start + self.seq_len
        target_end = target_start + self.pred_len
        origin = pd.Timestamp(self.dates[target_start])

        x_enc = self.pv_values[start:target_start]
        target = self.pv_values[target_start:target_end]
        x_mark_enc = self.calendar[start:target_start]
        x_mark_dec = self.calendar[target_start:target_end]
        x_dec = np.zeros_like(target, dtype=np.float32)
        nwp = self.nwp_fields[origin]

        return tuple(
            torch.from_numpy(array)
            for array in (x_enc, target, x_mark_enc, x_dec, x_mark_dec, nwp)
        )
