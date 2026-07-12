import numpy as np


def mae(prediction, target):
    return float(np.mean(np.abs(prediction - target)))


def mse(prediction, target):
    return float(np.mean(np.square(prediction - target)))


def rmse(prediction, target):
    return float(np.sqrt(mse(prediction, target)))


def metric(prediction, target):
    return {
        "mae": mae(prediction, target),
        "mse": mse(prediction, target),
        "rmse": rmse(prediction, target),
    }


class SiteMetricAccumulator:
    """Accumulate normalized-scale errors and report an equal-site average."""

    def __init__(self, num_sites):
        self.absolute_error_sum = np.zeros(num_sites, dtype=np.float64)
        self.squared_error_sum = np.zeros(num_sites, dtype=np.float64)
        self.valid_count = np.zeros(num_sites, dtype=np.int64)
        self.total_count = np.zeros(num_sites, dtype=np.int64)

    def update(self, prediction, target, valid_mask):
        if prediction.shape != target.shape or target.shape != valid_mask.shape:
            raise ValueError(
                "prediction, target, and valid_mask must have equal shapes."
            )
        if prediction.ndim != 3:
            raise ValueError("Metric inputs must have shape [batch, horizon, site].")

        error = prediction - target
        axes = (0, 1)
        self.absolute_error_sum += np.where(valid_mask, np.abs(error), 0.0).sum(
            axis=axes, dtype=np.float64
        )
        self.squared_error_sum += np.where(valid_mask, np.square(error), 0.0).sum(
            axis=axes, dtype=np.float64
        )
        self.valid_count += valid_mask.sum(axis=axes, dtype=np.int64)
        self.total_count += np.prod(valid_mask.shape[:2], dtype=np.int64)

    def compute(self):
        valid_sites = self.valid_count > 0
        if not np.any(valid_sites):
            raise ValueError("No site contains a valid target value.")

        site_mae = np.full(self.valid_count.shape, np.nan, dtype=np.float64)
        site_mse = np.full(self.valid_count.shape, np.nan, dtype=np.float64)
        site_mae[valid_sites] = (
            self.absolute_error_sum[valid_sites] / self.valid_count[valid_sites]
        )
        site_mse[valid_sites] = (
            self.squared_error_sum[valid_sites] / self.valid_count[valid_sites]
        )
        site_rmse = np.sqrt(site_mse)
        site_valid_fraction = np.divide(
            self.valid_count,
            self.total_count,
            out=np.zeros_like(site_mae),
            where=self.total_count > 0,
        )

        metrics = {
            "mae": float(np.mean(site_mae[valid_sites])),
            "mse": float(np.mean(site_mse[valid_sites])),
            "rmse": float(np.mean(site_rmse[valid_sites])),
            "valid_count": int(self.valid_count.sum()),
            "valid_fraction": float(self.valid_count.sum() / self.total_count.sum()),
            "valid_sites": int(valid_sites.sum()),
            "num_sites": int(valid_sites.size),
        }
        site_metrics = {
            "mae": site_mae,
            "mse": site_mse,
            "rmse": site_rmse,
            "valid_count": self.valid_count.copy(),
            "valid_fraction": site_valid_fraction,
        }
        return metrics, site_metrics
