import numpy as np


class CapacityMetricAccumulator:
    """Stream the four capacity-normalized metrics used in the paper.

    Site-level metrics first normalize each site's physical-unit error by its
    installed capacity and then give every evaluated site equal weight.
    Regional-total metrics first aggregate generation over all valid sites at
    each target timestamp and normalize that aggregate error by the matching
    sum of installed capacities.
    """

    def __init__(self, capacity):
        self.capacity = np.asarray(capacity, dtype=np.float64)
        if self.capacity.ndim != 1:
            raise ValueError("capacity must be a one-dimensional site vector.")
        if not np.isfinite(self.capacity).all() or not (self.capacity > 0).all():
            raise ValueError("Every site must have a finite positive capacity.")

        sites = len(self.capacity)
        self.absolute_error_sum = np.zeros(sites, dtype=np.float64)
        self.squared_error_sum = np.zeros(sites, dtype=np.float64)
        self.valid_count = np.zeros(sites, dtype=np.int64)
        self.regional_ratio_absolute_sum = 0.0
        self.regional_ratio_squared_sum = 0.0
        self.regional_count = 0

    def update(self, prediction, target, valid_mask):
        if prediction.shape != target.shape or target.shape != valid_mask.shape:
            raise ValueError(
                "prediction, target, and valid_mask must have equal shapes."
            )
        if prediction.ndim != 3 or prediction.shape[-1] != len(self.capacity):
            raise ValueError(
                "Metric inputs must have shape [batch, horizon, site]."
            )

        valid = (
            np.asarray(valid_mask, dtype=bool)
            & np.isfinite(prediction)
            & np.isfinite(target)
        )
        error = np.where(valid, prediction - target, 0.0)
        self.absolute_error_sum += np.abs(error).sum(axis=(0, 1), dtype=np.float64)
        self.squared_error_sum += np.square(error).sum(
            axis=(0, 1), dtype=np.float64
        )
        self.valid_count += valid.sum(axis=(0, 1), dtype=np.int64)

        # Regional totals use the same point-wise daylight mask. The
        # denominator contains exactly the capacities of contributing sites.
        active = valid.any(axis=-1)
        if np.any(active):
            prediction_total = np.where(valid, prediction, 0.0).sum(axis=-1)
            target_total = np.where(valid, target, 0.0).sum(axis=-1)
            capacity_total = np.where(
                valid, self.capacity[None, None, :], 0.0
            ).sum(axis=-1)
            ratio = (
                (prediction_total[active] - target_total[active])
                / capacity_total[active]
            )
            self.regional_ratio_absolute_sum += np.abs(ratio).sum(dtype=np.float64)
            self.regional_ratio_squared_sum += np.square(ratio).sum(dtype=np.float64)
            self.regional_count += int(active.sum())

    def compute(self):
        evaluated = self.valid_count > 0
        if not np.any(evaluated):
            raise ValueError("No site contains a valid daylight target value.")
        if self.regional_count == 0:
            raise ValueError("No regional timestamp contains a valid target value.")

        site_mae = self.absolute_error_sum[evaluated] / self.valid_count[evaluated]
        site_rmse = np.sqrt(
            self.squared_error_sum[evaluated] / self.valid_count[evaluated]
        )
        capacity = self.capacity[evaluated]

        return {
            "nMAE_percent": float(100.0 * np.mean(site_mae / capacity)),
            "nRMSE_percent": float(100.0 * np.mean(site_rmse / capacity)),
            "regional_total_nMAE_percent": float(
                100.0 * self.regional_ratio_absolute_sum / self.regional_count
            ),
            "regional_total_nRMSE_percent": float(
                100.0
                * np.sqrt(self.regional_ratio_squared_sum / self.regional_count)
            ),
        }
