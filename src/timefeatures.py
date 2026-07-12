import numpy as np
import pandas as pd


def time_features(dates):
    """Return normalized month, day, weekday, and hour features."""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.stack(
        [
            (index.month.to_numpy() - 1) / 11.0 - 0.5,
            (index.day.to_numpy() - 1) / 30.0 - 0.5,
            index.dayofweek.to_numpy() / 6.0 - 0.5,
            index.hour.to_numpy() / 23.0 - 0.5,
        ],
        axis=-1,
    ).astype(np.float32)
