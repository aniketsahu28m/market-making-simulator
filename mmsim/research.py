"""Research: does order book imbalance predict short-term moves in the mid price?"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mmsim.simulator import SessionResult


# Candidate signals: Observation field name -> the SessionResult series that records it.
FEATURES = {
    "queue_imbalance": "queue_imbalances",
    "imbalance": "imbalances",
    "flow_imbalance": "flow_imbalances",
}


def signal_samples(
    result: SessionResult, horizon: int, feature: str = "flow_imbalance"
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of (signal now, mid change over the next `horizon` steps) from one session.

    Samples are taken every `horizon` steps so the future windows don't overlap.
    Overlapping windows share price moves, which makes errors correlated and
    t-statistics look far better than they really are.
    """
    mids = np.array(result.mids[1:])
    signal = np.array(getattr(result, FEATURES[feature])[1:])
    idx = np.arange(0, len(mids) - horizon, horizon)
    return signal[idx], mids[idx + horizon] - mids[idx]


@dataclass(frozen=True)
class Regression:
    slope: float
    intercept: float
    slope_t_stat: float
    r_squared: float
    n: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * x


def ols(x: np.ndarray, y: np.ndarray) -> Regression:
    """Ordinary least squares of y on x with an intercept."""
    n = len(x)
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    slope = float(((x - x_mean) * (y - y_mean)).sum()) / sxx
    intercept = float(y_mean - slope * x_mean)
    residuals = y - (intercept + slope * x)
    sse = float((residuals**2).sum())
    sst = float(((y - y_mean) ** 2).sum())
    slope_se = math.sqrt(sse / (n - 2) / sxx)
    return Regression(
        slope=slope,
        intercept=intercept,
        slope_t_stat=slope / slope_se,
        r_squared=1 - sse / sst,
        n=n,
    )


def out_of_sample_r2(fit: Regression, x: np.ndarray, y: np.ndarray) -> float:
    """R² on data the model wasn't fitted on, against a forecast of zero price change.

    Positive means the signal beats assuming the price won't move. The benchmark is
    zero rather than the test-set mean, because the test-set mean isn't known in advance.
    """
    residuals = y - fit.predict(x)
    return 1 - float((residuals**2).sum()) / float((y**2).sum())


def binned_means(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Mean of x and y within equal-count bins of x, for plotting the signal."""
    order = np.argsort(x)
    bins = np.array_split(order, n_bins)
    return np.array([x[b].mean() for b in bins]), np.array([y[b].mean() for b in bins])
