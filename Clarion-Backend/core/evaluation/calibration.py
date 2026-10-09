"""
Statistically Sound Probability Calibration for Relation Validation.

Implements Platt Scaling, Temperature Scaling, and Isotonic Regression
along with standard calibration metrics (Expected Calibration Error, Maximum
Calibration Error, Brier Score) under strict train/val/test data isolation.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Standard evaluation metrics for probabilistic calibration."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Mean Squared Error to binary truth
    bin_accuracies: List[float] = field(default_factory=list)
    bin_confidences: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier_score": round(self.brier_score, 4),
            "bin_accuracies": [round(x, 4) for x in self.bin_accuracies],
            "bin_confidences": [round(x, 4) for x in self.bin_confidences],
            "bin_counts": list(self.bin_counts),
        }


def compute_calibration_metrics(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """
    Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and Brier score across binned predictions.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    MCE = max_{m=1}^M |acc(B_m) - conf(B_m)|
    Brier = (1 / N) * sum_{i=1}^N (p_i - y_i)^2
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0 or len(y_t) != len(y_p):
        raise ValueError("y_true and y_prob must be non-empty and equal length.")

    # Clip to valid probability bounds [0, 1]
    y_p = np.clip(y_p, 0.0, 1.0)

    # Brier score
    brier = float(np.mean((y_p - y_t) ** 2))

    # Binning
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accs = []
    bin_confs = []
    bin_cnts = []
    ece = 0.0
    mce = 0.0
    n = len(y_t)

    for lower, upper in zip(bin_lowers, bin_uppers):
        if upper == 1.0:
            in_bin = (y_p >= lower) & (y_p <= upper)
        else:
            in_bin = (y_p >= lower) & (y_p < upper)

        count = int(np.sum(in_bin))
        bin_cnts.append(count)

        if count > 0:
            acc = float(np.mean(y_t[in_bin]))
            conf = float(np.mean(y_p[in_bin]))
            abs_diff = abs(acc - conf)

            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (count / n) * abs_diff
            mce = max(mce, abs_diff)
        else:
            bin_accs.append(0.0)
            bin_confs.append((lower + upper) / 2.0)

    return CalibrationMetrics(
        ece=round(float(ece), 4),
        mce=round(float(mce), 4),
        brier_score=round(brier, 4),
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_cnts,
    )


class PlattCalibrator:
    """
    Parametric sigmoid calibration (Platt scaling).
    Fits P(y = 1 | s) = 1 / (1 + exp(-(A * s + B))) on validation scores.
    """

    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        self.is_fitted = False

    def fit(self, scores: Union[List[float], np.ndarray], y_val: Union[List[int], np.ndarray]) -> "PlattCalibrator":
        X = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = np.asarray(y_val, dtype=int)
        if len(np.unique(y)) < 2:
            logger.warning("Platt calibration requires at least 2 distinct classes in validation set.")
            self.is_fitted = False
            return self

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, scores: Union[List[float], np.ndarray]) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=float)
        if not self.is_fitted:
            # Fall back to clipping uncalibrated scores to [0, 1]
            clipped = np.clip(scores_arr, 0.0, 1.0)
            return np.vstack([1.0 - clipped, clipped]).T

        X = scores_arr.reshape(-1, 1)
        return self.model.predict_proba(X)

    def calibrate(self, scores: Union[List[float], np.ndarray]) -> np.ndarray:
        """Return 1D array of calibrated P(Valid)."""
        return self.predict_proba(scores)[:, 1]


class TemperatureCalibrator:
    """
    Temperature scaling for uncalibrated logits / confidence scores:
    p = sigmoid(logit / T) where T > 0 is learned on validation data by minimizing NLL.
    """

    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False

    def fit(self, logits: Union[List[float], np.ndarray], y_val: Union[List[int], np.ndarray]) -> "TemperatureCalibrator":
        z = np.asarray(logits, dtype=float)
        y = np.asarray(y_val, dtype=float)

        if len(np.unique(y)) < 2:
            self.temperature = 1.0
            self.is_fitted = False
            return self

        def nll(T):
            if T <= 1e-4:
                return 1e9
            scaled = z / T
            # Numerically stable log-loss
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            loss = -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
            return loss

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        self.is_fitted = True
        return self

    def calibrate(self, logits: Union[List[float], np.ndarray]) -> np.ndarray:
        z = np.asarray(logits, dtype=float)
        scaled = z / max(1e-4, self.temperature)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
        return np.clip(probs, 0.0, 1.0)


class IsotonicCalibrator:
    """
    Non-parametric monotonic step calibration using isotonic regression.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, scores: Union[List[float], np.ndarray], y_val: Union[List[int], np.ndarray]) -> "IsotonicCalibrator":
        X = np.asarray(scores, dtype=float)
        y = np.asarray(y_val, dtype=float)
        if len(X) < 4:
            self.is_fitted = False
            return self

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def calibrate(self, scores: Union[List[float], np.ndarray]) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=float)
        if not self.is_fitted:
            return np.clip(scores_arr, 0.0, 1.0)
        return np.clip(self.model.predict(scores_arr), 0.0, 1.0)
