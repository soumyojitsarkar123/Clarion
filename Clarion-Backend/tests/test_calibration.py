"""
Tests for Statistically Sound Probability Calibration and Calibration Metrics.
"""

import numpy as np
import pytest

from core.evaluation.calibration import (
    CalibrationMetrics,
    compute_calibration_metrics,
    PlattCalibrator,
    TemperatureCalibrator,
    IsotonicCalibrator,
)


def test_compute_calibration_metrics_perfect_calibration():
    """Verify ECE is near zero for perfectly calibrated predictions."""
    # Balanced set where predicted probabilities match true empirical rates
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # In bin [0.0, 0.1]: 5 samples, all 0 -> acc=0.0, conf=0.0 -> diff=0.0
    # In bin [0.9, 1.0]: 5 samples, all 1 -> acc=1.0, conf=1.0 -> diff=0.0
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)

    assert metrics.ece == pytest.approx(0.0, abs=1e-4)
    assert metrics.mce == pytest.approx(0.0, abs=1e-4)
    assert metrics.brier_score == pytest.approx(0.0, abs=1e-4)


def test_compute_calibration_metrics_overconfident_uncalibrated():
    """Verify ECE detects extreme overconfidence."""
    # Model predicts 0.95 confidence for negative samples
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.95] * 10)

    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)

    # All samples in bin [0.9, 1.0]: acc = 0.0, conf = 0.95 -> diff = 0.95
    assert metrics.ece == pytest.approx(0.95, abs=1e-3)
    assert metrics.mce == pytest.approx(0.95, abs=1e-3)
    assert metrics.brier_score == pytest.approx(0.95 ** 2, abs=1e-3)


def test_platt_calibrator_fitting():
    """Verify Platt scaling produces valid probabilities bounded in [0, 1]."""
    # Validation scores and labels
    val_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    val_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    calibrator = PlattCalibrator()
    calibrator.fit(val_scores, val_labels)

    assert calibrator.is_fitted is True

    test_scores = np.array([0.15, 0.50, 0.85])
    cal_probs = calibrator.calibrate(test_scores)

    assert len(cal_probs) == 3
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))
    # Monotonicity check
    assert cal_probs[0] < cal_probs[1] < cal_probs[2]


def test_temperature_calibrator_scales_logits():
    """Verify temperature scaling learns T to smooth overconfident logits."""
    val_logits = np.array([-3.0, -2.0, -1.5, 1.5, 2.0, 3.0])
    val_labels = np.array([0, 0, 1, 0, 1, 1])

    calibrator = TemperatureCalibrator()
    calibrator.fit(val_logits, val_labels)

    assert calibrator.is_fitted is True
    assert calibrator.temperature > 0.0

    test_logits = np.array([-2.0, 0.0, 2.0])
    probs = calibrator.calibrate(test_logits)

    assert len(probs) == 3
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert probs[0] < probs[1] < probs[2]


def test_isotonic_calibrator_monotonicity():
    """Verify isotonic regression produces monotonic step calibration."""
    val_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    val_labels = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    calibrator = IsotonicCalibrator()
    calibrator.fit(val_scores, val_labels)

    assert calibrator.is_fitted is True
    test_scores = np.array([0.05, 0.25, 0.55, 0.85, 1.1])
    probs = calibrator.calibrate(test_scores)

    # Check weak monotonicity
    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1]


def test_leak_free_validation_split_workflow():
    """
    Demonstrate that fitting Platt scaling strictly on validation split
    reduces test-set calibration error and Brier score without ever seeing test data.
    """
    rng = np.random.RandomState(42)

    # Validation split: true probabilities + labels + systematically overconfident model scores
    val_p = rng.uniform(0.1, 0.9, size=80)
    val_labels = (rng.uniform(size=80) < val_p).astype(int)
    val_scores = np.clip(val_p ** 0.3, 0.01, 0.99)  # Highly overconfident

    # Held-out Test split from same data generating distribution
    test_p = rng.uniform(0.1, 0.9, size=80)
    test_labels = (rng.uniform(size=80) < test_p).astype(int)
    test_scores = np.clip(test_p ** 0.3, 0.01, 0.99)

    # Baseline uncalibrated test metrics
    uncal_metrics = compute_calibration_metrics(test_labels, test_scores)

    # Fit Platt calibrator ONLY on validation set
    platt = PlattCalibrator()
    platt.fit(val_scores, val_labels)

    # Evaluate on held-out test set
    cal_test_probs = platt.calibrate(test_scores)
    cal_metrics = compute_calibration_metrics(test_labels, cal_test_probs)

    # Calibration should improve ECE and Brier score on test set
    assert cal_metrics.brier_score < uncal_metrics.brier_score
    assert cal_metrics.ece < uncal_metrics.ece

