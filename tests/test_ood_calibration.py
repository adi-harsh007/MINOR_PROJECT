"""The OOD calibration script must not install a gate that rejects real lesions.

Fitting Mahalanobis distance in EfficientNet-B3's 1536-dimensional feature space
needs far more images than dimensions. Fitted on the 21 bundled samples the gate
rejected the held-out lesion 21 times out of 21, at distances around 1e6 against
a cutoff of 18 - while the script's own self-report said 4.8%, because it
measured false rejects on the images it had just fitted to.

An OOD gate that rejects everything scores a perfect OOD rejection rate. These
tests keep the script from ever installing one.
"""
import numpy as np
import pytest

calibrate = pytest.importorskip("scripts.calibrate_ood")


def gaussian_features(n, dim, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim))


def test_holdout_split_is_disjoint_and_deterministic():
    features = gaussian_features(100, 8)
    holdout_a, fit_a = calibrate.split_holdout(features, 0.2)
    holdout_b, fit_b = calibrate.split_holdout(features, 0.2)

    assert holdout_a.shape[0] == 20
    assert fit_a.shape[0] == 80
    assert np.array_equal(holdout_a, holdout_b)     # rerunning says the same thing
    assert np.array_equal(fit_a, fit_b)


def test_holdout_split_degrades_safely_when_there_is_nothing_to_hold_out():
    features = gaussian_features(3, 8)
    holdout, fit = calibrate.split_holdout(features, 0.01)
    assert holdout.shape[0] == 0
    assert fit.shape[0] == 3


def test_a_gate_fitted_on_too_few_samples_rejects_unseen_in_distribution_data():
    """The measurement that makes the guard necessary.

    Far fewer samples than dimensions leaves the covariance rank-deficient, the
    pseudo-inverse explodes along every unconstrained direction, and points from
    the very same distribution land astronomically outside the cutoff.
    """
    dim = 256
    fitted = gaussian_features(20, dim, seed=1)
    unseen = gaussian_features(20, dim, seed=2)      # same distribution

    mean, inv_cov, cutoff, _ = calibrate.fit_mahalanobis(fitted, 99.0)
    centered = unseen - mean
    distances = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)

    assert (distances > cutoff).mean() == 1.0
    assert distances.min() > cutoff * 100


def test_a_gate_fitted_on_ample_samples_accepts_unseen_in_distribution_data():
    """The same code is sound once there are many more samples than dimensions."""
    dim = 8
    fitted = gaussian_features(4000, dim, seed=1)
    unseen = gaussian_features(1000, dim, seed=2)

    mean, inv_cov, cutoff, _ = calibrate.fit_mahalanobis(fitted, 99.0)
    centered = unseen - mean
    distances = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)

    assert (distances > cutoff).mean() < 0.05


def test_minimum_holdout_is_large_enough_to_mean_something():
    assert calibrate.MIN_HOLDOUT >= 25
