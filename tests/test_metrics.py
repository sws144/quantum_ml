import numpy as np
from quantumtools import metrics


def test_gini_one():
    y_true = [1, 1, 1, 1, 1000]
    y_pred = [1, 2, 3, 4, 5]
    sample_weight = [1, 1, 1, 1, 1]

    assert (
        np.round(metrics.gini(y_true, y_pred, sample_weight), 3) == 0.995
    ), "sample value should match"


def test_ginierror_highgini():
    y_true = [100, 2, 3, 4, 1]
    y_pred = [100, 2, 3, 4, 1]
    sample_weight = [1, 1, 1, 1, 10]

    assert (
        np.round(
            metrics.ginierror(y_true, y_pred, sample_weight, split=5, plot=False), 3
        )
        == 0.869
    ), "sample value should match"


def test_ginierror_noweight():
    y_true = [100, 2, 3, 4, 1]
    y_pred = [100, 2, 3, 4, 1]

    assert (
        np.round(metrics.ginierror(y_true, y_pred, split=5, plot=False), 3) == 0.909
    ), "sample value should match"
