"""This module contains useful functions for other modules.
"""

import numpy as np


def poly(x, order=3):
    """Evaluates the different powers of an input vector.

    The input vector is evaluated element-wise
    to the power 1, 2, ..., `order`. The resulting vectors
    are then concatenated and returned.

    Parameters
    ----------
    x: array_like
        The input vector, of shape `(n, 1)`.
    order: int
        The maximum order to which the powers of
        `x`are computed.

    Returns
    -------
    x_out: array_like
        The concatenation of all
        the powers of `x`, of shape `(n, order)`.

    """
    x_out = x
    for i in range(2, order + 1):
        x_out = np.concatenate((x_out, np.power(x, i)), axis=1)
    return x_out


def paths(hidden_layers=2, dropout_rate=0.0):
    """File paths for model weights and metrics from model parameters.

    The input vector is evaluated element-wise
    to the power 1, 2, ..., `order`. The resulting vectors
    are then concatenated and returned.

    Parameters
    ----------
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.

    Returns
    -------
    path_model: string
        The file path of the model weights.
    path_metrics: string
        The file path of the model metrics computed during training.

    """

    base_name = (
        "saved_models/fmnist_mlp_hidden="
        + str(hidden_layers)
        + "_dropout_rate="
        + str(dropout_rate)
    )
    path_weights = base_name + ".pth"
    path_metrics = base_name + "_metrics.csv"
    return path_weights, path_metrics
