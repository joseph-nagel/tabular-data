"""Synthetic data generation."""

import numpy as np


def make_regression_data(
    num_samples: int,
    regular_grid: bool = False,
    random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate heteroscedastic regression data.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    regular_grid : bool
        Determines whether x-values are on regular grid or sampled uniformly.
    random_seed : int | None
        Random seed.

    """

    rng = np.random.default_rng(seed=random_seed)

    # get x and calculate y
    if regular_grid:
        x = np.linspace(0, 10, num_samples)
    else:
        x = rng.uniform(0, 10, num_samples)

    y = x + np.sin(x) * x
    y /= 10

    # add heteroscedastic noise
    noise = rng.normal(loc=0, scale=0.005 * x**2, size=x.shape[0])
    y += noise

    # reshape
    x = x.reshape(-1, 1)

    return x, y
