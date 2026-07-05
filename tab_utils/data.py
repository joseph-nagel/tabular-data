"""Synthetic data generation."""

import numpy as np


def make_regression_data(
    num_samples: int,
    regular_grid: bool = False,
    random_seed: int | None = None,
    x_limits: tuple[float | int, float | int] = (0, 10),
    noise_scale: float = 0.05,
    y_scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate regression data with heteroscedastic noise.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    regular_grid : bool
        Determines whether x-values are evenly spaced or uniformly sampled.
    random_seed : int | None
        Random seed.
    x_limits : tuple[float | int, float | int]
        Minimum and maximum value of x.
    noise_scale : float
        Noise scaling factor.
    y_scale : float
        Final scaling for y.

    """

    rng = np.random.default_rng(seed=random_seed)

    # create x
    if regular_grid:
        x = np.linspace(*x_limits, num_samples)
    else:
        x = rng.uniform(*x_limits, num_samples)

    # calculate y
    y = x + np.sin(x) * x
    if noise_scale > 0:
        y += rng.normal(loc=0, scale=noise_scale * x**2, size=num_samples)
    y *= y_scale

    # reshape x
    x = x.reshape(-1, 1)

    return x, y
