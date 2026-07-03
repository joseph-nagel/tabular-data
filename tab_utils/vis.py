"""Visualization tools."""

import matplotlib.pyplot as plt
import numpy as np


def plot_data_2d(
    X,
    y,
    labels=None,
    colors=None,
    zorder=None,
    ax=None,
):
    """Plot data points with labels on a two-dim. plane."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if labels is None:
        labels = np.unique(y)

    for idx, label in enumerate(labels):
        ax.scatter(
            X[y == label, 0],
            X[y == label, 1],
            color=None if colors is None else colors[idx],
            alpha=0.7,
            edgecolors="none",
            label=f"y={label}",
            zorder=zorder,
        )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    if fig is None:
        return ax
    else:
        return fig, ax


def plot_function_2d(
    function,
    gridpoints=101,
    levels=(0.1, 0.3, 0.5, 0.7, 0.9),
    x_limits=None,
    y_limits=None,
    z_limits=None,
    colorbar=True,
    zorder=None,
    ax=None,
):
    """Plot a function of two features on the plane."""

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if x_limits is None:
        x_limits = ax.get_xlim()

    if y_limits is None:
        y_limits = ax.get_ylim()

    # create inputs
    x_values = np.linspace(*x_limits, num=gridpoints)
    y_values = np.linspace(*y_limits, num=gridpoints)

    (x_grid, y_grid) = np.meshgrid(x_values, y_values)
    xy_values = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)

    # compute outputs
    z_values = function(xy_values)
    z_grid = z_values.reshape(x_grid.shape)

    # plot function
    im1 = ax.imshow(
        z_grid,
        origin="lower",
        extent=(*x_limits, *y_limits),
        interpolation="bicubic",
        cmap="Greys",
        clim=z_limits,
        alpha=0.4,
        zorder=zorder,
    )

    # im1 = ax.contourf(
    #     x_grid,
    #     y_grid,
    #     z_grid,
    #     levels=10,
    #     cmap="Greys",
    #     alpha=0.4,
    #     zorder=zorder,
    # )

    im2 = ax.contour(
        x_grid,
        y_grid,
        z_grid,
        levels=levels,
        colors="black",
        alpha=0.6,
        zorder=zorder,
    )

    if colorbar:
        plt.colorbar(im1, ax=ax)

    plt.clabel(im2, fmt="%1.2f")

    if fig is None:
        return ax
    else:
        return fig, ax
