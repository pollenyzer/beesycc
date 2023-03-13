""" Helper module to visualize colors. """

import matplotlib.pyplot as plt
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def scatter_bgr_colors(bgr_colors, cluster_ids=None):
    """ Creates a 1x3 subplot of LAB colors.

    Args:
        bgr_colors (np.ndarray): The array of shape Nx3 that contains colors.
        cluster_ids (np.ndarray, optional): An optional array that contains
            cluster ids. Defaults to None.

    Returns:
        plt.fig, List[plt.axis, plt.axis, plt.axis]: The plot is returned.
    """
    rgba_colors = np.c_[cv2.cvtColor(bgr_colors[None], cv2.COLOR_BGR2RGB)[
        0] / 255, np.ones(len(bgr_colors))]
    lab_colors = cv2.cvtColor(bgr_colors[None], cv2.COLOR_BGR2LAB)[0]
    fig_size = 4

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1,
        ncols=3,
        sharey=True, sharex=True,
        figsize=(3*fig_size, fig_size)
    )
    ax1.set_xlim((0, 255))
    ax1.set_ylim((0, 255))
    ax1.set_xlabel("L (lightness)")
    ax1.set_ylabel("A (green-red)")
    ax2.set_xlabel("L (lightness)")
    ax2.set_ylabel("B (blue-yellow)")
    ax3.set_xlabel("A (green-red")
    ax3.set_ylabel("B (blue-yellow")

    ax1.scatter(x=lab_colors[:, 0], y=lab_colors[:, 1], c=rgba_colors, s=14)
    ax2.scatter(x=lab_colors[:, 0], y=lab_colors[:, 2], c=rgba_colors, s=14)
    ax3.scatter(x=lab_colors[:, 1], y=lab_colors[:, 2], c=rgba_colors, s=14)

    if cluster_ids is not None:
        cluster_colors = np.c_[
            np.linspace(
                [0, 0, 0], [255, 255, 255], int(np.max(cluster_ids) + 1)
            ),
            np.ones(int(np.max(cluster_ids)+1))*255
        ] / 255
        np.random.shuffle(cluster_colors[:, 0])
        np.random.shuffle(cluster_colors[:, 1])
        np.random.shuffle(cluster_colors[:, 2])

        for cluster_id in np.unique(cluster_ids).astype(np.int):
            ax1.scatter(
                x=lab_colors[cluster_ids == cluster_id, 0],
                y=lab_colors[cluster_ids == cluster_id, 1],
                color=cluster_colors[cluster_id],
                s=0.3
            )
            ax2.scatter(
                x=lab_colors[cluster_ids == cluster_id, 0],
                y=lab_colors[cluster_ids == cluster_id, 2],
                color=cluster_colors[cluster_id],
                s=0.3
            )
            ax3.scatter(
                x=lab_colors[cluster_ids == cluster_id, 1],
                y=lab_colors[cluster_ids == cluster_id, 2],
                color=cluster_colors[cluster_id],
                s=0.3
            )
    return fig, [ax1, ax2, ax3]


def scatter_bgr_colors_3d(bgr_colors):
    """ Creates a 3D scatter plot in LAB space.

    Args:
        bgr_colors (np.ndarray): The colors to plot.

    Returns:
        Tuple[plt.figure, plt.axis]: The generated 3D scatter plot.
    """
    rgba_colors = np.c_[bgr_colors[:, ::-1] / 255, np.ones(len(bgr_colors))]
    lab_colors = cv2.cvtColor(bgr_colors[None], cv2.COLOR_BGR2LAB)[0]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(
        lab_colors[:, 0],
        lab_colors[:, 1],
        lab_colors[:, 2],
        c=rgba_colors
    )
    return fig, ax
