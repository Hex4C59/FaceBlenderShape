from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class MatplotlibPointViewer:
    def __init__(self) -> None:
        fig = plt.figure("Face Blender Shape Mesh", figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        all_points, = ax.plot(
            np.zeros(8800),
            np.zeros(8800),
            np.zeros(8800),
            ".",
            color="black",
            markersize=1,
        )
        lip_points, = ax.plot(
            np.zeros(76),
            np.zeros(76),
            np.zeros(76),
            ".",
            color="red",
            markersize=8,
        )
        ax.set(xlim=(-10, 10), ylim=(-10, 10), zlim=(0, 10))
        ax.grid(False)
        ax.view_init(elev=115, azim=0, roll=90)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.set_axis_off()
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")])

        self.figure = fig
        self.axis = ax
        self.all_points = all_points
        self.lip_points = lip_points

    def update_all(self, keypoints: np.ndarray) -> None:
        self._update_points(self.all_points, keypoints)

    def update_lip(self, keypoints: np.ndarray) -> None:
        self._update_points(self.lip_points, keypoints)

    @staticmethod
    def _update_points(plotter, keypoints: np.ndarray) -> None:
        plotter.set_xdata(keypoints[:, 0])
        plotter.set_ydata(keypoints[:, 1])
        plotter.set_3d_properties(keypoints[:, 2])
        plt.pause(0.01)
