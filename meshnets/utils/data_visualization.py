"""Methods for visualizing mesh objects."""
import matplotlib.pyplot as plt

from typing import Union
from typing import Tuple

import numpy as np
import pyvista as pv


def plot_3d_graph_and_predictions(example,
                                  predictions,
                                  point_size=100,
                                  save_path=None,
                                  figsize=(12, 6)):
    node_positions = example["nodes"]

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection="3d")
    scatter1 = ax1.scatter([pos[0] for pos in node_positions],
                           [pos[1] for pos in node_positions],
                           [pos[2] for pos in node_positions],
                           c=example["wind_pressures"],
                           cmap=plt.cm.jet,
                           marker="o",
                           s=point_size)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Wind Pressures")

    ax2 = fig.add_subplot(122, projection="3d")
    scatter2 = ax2.scatter([pos[0] for pos in node_positions],
                           [pos[1] for pos in node_positions],
                           [pos[2] for pos in node_positions],
                           c=predictions,
                           cmap=plt.cm.jet,
                           marker="o",
                           s=point_size)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Predictions")

    # Add a common colorbar
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], label="Color")

    if save_path is not None:
        save_directory = os.path.dirname(save_path)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        plt.savefig(save_path)
    else:
        plt.show()


def plot_mesh(mesh_path: str,
              clim: Union[Tuple[float, float], None] = None,
              rotate_z: int = 0,
              off_screen: bool = False,
              screenshot: Union[str, None] = None) -> None:
    """Plot a mesh object with its active scalar array."""

    mesh = pv.read(mesh_path)
    mesh = mesh.rotate_z(rotate_z)
    mesh.plot(show_edges=True,
              cmap="RdBu_r",
              clim=clim,
              off_screen=off_screen,
              screenshot=screenshot)


def _validate_scalars(scalars: np.array) -> np.array:

    if scalars is None:
        raise TypeError("Argument 'scalars' is None.")

    if scalars.ndim == 1:
        return scalars
    elif scalars.ndim == 2 and scalars.shape[1] == 1:
        return scalars.flatten()
    else:
        raise ValueError(
            f"Argument 'scalars' has invalid dimensions : {scalars.shape}. "
            "Dimensions must be (None,) or (None, 1).")


def plot_mesh_with_scalars(mesh_path: str,
                           scalars: np.array,
                           clim: Union[Tuple[float, float], None] = None,
                           rotate_z: int = 0,
                           off_screen: bool = False,
                           screenshot: Union[str, None] = None) -> None:
    """Plot a mesh object with a given scalar array."""

    mesh = pv.read(mesh_path)
    mesh = mesh.rotate_z(rotate_z)

    scalars = _validate_scalars(scalars)

    mesh.plot(scalars=scalars,
              show_edges=True,
              cmap="RdBu_r",
              clim=clim,
              off_screen=off_screen,
              screenshot=screenshot)


def plot_mesh_comparison(mesh_path: str,
                         ground_truth: np.array,
                         prediction: np.array,
                         clim: Union[Tuple[float, float], None] = None,
                         rotate_z: int = 0,
                         off_screen: bool = False,
                         screenshot: Union[str, None] = None) -> None:
    """Plot a mesh object with two given scalar arrays and their difference."""

    mesh = pv.read(mesh_path)
    mesh = mesh.rotate_z(rotate_z)

    ground_truth = _validate_scalars(ground_truth)
    prediction = _validate_scalars(prediction)

    plotter = pv.Plotter(shape=(1, 3), off_screen=off_screen)

    # Groundtruth
    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth")
    plotter.add_mesh(mesh,
                     scalars=ground_truth,
                     show_edges=True,
                     cmap="RdBu_r",
                     clim=clim,
                     copy_mesh=True)

    # Prediction
    plotter.subplot(0, 1)
    plotter.add_text("Prediction")
    plotter.add_mesh(mesh,
                     scalars=prediction,
                     show_edges=True,
                     cmap="RdBu_r",
                     clim=clim,
                     copy_mesh=True)

    # Difference
    difference = prediction - ground_truth

    plotter.subplot(0, 2)
    plotter.add_text("Error")
    plotter.add_mesh(mesh,
                     scalars=difference,
                     show_edges=True,
                     cmap="RdBu_r",
                     clim=clim,
                     copy_mesh=True)
    plotter.show(screenshot=screenshot)


def plot_relative_error(mesh_path: str,
                        ground_truth: np.array,
                        prediction: np.array,
                        clim: Union[Tuple[float, float], None] = None,
                        rotate_z: int = 0,
                        off_screen: bool = False,
                        screenshot: Union[str, None] = None):
    mesh = pv.read(mesh_path)
    mesh = mesh.rotate_z(rotate_z)

    ground_truth = _validate_scalars(ground_truth)
    prediction = _validate_scalars(prediction)

    plotter = pv.Plotter(off_screen=off_screen)

    # Relative error
    difference = prediction - ground_truth
    relative_error = np.abs(difference / (ground_truth + 1e-7))

    plotter.add_text("Relative Error")
    plotter.add_mesh(mesh,
                     scalars=relative_error,
                     show_edges=True,
                     roughness=1,
                     clim=clim,
                     copy_mesh=True)

    plotter.show(screenshot=screenshot)
