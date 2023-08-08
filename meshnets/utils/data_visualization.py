"""Methods for visualizing mesh objects."""

from typing import Union
from typing import Tuple
import warnings

import numpy as np
import pyvista as pv


def plot_mesh(mesh_path: str,
              clim: Union[Tuple[float, float], None] = None,
              rot_z: int = 0,
              off_screen: bool = False,
              screenshot: Union[str, None] = None) -> None:
    """Plot a mesh object with its active scalar array."""

    mesh = pv.read(mesh_path)
    mesh = mesh.rotate_z(rot_z)
    mesh.plot(show_edges=True,
              cmap='RdBu_r',
              clim=clim,
              off_screen=off_screen,
              screenshot=screenshot)


def plot_mesh_with_scalars(mesh_path: str,
                           scalars: np.array,
                           clim: Union[Tuple[float, float], None] = None,
                           rot_z: int = 0,
                           off_screen: bool = False,
                           screenshot: Union[str, None] = None) -> None:
    """Plot a mesh object with a given scalar array."""

    mesh = pv.read(mesh_path)
    mesh = mesh.rotate_z(rot_z)

    if scalars is None:
        warnings.warn(
            "Argument 'scalars' is None. Displaying with active scalar array.")

    mesh.plot(scalars=scalars,
              show_edges=True,
              cmap='RdBu_r',
              clim=clim,
              off_screen=off_screen,
              screenshot=screenshot)


def plot_mesh_comparison(mesh_path: str,
                         scalars_1: np.array,
                         scalars_2: np.array,
                         clim: Union[Tuple[float, float], None] = None,
                         rot_z: int = 0,
                         off_screen: bool = False,
                         screenshot: Union[str, None] = None) -> None:
    """Plot a mesh object with two given scalar arrays and their difference."""

    mesh = pv.read(mesh_path)
    mesh = mesh.rotate_z(rot_z)

    plotter = pv.Plotter(shape=(1, 3), off_screen=off_screen)

    # Groundtruth
    plotter.subplot(0, 0)
    plotter.add_text('Ground Truth')
    plotter.add_mesh(mesh,
                     scalars=scalars_1,
                     show_edges=True,
                     cmap='RdBu_r',
                     clim=clim,
                     copy_mesh=True)

    # Prediction
    plotter.subplot(0, 1)
    plotter.add_text('Prediction')
    plotter.add_mesh(mesh,
                     scalars=scalars_2,
                     show_edges=True,
                     cmap='RdBu_r',
                     clim=clim,
                     copy_mesh=True)

    # Difference
    difference = scalars_2 - scalars_1

    plotter.subplot(0, 2)
    plotter.add_text('Error')
    plotter.add_mesh(mesh,
                     scalars=difference,
                     show_edges=True,
                     cmap='RdBu_r',
                     clim=clim,
                     copy_mesh=True)

    plotter.show(screenshot=screenshot)
