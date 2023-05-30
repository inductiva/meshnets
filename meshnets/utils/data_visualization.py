"""Load an object and visualize it using pyvista."""

from absl import logging
import pyvista as pv


def plot_data(obj_path: str, verbose: bool = False) -> None:
    "Plot a mesh object with pyvista."

    if verbose:
        logging.info("Loading the object from %s", obj_path)
    mesh = pv.read(obj_path)

    if verbose:
        logging.info("Plotting the object")
    mesh.plot(show_edges=True)
