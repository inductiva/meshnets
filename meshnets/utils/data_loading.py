""""Methods for loading mesh data"""

from typing import Tuple

from absl import logging
import meshio
import numpy as np


def load_triangle_mesh(obj_path: str,
                       verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load nodes and cells from a mesh file representing a triangle mesh.
    
    This method is deprecated and only works for meshes composed of triangle
    faces only."""

    mesh = meshio.read(obj_path)

    nodes = mesh.points
    cells = mesh.cells_dict['triangle']

    if verbose:
        logging.info('Nodes shape : %s', nodes.shape)
        logging.info('Cells shape : %s', nodes.shape)

    return nodes, cells


# TODO(victor): Load windspeed tuple
