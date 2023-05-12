"""Load the mesh data"""

from absl import logging
import meshio
import numpy as np
from typing import Tuple

# TODO(victor): modify the loader once a data format has been decided

def load_mesh_from_obj(obj_path: str, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    mesh = meshio.read(obj_path)

    nodes = mesh.points
    cells = mesh.cells_dict['triangle']

    if verbose:
        logging.debug('Shape of nodes :', nodes.shape)
        logging.debug('Shape of cells :', cells.shape)

    return nodes, cells

# TODO(victor): Load pressure label
# TODO(victor): Load windspeed tuple