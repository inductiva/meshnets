"""Load the mesh data"""

from typing import Tuple

from absl import logging
import meshio
import numpy as np

# TODO(victor): modify the loader once a data format has been decided


def load_mesh_from_obj(obj_path: str,
                       verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load nodes and cells from a .obj file representing a triangle mesh"""

    mesh = meshio.read(obj_path)

    nodes = mesh.points
    cells = mesh.cells_dict['triangle']

    if verbose:
        logging.info('Nodes shape : %s', nodes.shape)
        logging.info('Cells shape : %s', nodes.shape)

    return nodes, cells


# TODO(victor): Load pressure label
# TODO(victor): Load windspeed tuple
