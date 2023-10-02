""""Methods for loading mesh data"""

from typing import Union
from typing import Tuple

from absl import logging
import numpy as np
import pyvista as pv


def load_edge_mesh_pv(
    obj_path: str,
    load_pressure: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """Extract node coordinates, edges and pressure at nodes from a mesh file
    using pyvista.
    
    If `load_pressure=False`, the returned pressure will be None."""

    mesh = pv.read(obj_path)

    # Extract all the internal/external edges of the dataset as PolyData.
    # This produces a full wireframe representation of the input dataset.
    # WARNING: `use_all_points=False` discards duplicates and unused nodes,
    # but remeshing should be done prior on the mesh file, not here
    edge_mesh = mesh.extract_all_edges(use_all_points=True)

    nodes = edge_mesh.points

    # Returns the edges as an array with format :
    # [2, point_id_1, point_id_2, 2, point_id_3, point_id_4, 2, ...]
    # Reshape it to (number_of_edges, 3) and remove the first column
    # containing only 2s
    edge_list = edge_mesh.lines.reshape((-1, 3))[:, 1:]

    if verbose:
        logging.info('Nodes shape : %s', nodes.shape)
        logging.info('Edge list shape : %s', edge_list.shape)

    if load_pressure:
        # Reshape from (number_of_points,) to (number_of_points, 1)
        # to be consistent with cases of multiple node features
        # TODO(victor): decide if array reshape is necessary
        pressure = edge_mesh.get_array('p', preference='point').reshape(-1, 1)
        if verbose:
            logging.info('Pressure shape : %s', pressure.shape)

        return nodes, edge_list, pressure
    else:
        if verbose:
            logging.info('Pressure is : %s', None)
        return nodes, edge_list, None
