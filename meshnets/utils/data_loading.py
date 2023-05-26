""""Methods for loading mesh data"""

from typing import Union
from typing import Tuple

from absl import logging
import meshio
import numpy as np
import pyvista as pv


def load_edge_mesh_pv(
    obj_path: str,
    get_pressure: bool = True,
    verbose: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                            np.ndarray]]:
    """Extract node coordinates, edges and pressure at nodes from a mesh file
    using pyvista."""

    mesh = pv.read(obj_path)
    # `use_all_points=False` discards duplicates and renumber the nodes
    # TODO(victor): verify that node pressure array is modified accordingly
    # WARNING: the remeshing should be done prior on the mesh file, not here
    edge_mesh = mesh.extract_all_edges(use_all_points=True)

    nodes = edge_mesh.points
    edge_list = edge_mesh.lines.reshape((-1, 3))[:, 1:]

    if verbose:
        logging.info('Nodes shape : %s', nodes.shape)
        logging.info('Edge list shape : %s', edge_list.shape)

    if get_pressure:
        pressure = edge_mesh.get_array('p', preference='point')
        if verbose:
            logging.info('Pressure shape : %s', pressure.shape)

        return nodes, edge_list, pressure
    else:
        return nodes, edge_list


def load_edge_mesh_meshio(
    obj_path: str,
    get_pressure: bool = True,
    verbose: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                            np.ndarray]]:
    """Extract node coordinates, edges and pressure at nodes from a mesh file
    using meshio.
    
    Edges can include duplicates.
    """

    mesh = meshio.read(obj_path)
    # `byteswap().newbyteorder('<')` is required because the mesh points array
    # is sometimes in big endian byte order, which is not supported by torch
    # Behavior observed on .vtk files
    # TODO: this should probably be avoided at file level
    nodes = mesh.points.byteswap().newbyteorder('<')

    # extract all the edges in the mesh
    edges_list = []
    for cell in mesh.cells:
        cell_faces = cell.data
        shape_vertices_nb = cell_faces.shape[1]

        for i in range(shape_vertices_nb - 1):
            edges_list.append(cell_faces[:, i:(i + 2)])
        if shape_vertices_nb > 2:
            # get the edge between the first and last node of the faces
            edges_list.append(cell_faces[:, ::(shape_vertices_nb - 1)])

        # a bit easier to read but runs approximately 10 times slower
        # for i in range(shape_vertices_nb - 1):
        #     edges_list.append(cell_faces[:, [i, i + 1]])
        # if shape_vertices_nb > 2:
        #     edges_list.append(cell_faces[:, [0, -1]])

    edges = np.concatenate(edges_list, axis=0)

    if verbose:
        logging.info('Nodes shape : %s', nodes.shape)
        logging.info('Edge list shape : %s', edges.shape)

    if get_pressure:
        pressure = mesh.point_data['p']
        if verbose:
            logging.info('Pressure shape : %s', pressure.shape)

        return nodes, edges, pressure
    else:
        return nodes, edges


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
