""""Methods for loading mesh data"""

from typing import Union
from typing import Tuple

from absl import logging
import meshio
from meshio._mesh import Mesh
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

    if get_pressure:
        # Reshape from (number_of_points,) to (number_of_points, 1)
        # to be consistent with cases of multiple node features
        # TODO(victor): decide if array reshape is necessary
        pressure = edge_mesh.get_array('p', preference='point').reshape(-1, 1)
        if verbose:
            logging.info('Pressure shape : %s', pressure.shape)

        return nodes, edge_list, pressure
    else:
        return nodes, edge_list


def edges_from_meshio_mesh(mesh: Mesh) -> np.ndarray:
    """Produce an array containing all the edges in a meshio Mesh object.
    
    This array can include duplicates."""

    edges_list = []
    for cell in mesh.cells:
        cell_faces = cell.data
        shape_vertices_nb = cell_faces.shape[1]

        for i in range(shape_vertices_nb - 1):
            edges_list.append(cell_faces[:, i:(i + 2)])
        if shape_vertices_nb > 2:
            # Get the edge between the first and last node of the faces
            edges_list.append(cell_faces[:, ::(shape_vertices_nb - 1)])

        # Easier to understand but runs approximately 10 times slower
        # for i in range(shape_vertices_nb - 1):
        #     edges_list.append(cell_faces[:, [i, i + 1]])
        # if shape_vertices_nb > 2:
        #     edges_list.append(cell_faces[:, [0, -1]])

    return np.concatenate(edges_list, axis=0)


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

    # On some files, meshio returns arrays in big endian byte order
    # which is not supported by torch. This changes the array to
    # small endian without modfying the values in it.
    # Behavior observed on .vtk files
    nodes = mesh.points
    if nodes.dtype.byteorder == '>':
        nodes = nodes.byteswap().newbyteorder('<')

    edges = edges_from_meshio_mesh(mesh)

    if verbose:
        logging.info('Nodes shape : %s', nodes.shape)
        logging.info('Edge list shape : %s', edges.shape)

    if get_pressure:
        # Same behavior as mesh.points
        pressure = mesh.point_data['p']
        if pressure.dtype.byteorder == '>':
            pressure = pressure.byteswap().newbyteorder('<')
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
