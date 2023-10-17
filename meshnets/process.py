"""The main file for data processing."""
import os
import json
import warnings

from absl import app
from absl import flags

import torch
import pyvista as pv
import numpy as np

from meshnets.utils import data_processing, data_loading

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', os.path.join('data', 'dataset'),
                    'Path to the folder containing the mesh files.')

flags.DEFINE_string('original_pressure_field_name', 'pressure_field.vtk',
                    'Name of the original pressure field.')
flags.DEFINE_string('interpolated_pressure_field_name',
                    'pressure_field_interpolated.vtk',
                    'Name of the interpolated pressure field.')
flags.DEFINE_string('original_mesh_name', 'object.obj',
                    'Name of the original mesh.')
flags.DEFINE_string('wind_vector_name', 'flow_velocity.json',
                    'The name of the file containing the wind vector.')
flags.DEFINE_string('edge_index_name', 'edge_index.npy',
                    'Name of the edge index file.')
flags.DEFINE_string('edge_features_name', 'edge_features.npy',
                    'Name of the edge attribute file.')
flags.DEFINE_string('node_features_name', 'node_features.npy',
                    'Name of the node attribute file.')
flags.DEFINE_string('wind_pressures_name', 'wind_pressures.npy',
                    'Name of the wind pressures file.')
flags.DEFINE_string('torch_graph_name', 'pressure_field.pt',
                    'Name of the torch graph.')
flags.DEFINE_list(
    'default_wind_vector', [30., 0, 0],
    'Default wind vector if none is present in the simulation folder.')

flags.DEFINE_float('tolerance', 1.5, 'Tolerance for the mesh sampling.')


def main(_):
    """
    Converts the .vtk files in the data_dir to graph data and saves them
    as .pt files.

    Assumes that the data_dir contains folders with the following structure:

    data_dir
        - 199834348
            - pressure_field.vtk
        - 129383949
            - pressure_field.vtk

    """
    sim_folders = [
        os.path.join(FLAGS.data_dir, f) for f in os.listdir(FLAGS.data_dir)
    ]

    # Remove folders that do not contain a .vtk file.
    for folder in sim_folders.copy():
        if not os.path.exists(
                os.path.join(folder, FLAGS.original_pressure_field_name)):
            warnings.warn(f'{folder} does not contain a .vtk file.')
            sim_folders.remove(folder)

    for folder in sim_folders:
        openfoam_mesh_path = os.path.join(folder,
                                          FLAGS.original_pressure_field_name)
        original_mesh_path = os.path.join(folder, FLAGS.original_mesh_name)

        openfoam_mesh = pv.read(openfoam_mesh_path)
        original_mesh = pv.read(original_mesh_path)
        original_mesh = original_mesh.clean()

        interpolated_mesh = original_mesh.sample(openfoam_mesh,
                                                 tolerance=FLAGS.tolerance)
        interpolated_mesh_path = os.path.join(
            folder, FLAGS.interpolated_pressure_field_name)
        interpolated_mesh.save(interpolated_mesh_path)

        nodes, edges, pressures = data_loading.load_edge_mesh_pv(
            interpolated_mesh_path, load_pressure=True)

        edge_index, edge_features =\
            data_processing.make_edge_index_and_features(nodes, edges)

        wind_vector_path = os.path.join(folder, FLAGS.wind_vector_name)
        if os.path.exists(wind_vector_path):
            with open(wind_vector_path, encoding='utf-8') as f:
                wind_vector = json.load(f)
        else:
            wind_vector = FLAGS.default_wind_vector
        node_features = data_processing.make_node_features(nodes, wind_vector)

        wind_pressures = np.array(pressures, dtype=np.float32)

        # Save as numpy arrays.
        np.save(os.path.join(folder, FLAGS.edge_index_name), edge_index)
        np.save(os.path.join(folder, FLAGS.edge_features_name), edge_features)
        np.save(os.path.join(folder, FLAGS.node_features_name), node_features)
        np.save(os.path.join(folder, FLAGS.wind_pressures_name), wind_pressures)

        # TODO(augusto): Remove this in the future. Leaving it here for
        # compatibility with the old code.
        graph_data = data_processing.mesh_file_to_graph_data(
            interpolated_mesh_path, wind_vector, load_pressure=True)
        torch.save(graph_data, os.path.join(folder, FLAGS.torch_graph_name))


if __name__ == '__main__':
    app.run(main)
