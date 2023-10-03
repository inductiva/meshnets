"""The main file for data processing."""
import os
import warnings

from absl import app
from absl import flags

import torch
import pyvista as pv

from meshnets.utils import data_processing

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
flags.DEFINE_string('torch_graph_name', 'pressure_field.pt',
                    'Name of the torch graph.')

flags.DEFINE_float('tolerance', 1.5, 'Tolerance for the mesh sampling.')

# TODO: This will be deprecated soon as we will start training with
# varying wind speeds. At the moment this is here just for support
# while not metadata is added to the dataset containing the wind
# speed.
WIND_VECTOR = [30., 0, 0]


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
    for folder in sim_folders:
        if not os.path.exists(
                os.path.join(folder, FLAGS.original_pressure_field_name)):
            warnings.warn(f'{folder} does not contain a .vtk file.')
            sim_folders.remove(folder)

    # Convert the .vtk files to graph data.
    for folder in sim_folders:
        openfoam_mesh_path = os.path.join(folder,
                                          FLAGS.original_pressure_field_name)
        original_mesh_path = os.path.join(folder, FLAGS.original_mesh_name)

        openfoam_mesh = pv.read(openfoam_mesh_path)
        original_mesh = pv.read(original_mesh_path)

        interpolated_mesh = original_mesh.sample(openfoam_mesh,
                                                 tolerance=FLAGS.tolerance)
        interpolated_mesh_path = os.path.join(
            folder, FLAGS.interpolated_pressure_field_name)
        interpolated_mesh.save(interpolated_mesh_path)

        graph_data = data_processing.mesh_file_to_graph_data(
            interpolated_mesh_path, WIND_VECTOR, load_pressure=True)

        torch.save(graph_data, os.path.join(folder, FLAGS.torch_graph_name))


if __name__ == '__main__':
    app.run(main)
