"""The main file for data processing."""
import os
import warnings

from absl import app
from absl import flags

import torch

from meshnets.utils import data_processing

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', os.path.join('data', 'dataset'),
                    'Path to the folder containing the mesh files.')

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
        if not os.path.exists(os.path.join(folder, 'pressure_field.vtk')):
            warnings.warn(f'{folder} does not contain a .vtk file.')
            sim_folders.remove(folder)

    # Convert the .vtk files to graph data.
    for folder in sim_folders:
        vtk_file_path = os.path.join(folder, 'pressure_field.vtk')
        graph_data = data_processing.mesh_file_to_graph_data(vtk_file_path,
                                                             WIND_VECTOR,
                                                             load_pressure=True)

        # TODO: Will add suport here for interpolation to the original
        # mesh in a future pull request.

        torch.save(graph_data, os.path.join(folder, 'pressure_field.pt'))


if __name__ == '__main__':
    app.run(main)
