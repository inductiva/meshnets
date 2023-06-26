"""The main file for data processing.

It processes mesh files from a given directory and produce graph data
as '.pt' files in another directory.
"""

import os
from pathlib import Path

from absl import app
from absl import flags
from absl import logging
import torch

from meshnets.utils import data_processing

# Wind vector associated with the simulations
WIND_VECTOR = (10, 0, 0)

FLAGS = flags.FLAGS

# Path flags
flags.DEFINE_string('mesh_data_dir', os.path.join('data', 'vtk'),
                    'Path to the folder containing the mesh files.')
flags.DEFINE_string('processed_data_dir', os.path.join('data', 'pt'),
                    'Path to the folder for the processed data files.')

flags.DEFINE_bool('get_pressure', True,
                  'Whether or not to extract pressure from the mesh files.')

flags.DEFINE_bool(
    'verbose', False,
    'Wether or not the data processing should log processing information.')


def main(_):
    """Process mesh data files to encoded graph data files.
    
    Loop through a directory containing mesh files (e.g. `.vtk`),
    produce a graph representation of each mesh and save the graph as
    a `.pt` file in the processed data directory."""

    mesh_data_dir = FLAGS.mesh_data_dir
    processed_data_dir = FLAGS.processed_data_dir
    get_pressure = FLAGS.get_pressure
    verbose = FLAGS.verbose

    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    for mesh_file in os.listdir(mesh_data_dir):

        mesh_file_path = os.path.join(mesh_data_dir, mesh_file)
        processed_graph = data_processing.mesh_file_to_graph_data(
            mesh_file_path,
            WIND_VECTOR,
            get_pressure=get_pressure,
            verbose=verbose)

        processed_file = Path(mesh_file).with_suffix('.pt')
        processed_file_path = os.path.join(processed_data_dir, processed_file)

        if verbose:
            logging.info('Saving graph object to : %s', processed_file_path)
        torch.save(processed_graph, processed_file_path)


if __name__ == '__main__':
    app.run(main)
