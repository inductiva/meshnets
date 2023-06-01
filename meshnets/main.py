"""The main file to run our code

Currently, it loads mesh data from a .vtk file, defines a wind vector,
and creates a Graph object from them
"""

import os
from pathlib import Path

from absl import app, flags, logging
import torch
from torch_geometric.loader import DataLoader

from meshnets.utils import data_processing
from meshnets.utils.datasets import FromDiskGraphDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('raw_data_dir', os.path.join('data', 'vtk'),
                    'Path to the folder containing the raw data files')
flags.DEFINE_string('processed_data_dir', os.path.join('data', 'pt'),
                    'Path to the folder for the processed data files')
flags.DEFINE_bool('process_raw', False,
                  'Indicate if the raw data must be processed or not')

# Wind vector associated with the simulations
WIND_VECTOR = (10, 0, 0)


def main(_):

    logging.info('Process the raw data : %s', FLAGS.process_raw)
    if FLAGS.process_raw:
        for raw_file in os.listdir(FLAGS.raw_data_dir):

            raw_file_path = os.path.join(FLAGS.raw_data_dir, raw_file)
            processed_graph = data_processing.mesh_file_to_graph_data(
                raw_file_path, WIND_VECTOR)

            processed_file = Path(raw_file).with_suffix('.pt')
            processed_file_path = os.path.join(FLAGS.processed_data_dir,
                                               processed_file)

            torch.save(processed_graph, processed_file_path)

    dataset = FromDiskGraphDataset(FLAGS.processed_data_dir)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for batch in dataloader:
        logging.info(batch)


if __name__ == '__main__':
    app.run(main)
