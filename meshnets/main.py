"""The main file to run our code

Currently, it processes mesh data to graph data and creates training
and testing datasets and dataloaders from the processed '.pt' files.
"""

import os
from pathlib import Path

from absl import app, flags, logging
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from meshnets.utils import data_processing
from meshnets.utils.datasets import FromDiskDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('raw_data_dir', os.path.join('data', 'vtk'),
                    'Path to the folder containing the raw data files')
flags.DEFINE_string('processed_data_dir', os.path.join('data', 'pt'),
                    'Path to the folder for the processed data files')
flags.DEFINE_bool('process_raw', True,
                  'Indicate if the raw data must be processed or not')

# Wind vector associated with the simulations
WIND_VECTOR = (10, 0, 0)


def main(_):

    torch.manual_seed(21)

    logging.info('Process the raw data : %s', FLAGS.process_raw)
    if FLAGS.process_raw:
        for raw_file in os.listdir(FLAGS.raw_data_dir):

            raw_file_path = os.path.join(FLAGS.raw_data_dir, raw_file)
            processed_graph = data_processing.mesh_file_to_graph_data(
                raw_file_path, WIND_VECTOR, get_pressure=True, verbose=False)

            processed_file = Path(raw_file).with_suffix('.pt')
            processed_file_path = os.path.join(FLAGS.processed_data_dir,
                                               processed_file)

            torch.save(processed_graph, processed_file_path)

    dataset = FromDiskDataset(FLAGS.processed_data_dir)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    # If label 'y' is absent (e.g. for inference), use `exclude_keys=['y']`
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=3,
        shuffle=True,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    for batch in train_dataloader:
        logging.info('Training batch: %s', batch)
    for batch in test_dataloader:
        logging.info('Testing batch: %s', batch)


if __name__ == '__main__':
    app.run(main)
