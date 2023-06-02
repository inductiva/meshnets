"""The main file to run our code

Currently, it processes mesh data to graph data and creates training
and testing datasets and dataloaders from the processed '.pt' files.
"""

import os

from absl import app, flags, logging
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from meshnets.utils import data_processing
from meshnets.utils.datasets import FromDiskDataset

FLAGS = flags.FLAGS
flags.DEFINE_string('mesh_data_dir', os.path.join('data', 'vtk'),
                    'Path to the folder containing the mesh files')
flags.DEFINE_string('processed_data_dir', os.path.join('data', 'pt'),
                    'Path to the folder for the processed data files')
flags.DEFINE_bool('process_meshes', True,
                  'Indicate if the mesh data must be processed or not')

# Wind vector associated with the simulations
WIND_VECTOR = (10, 0, 0)


def main(_):

    torch.manual_seed(21)

    logging.info('Process the mesh data : %s', FLAGS.process_meshes)
    if FLAGS.process_meshes:
        data_processing.mesh_dataset_to_graph_dataset(FLAGS.mesh_data_dir,
                                                      FLAGS.processed_data_dir,
                                                      WIND_VECTOR,
                                                      get_pressure=True,
                                                      verbose=False)

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
