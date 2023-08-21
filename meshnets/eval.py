"""The main file for model evaluation."""

import os

from absl import app
from absl import flags
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from meshnets.utils.datasets import FromDiskGeometricDataset
from meshnets.utils import model_loading

FLAGS = flags.FLAGS

flags.DEFINE_integer('random_seed', 21,
                     'The seed to initialize the random number generator.')

flags.DEFINE_string('data_dir', os.path.join('data', 'dataset'),
                    'Path to the folder containing the mesh files.')

flags.DEFINE_float('train_split', 0.9,
                   'The fraction of the dataset used for traing.')
flags.DEFINE_float('validation_split', 0.1,
                   'The fraction of the dataset used for validation.')

flags.DEFINE_string('tracking_uri', None,
                    'The tracking URI for the MLFlow experiments.')

flags.DEFINE_string('run_id', None, 'The run id of the experiment to load.')

flags.DEFINE_integer('checkpoint', 0,
                     'The checkpoint to load from the experiment.')


def main(_):

    random_seed = FLAGS.random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    dataset = FromDiskGeometricDataset(FLAGS.data_dir)
    _, validation_dataset = random_split(
        dataset, [FLAGS.train_split, FLAGS.validation_split])

    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=8,
                                       shuffle=False)

    wrapper = model_loading.load_model_from_mlflow(FLAGS.tracking_uri,
                                                   FLAGS.run_id,
                                                   FLAGS.checkpoint)

    trainer = pl.Trainer()
    results = trainer.validate(wrapper, dataloaders=validation_dataloader)

    return results


if __name__ == '__main__':
    app.run(main)
