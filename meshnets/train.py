"""The main file for model training."""

import os

from absl import app
from absl import flags
import torch
from torch.utils.data import random_split

from meshnets.utils import model_training
from meshnets.utils.datasets import FromDiskGeometricDataset

FLAGS = flags.FLAGS

# Processed data path
flags.DEFINE_string('processed_data_dir', os.path.join('data', 'pt'),
                    'Path to the folder for the processed data files.')

# Dataloaders flags
flags.DEFINE_integer('batch_size', 3, 'The batch size.')
flags.DEFINE_integer('num_workers_loader', 2,
                     'The number of workers for the data loaders.')

# Model parameters flags
flags.DEFINE_integer('latent_size', 8,
                     'The size of the latent features in the model.')
flags.DEFINE_integer('num_mlp_layers', 2,
                     'The number of hidden layers in the MLPs.')
flags.DEFINE_integer('message_passing_steps', 5,
                     'The number of message passing steps in the processor.')

# Lightning wrapper flags
flags.DEFINE_float('learning_rate', 1e-3, 'The training learning rate.')

# Logger flags
flags.DEFINE_string('experiment_name', 'MGN-training',
                    'The MLFlow experiment name.')

# Checkpoint flags
flags.DEFINE_integer('save_top_k', 3, 'The number of models to save.')

# Strategy flags
flags.DEFINE_integer('num_workers_ray', 2, 'The number of workers.')
flags.DEFINE_integer('num_cpus_per_worker', 1,
                     'The number of cpus for each worker.')
flags.DEFINE_bool('use_gpu', False, 'Whether to use gpu or not.')

# Trainer flags
flags.DEFINE_integer('max_epochs', 10, 'The number of epochs.')
flags.DEFINE_integer('log_every_n_steps', 1, 'How often to log within steps.')


def main(_):

    torch.manual_seed(21)

    dataset = FromDiskGeometricDataset(FLAGS.processed_data_dir)
    train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2])

    config = {
        'batch_size': FLAGS.batch_size,
        'latent_size': FLAGS.latent_size,
        'num_mlp_layers': FLAGS.num_mlp_layers,
        'message_passing_steps': FLAGS.message_passing_steps,
        'learning_rate': FLAGS.learning_rate
    }
    experiment_config = {
        'experiment_name': FLAGS.experiment_name,
        'num_workers_loader': FLAGS.num_workers_loader,
        'num_workers_ray': FLAGS.num_workers_ray,
        'num_cpus_per_worker': FLAGS.num_cpus_per_worker,
        'use_gpu': FLAGS.use_gpu,
        'max_epochs': FLAGS.max_epochs,
        'log_every_n_steps': FLAGS.log_every_n_steps,
        'save_top_k': FLAGS.save_top_k,
    }

    model_training.train_model(config, experiment_config, train_dataset,
                               validation_dataset)


if __name__ == '__main__':
    app.run(main)
