"""The main file for model training."""

import os

from absl import app
from absl import flags
import ray
from ray import tune
import torch
from torch.utils.data import random_split

from meshnets.utils import model_training
from meshnets.utils.datasets import FromDiskGeometricDataset

TUNING_RUN = True
FLAGS = flags.FLAGS

# Random seed flag
flags.DEFINE_integer('random_seed', 21,
                     'The seed to initialize the random number generator.')

# Processed data path
flags.DEFINE_string('processed_data_dir', os.path.join('data', 'ml'),
                    'Path to the folder for the processed data files.')

# Dataset splits path
flags.DEFINE_float('train_split', 0.8,
                   'The fraction of the dataset used for traing.')
flags.DEFINE_float('validation_split', 0.2,
                   'The fraction of the dataset used for validation.')

# Dataloaders flags
flags.DEFINE_multi_integer('batch_size', [4, 8], 'The batch size.')
flags.DEFINE_integer('num_workers_loader', 2,
                     'The number of workers for the data loaders.')

# Model parameters flags
flags.DEFINE_list('latent_size', [8, 16, 32, 64],
                  'The size of the latent features in the model.')
flags.DEFINE_list('num_mlp_layers', [2, 3],
                  'The number of hidden layers in the MLPs.')
flags.DEFINE_list('message_passing_steps', [5, 10, 15],
                  'The number of message passing steps in the processor.')

# Lightning wrapper flags
flags.DEFINE_list('learning_rate', [1e-2, 1e-3, 1e-4],
                  'The training learning rate.')

# Logger flags
flags.DEFINE_string('experiment_name', 'MGN-tuning',
                    'The MLFlow experiment name.')

# Checkpoint flags
flags.DEFINE_integer('save_top_k', 3, 'The number of models to save.')

# Strategy flags
flags.DEFINE_integer('num_workers_ray', 2, 'The number of workers.')
flags.DEFINE_integer('num_cpus_per_worker', 12,
                     'The number of cpus for each worker.')
flags.DEFINE_float(
    'num_gpus_per_worker', 0.5,
    'The number of gpus for each worker. If set to zero, GPUs are not used.')

# Trainer flags
flags.DEFINE_integer('max_epochs', 150, 'The number of epochs.')
flags.DEFINE_integer('log_every_n_steps', 1, 'How often to log within steps.')


def main(_):

    random_seed = FLAGS.random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    dataset = FromDiskGeometricDataset(FLAGS.processed_data_dir)
    train_dataset, validation_dataset = random_split(
        dataset, [FLAGS.train_split, FLAGS.validation_split])

    # Define the search space over which `tune` will run.
    config = {
        'batch_size':
            tune.grid_search(list(map(int, FLAGS.batch_size))),
        'latent_size':
            tune.grid_search(list(map(int, FLAGS.latent_size))),
        'num_mlp_layers':
            tune.grid_search(list(map(int, FLAGS.num_mlp_layers))),
        'message_passing_steps':
            tune.grid_search(list(map(int, FLAGS.message_passing_steps))),
        'learning_rate':
            tune.grid_search(list(map(float, FLAGS.learning_rate)))
    }

    experiment_config = {
        'experiment_name': FLAGS.experiment_name,
        'num_workers_loader': FLAGS.num_workers_loader,
        'tuning_run': TUNING_RUN,
        # The two following parameters are unused in tuning regime
        'num_workers_ray': None,
        'num_cpus_per_worker': None,
        'use_gpu': bool(FLAGS.num_gpus_per_worker),
        'max_epochs': FLAGS.max_epochs,
        'log_every_n_steps': FLAGS.log_every_n_steps,
        'save_top_k': FLAGS.save_top_k,
    }

    # Alocate resources per trial.
    resources_per_trial = {
        'cpu': FLAGS.num_cpus_per_worker,
        'gpu': FLAGS.num_gpus_per_worker
    }

    trainable = tune.with_parameters(model_training.train_model,
                                     experiment_config=experiment_config,
                                     train_dataset=train_dataset,
                                     validation_dataset=validation_dataset)

    ray.init()
    tune.run(trainable,
             config=config,
             num_samples=1,
             resources_per_trial=resources_per_trial)


if __name__ == '__main__':
    app.run(main)
