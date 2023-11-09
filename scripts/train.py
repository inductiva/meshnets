"""The main file for model training."""

import os

from absl import app
from absl import flags
import torch

import meshnets

TUNING_RUN = False
FLAGS = flags.FLAGS

# Random seed flag
flags.DEFINE_integer('random_seed', 21,
                     'The seed to initialize the random number generator.')

# Processed data path
flags.DEFINE_string('data_dir', os.path.join('data', 'dataset'),
                    'Path to the folder for the processed data files.')
flags.DEFINE_multi_string(
    'val_data_dirs', [], 'Paths to the folders for the validation data files.')

# Dataset splits path
flags.DEFINE_float('train_split', 0.9,
                   'The fraction of the dataset used for traing.')
flags.DEFINE_float('validation_split', 0.1,
                   'The fraction of the dataset used for validation.')

# Dataloaders flags
flags.DEFINE_integer('batch_size', 8, 'The batch size.')
flags.DEFINE_integer('num_workers_loader', 2,
                     'The number of workers for the data loaders.')

# Model parameters flags
flags.DEFINE_integer('latent_size', 32,
                     'The size of the latent features in the model.')
flags.DEFINE_integer('num_mlp_layers', 2,
                     'The number of hidden layers in the MLPs.')
flags.DEFINE_integer('message_passing_steps', 30,
                     'The number of message passing steps in the processor.')

# Lightning wrapper flags
flags.DEFINE_float('learning_rate', 1e-3, 'The training learning rate.')

# Logger flags
flags.DEFINE_string('experiment_name', 'MGN-training',
                    'The MLFlow experiment name.')

# Checkpoint flags
flags.DEFINE_integer('save_top_k', 3, 'The number of models to save.')

# Strategy flags
flags.DEFINE_integer('num_workers_ray', 1, 'The number of workers.')
flags.DEFINE_integer('num_cpus_per_worker', 24,
                     'The number of cpus for each worker.')
flags.DEFINE_bool('use_gpu', True, 'Whether to use gpu or not.')

# Trainer flags
flags.DEFINE_integer('max_epochs', 150, 'The number of epochs.')
flags.DEFINE_integer('log_every_n_steps', 75, 'How often to log within steps.')


def main(_):

    random_seed = FLAGS.random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    dataset_name = os.path.basename(FLAGS.data_dir)
    dataset = meshnets.utils.datasets.FromDiskGeometricDataset(FLAGS.data_dir)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [FLAGS.train_split, FLAGS.validation_split])

    val_datasets_names = [dataset_name]
    val_datasets = [validation_dataset]

    for val_data_dir in FLAGS.val_data_dirs:
        dataset_name = os.path.basename(val_data_dir)
        dataset = meshnets.utils.datasets.FromDiskGeometricDataset(val_data_dir)
        _, validation_dataset = torch.utils.data.random_split(
            dataset, [FLAGS.train_split, FLAGS.validation_split])

        val_datasets_names.append(dataset_name)
        val_datasets.append(validation_dataset)

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
        'tuning_run': TUNING_RUN,
        'num_workers_ray': FLAGS.num_workers_ray,
        'num_cpus_per_worker': FLAGS.num_cpus_per_worker,
        'use_gpu': FLAGS.use_gpu,
        'max_epochs': FLAGS.max_epochs,
        'log_every_n_steps': FLAGS.log_every_n_steps,
        'save_top_k': FLAGS.save_top_k,
    }

    meshnets.utils.model_training.train_model(
        config,
        experiment_config,
        train_dataset,
        validation_datasets_names=val_datasets_names,
        validation_datasets=val_datasets)


if __name__ == '__main__':
    app.run(main)
