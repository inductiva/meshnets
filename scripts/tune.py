"""The main file for model training."""
from absl import app
from absl import flags
import mlflow

from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.tune.schedulers import ASHAScheduler

import torch

import meshnets

TUNING_RUN = True
FLAGS = flags.FLAGS

# Random seed flag
flags.DEFINE_integer('random_seed', 21,
                     'The seed to initialize the random number generator.')

flags.DEFINE_string('dataset_version', None,
                    'The dataset version to be used by hugging face.')

flags.DEFINE_multi_string(
    'val_dataset_versions', [],
    'The dataset versions to be used by hugging face for validation.')

# Dataset splits path
flags.DEFINE_float('train_split', 0.9,
                   'The fraction of the dataset used for traing.')
flags.DEFINE_float('validation_split', 0.1,
                   'The fraction of the dataset used for validation.')

# Dataloaders flags
flags.DEFINE_multi_integer('batch_size', [8, 16], 'The batch size.')
flags.DEFINE_integer('num_workers_loader', 2,
                     'The number of workers for the data loaders.')

# Model parameters flags
flags.DEFINE_list('latent_size', [16, 32, 64],
                  'The size of the latent features in the model.')
flags.DEFINE_list('num_mlp_layers', [2],
                  'The number of hidden layers in the MLPs.')
flags.DEFINE_list('message_passing_steps', [20, 30, 40],
                  'The number of message passing steps in the processor.')

# Lightning wrapper flags
flags.DEFINE_list('learning_rate', [1e-3], 'The training learning rate.')

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
flags.DEFINE_integer('log_every_n_steps', 75, 'How often to log within steps.')


def main(_):

    random_seed = FLAGS.random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Define the search space over which `tune` will run.
    search_space = {
        'batch_size':
            tune.grid_search(list(map(int, FLAGS.batch_size))),
        'latent_size':
            tune.grid_search(list(map(int, FLAGS.latent_size))),
        'num_mlp_layers':
            tune.grid_search(list(map(int, FLAGS.num_mlp_layers))),
        'learning_rate':
            tune.grid_search(list(map(float, FLAGS.learning_rate))),
        'message_passing_steps':
            tune.grid_search(list(map(int, FLAGS.message_passing_steps)))
    }

    config = {
        'dataset_version': FLAGS.dataset_version,
        'val_dataset_versions': FLAGS.val_dataset_versions,
        'num_workers_loader': FLAGS.num_workers_loader,
        'experiment_name': FLAGS.experiment_name,
        'max_epochs': FLAGS.max_epochs,
        'log_every_n_steps': FLAGS.log_every_n_steps,
        'save_top_k': FLAGS.save_top_k,
    }

    mlflow.create_experiment(FLAGS.experiment_name)

    # Alocate resources per trial.
    resources_per_trial = {
        'cpu': FLAGS.num_cpus_per_worker,
        'gpu': FLAGS.num_gpus_per_worker
    }

    scaling_config = ScalingConfig(num_workers=FLAGS.num_workers_ray,
                                   resources_per_worker=resources_per_trial)

    trainer = TorchTrainer(meshnets.utils.model_training.train_model,
                           scaling_config=scaling_config,
                           train_loop_config=config)
    scheduler = ASHAScheduler()

    tuner = tune.Tuner(
        trainer,
        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            num_samples=1,
            scheduler=scheduler,
        ),
    )
    tuner.fit()


if __name__ == '__main__':
    app.run(main)
