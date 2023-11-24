"""The main file for model training."""
from absl import app
from absl import flags
import torch

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

import meshnets

TUNING_RUN = False
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

flags.DEFINE_integer(
    'num_examples_dataset_stats', 1000,
    'Number of examples in the dataset to use for statistics.')

# Trainer flags
flags.DEFINE_integer('max_epochs', 150, 'The number of epochs.')
flags.DEFINE_integer('log_every_n_steps', 75, 'How often to log within steps.')

flags.mark_flag_as_required('dataset_version')


def main(_):

    random_seed = FLAGS.random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    config = {
        'dataset_version': FLAGS.dataset_version,
        'val_dataset_versions': FLAGS.val_dataset_versions,
        'batch_size': FLAGS.batch_size,
        'latent_size': FLAGS.latent_size,
        'num_mlp_layers': FLAGS.num_mlp_layers,
        'message_passing_steps': FLAGS.message_passing_steps,
        'learning_rate': FLAGS.learning_rate,
        'experiment_name': FLAGS.experiment_name,
        'num_workers_loader': FLAGS.num_workers_loader,
        'max_epochs': FLAGS.max_epochs,
        'log_every_n_steps': FLAGS.log_every_n_steps,
        'save_top_k': FLAGS.save_top_k,
        'num_examples_dataset_stats': FLAGS.num_examples_dataset_stats
    }

    resources_per_worker = {
        'CPU': FLAGS.num_cpus_per_worker,
        'GPU': 1 if FLAGS.use_gpu else 0
    }
    scaling_config = ScalingConfig(num_workers=FLAGS.num_workers_ray,
                                   use_gpu=FLAGS.use_gpu,
                                   resources_per_worker=resources_per_worker)

    trainer = TorchTrainer(meshnets.utils.model_training.train_model,
                           scaling_config=scaling_config,
                           train_loop_config=config)
    trainer.fit()


if __name__ == '__main__':
    app.run(main)
