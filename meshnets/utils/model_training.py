"""Train the MeshGraphNet model on a cluster.

The `train_model` method can be called for standard training with
a Ray strategy."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from torch_geometric.loader import DataLoader

import ray
from ray_lightning import RayStrategy

from meshnets.modules.lightning_wrapper import MGNLightningWrapper
from meshnets.modules.model import MeshGraphNet


def train_model(config, experiment_config, train_dataset, validation_dataset):
    """Train the MeshGraphNet model given the training config, experiment config
    and datasets.
    
    `config` contains parameters that can impact the training results and should
     be tuned.
    
    `experiment_config` contains parameters defining the experiment logging and
    the computing ressources to use.
    
    This method allows to run standard training with a Ray strategy."""

    # Config access
    # Dataloader config
    batch_size = config['batch_size']
    # Model config
    latent_size = config['latent_size']
    num_mlp_layers = config['num_mlp_layers']
    message_passing_steps = config['message_passing_steps']
    # Lightning wrapper config
    learning_rate = config['learning_rate']

    # Experiment config access
    # MLFlow config
    experiment_name = experiment_config['experiment_name']
    # Dataloader config
    num_workers_loader = experiment_config['num_workers_loader']

    # Ray config
    use_gpu = experiment_config['use_gpu']
    num_workers_ray = experiment_config['num_workers_ray']
    num_cpus_per_worker = experiment_config['num_cpus_per_worker']

    # Trainer config
    max_epochs = experiment_config['max_epochs']
    log_every_n_steps = experiment_config['log_every_n_steps']
    # Checkpoint config
    save_top_k = experiment_config['save_top_k']

    # Load the datasets
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers_loader,
                              persistent_workers=True,
                              shuffle=True)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers_loader,
                                   persistent_workers=True,
                                   shuffle=False)

    # Define the model
    model = MeshGraphNet(
        node_features_size=train_dataset.dataset.num_node_features,
        edge_features_size=train_dataset.dataset.num_edge_features,
        output_size=train_dataset.dataset.num_label_features,
        latent_size=latent_size,
        num_mlp_layers=num_mlp_layers,
        message_passing_steps=message_passing_steps)

    lightning_wrapper = MGNLightningWrapper(model, learning_rate=learning_rate)

    # The logger creates a new MLFlow run automatically
    mlf_logger = MLFlowLogger(experiment_name=experiment_name)
    # Log the config parameters for the run to MLFlow
    mlf_logger.log_hyperparams(config)

    # Define the list of callbacks
    callbacks = []

    # TODO(victor): This saves checkpoints locally in the mlflow folder
    # but will not log them as artifacts on the mlflow server
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          save_top_k=save_top_k)
    callbacks.append(checkpoint_callback)

    # Launch Ray and define a Ray strategy
    ray.init()
    strategy = RayStrategy(num_workers=num_workers_ray,
                           num_cpus_per_worker=num_cpus_per_worker,
                           use_gpu=use_gpu)

    # Instanciate a Lightning trainer with logger, callbacks and strategy
    trainer = pl.Trainer(logger=mlf_logger,
                         callbacks=callbacks,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         log_every_n_steps=log_every_n_steps)

    trainer.fit(lightning_wrapper, train_loader, validation_loader)

    return lightning_wrapper
