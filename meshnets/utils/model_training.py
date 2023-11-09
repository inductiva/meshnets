"""Train the MeshGraphNet model on a cluster.

The `train_model` method can be called for standard training with a Ray strategy
or be used for tuning using Ray."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

import ray
import ray.train.lightning

from meshnets.modules.lightning_wrapper import MGNLightningWrapper
from meshnets.modules.model import MeshGraphNet
from meshnets.utils.callbacks import GeometricBatchSize
from meshnets.utils.callbacks import GPUUsage
from meshnets.utils.callbacks import GradientNorm
from meshnets.utils.callbacks import MLFlowLoggerFinalizeCheckpointer


def train_model(config):
    """Train the MeshGraphNet model given the training config, experiment config
    and datasets.
    """
    train_dataset = config['train_dataset']
    validation_datasets = config['validation_datasets']
    validation_datasets_names = config['validation_datasets_names']

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
    experiment_name = config['experiment_name']
    # Dataloader config
    num_workers_loader = config['num_workers_loader']

    # Trainer config
    max_epochs = config['max_epochs']
    log_every_n_steps = config['log_every_n_steps']
    # Checkpoint config
    save_top_k = config['save_top_k']

    # Compute the training dataset stats for normalization
    train_stats = train_dataset.dataset[train_dataset.indices].get_stats()

    # Load the training dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers_loader,
                              persistent_workers=True,
                              shuffle=True)

    # Load the validation datasets
    validation_loaders = []
    for val_dataset in validation_datasets:
        validation_loaders.append(
            DataLoader(val_dataset,
                       batch_size=batch_size,
                       num_workers=num_workers_loader,
                       persistent_workers=True,
                       shuffle=False))

    train_dataset_name = validation_datasets_names[0]

    lightning_wrapper = MGNLightningWrapper(
        MeshGraphNet,
        node_features_size=train_dataset.dataset.num_node_features,
        edge_features_size=train_dataset.dataset.num_edge_features,
        output_size=train_dataset.dataset.num_label_features,
        latent_size=latent_size,
        num_mlp_layers=num_mlp_layers,
        message_passing_steps=message_passing_steps,
        x_mean=train_stats['x_mean'],
        x_std=train_stats['x_std'],
        edge_attr_mean=train_stats['edge_attr_mean'],
        edge_attr_std=train_stats['edge_attr_std'],
        y_mean=train_stats['y_mean'],
        y_std=train_stats['y_std'],
        validation_datasets_names=validation_datasets_names,
        learning_rate=learning_rate)

    # The logger creates a new MLFlow run automatically
    # Checkpoints are logged as artifacts at the end of training
    mlf_logger = MLFlowLoggerFinalizeCheckpointer(
        experiment_name=experiment_name)

    num_params = sum(p.numel() for p in lightning_wrapper.model.parameters())
    # Log the config parameters and training dataset stats for the run to MLFlow
    mlf_logger.log_hyperparams({
        'train_dataset_name': train_dataset_name,
        'batch_size': batch_size,
        'num_params': num_params
    })

    callbacks = []
    monitor_metric = f'val_loss_{train_dataset_name}'
    # Add a suffix following lightning logging behavior
    suffix = '' if len(validation_datasets_names) == 1 else '/dataloader_idx_0'

    # Save checkpoints locally in the mlflow folder
    checkpoint_callback = ModelCheckpoint(monitor=monitor_metric + suffix,
                                          save_top_k=save_top_k)
    callbacks.append(checkpoint_callback)

    gpu_callback = GPUUsage(log_freq=log_every_n_steps)
    callbacks.append(gpu_callback)

    gradient_callback = GradientNorm(log_freq=log_every_n_steps)
    callbacks.append(gradient_callback)

    batch_size_callback = GeometricBatchSize(log_freq=log_every_n_steps)
    callbacks.append(batch_size_callback)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices='auto',
        accelerator='auto',
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=[ray.train.lightning.RayTrainReportCallback()],
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(lightning_wrapper, train_loader, validation_loaders)
