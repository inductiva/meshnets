"""Train the MeshGraphNet model on a cluster.

The `train_model` method can be called for standard training with a Ray strategy
or be used for tuning using Ray."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

import datasets

import ray
import ray.train.lightning

from .. import modules
from .. import data_processing
from . import callbacks


def get_node_feature_size(dataloader):
    example = next(iter(dataloader))
    return example.x.shape[-1]


def get_edge_feature_size(dataloader):
    example = next(iter(dataloader))
    return example.edge_attr.shape[-1]


def get_output_size(dataloader):
    example = next(iter(dataloader))
    if example.y.ndim == 1:
        return 1
    return example.y.shape[-1]


def train_model(config):
    """Train the MeshGraphNet model given the training config, experiment config
    and datasets.
    """
    dataset_version = config['dataset_version']
    val_dataset_versions = config['val_dataset_versions']

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

    # TODO(augusto): Compute the training dataset stats for
    # normalization. Previously this was done in the dataset
    # class. Now we need a way to iterate over the dataset to compute
    # the stats.

    train_dataset = datasets.load_dataset('inductiva/wind_tunnel',
                                          version=dataset_version,
                                          split='train')
    train_dataset = (train_dataset.map(
        lambda x: data_processing.data_mappers.to_undirected(x, 'edges')).map(
            data_processing.data_mappers.make_edge_features).map(
                lambda x: data_processing.data_mappers.make_node_features(
                    x, 'wind_vector')))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=data_processing.torch_utils.dict_to_geometric_data,
        batch_size=batch_size,
        num_workers=num_workers_loader,
        shuffle=True)

    validation_loaders = []
    for val_dataset_version in val_dataset_versions:
        val_dataset = datasets.load_dataset('inductiva/wind_tunnel',
                                            version=val_dataset_version,
                                            split='train')
        val_dataset = (
            val_dataset.map(lambda x: data_processing.data_mappers.
                            to_undirected(x, 'edges')).map(
                                data_processing.data_mappers.make_edge_features
                            ).map(lambda x: data_processing.data_mappers.
                                  make_node_features(x, 'wind_vector')))

        validation_loaders.append(
            torch.utils.data.DataLoader(val_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers_loader,
                                        shuffle=False))

    node_feature_size = get_node_feature_size(train_loader)
    edge_feature_size = get_edge_feature_size(train_loader)
    output_size = get_output_size(train_loader)

    # TODO(augusto): Priority: This must be changes. At the moment
    # this is just for testing the hugging face integration. It will
    # be changed with the highest priority.
    train_stats = {
        'x_mean': torch.tensor([0.0]),
        'x_std': torch.tensor([1.0]),
        'edge_attr_mean': torch.tensor([0.0]),
        'edge_attr_std': torch.tensor([1.0]),
        'y_mean': torch.tensor([0.0]),
        'y_std': torch.tensor([1.])
    }

    model = modules.lightning_wrapper.MGNLightningWrapper(
        modules.model.MeshGraphNet,
        node_features_size=node_feature_size,
        edge_features_size=edge_feature_size,
        output_size=output_size,
        latent_size=latent_size,
        num_mlp_layers=num_mlp_layers,
        message_passing_steps=message_passing_steps,
        x_mean=train_stats['x_mean'],
        x_std=train_stats['x_std'],
        edge_attr_mean=train_stats['edge_attr_mean'],
        edge_attr_std=train_stats['edge_attr_std'],
        y_mean=train_stats['y_mean'],
        y_std=train_stats['y_std'],
        validation_datasets_names=val_dataset_versions,
        learning_rate=learning_rate)

    # The logger creates a new MLFlow run automatically
    # Checkpoints are logged as artifacts at the end of training
    mlf_logger = callbacks.MLFlowLoggerFinalizeCheckpointer(
        experiment_name=experiment_name)

    num_params = sum(p.numel() for p in model.model.parameters())
    # Log the config parameters and training dataset stats for the run to MLFlow
    mlf_logger.log_hyperparams({
        'train_dataset_name': dataset_version,
        'batch_size': batch_size,
        'num_params': num_params
    })

    all_callbacks = []
    monitor_metric = f'val_loss_{dataset_version}'
    # Add a suffix following lightning logging behavior
    suffix = '' if len(val_dataset_versions) == 1 else '/dataloader_idx_0'

    # Save checkpoints locally in the mlflow folder
    checkpoint_callback = ModelCheckpoint(monitor=monitor_metric + suffix,
                                          save_top_k=save_top_k)
    all_callbacks.append(checkpoint_callback)

    gpu_callback = callbacks.GPUUsage(log_freq=log_every_n_steps)
    all_callbacks.append(gpu_callback)

    gradient_callback = callbacks.GradientNorm(log_freq=log_every_n_steps)
    all_callbacks.append(gradient_callback)

    batch_size_callback = callbacks.GeometricBatchSize(
        log_freq=log_every_n_steps)
    all_callbacks.append(batch_size_callback)

    all_callbacks.append(ray.train.lightning.RayTrainReportCallback())
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices='auto',
        accelerator='auto',
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        callbacks=all_callbacks,
    )

    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model, train_loader, validation_loaders)
