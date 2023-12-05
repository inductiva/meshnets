"""Train the MeshGraphNet model on a cluster.

The `train_model` method can be called for standard training with a Ray strategy
or be used for tuning using Ray."""
from absl import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

import datasets

import ray
import ray.train.lightning

from meshnets.utils import callbacks
from meshnets import modules
from meshnets import data_processing


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


def compute_train_stats(dataset, max_examples=1000):
    """Compute the mean and standard deviation of dataset features.

    Args:
        dataset: A `torch_geometric.data.Dataset` object.
        max_examples: The maximum number of examples to use for computing the
            statistics.

    Returns:
        A dictionary with the mean and standard deviation of the node features,
    """
    y_mean, y_std = data_processing.dataset_statistics.compute_mean_and_std(
        dataset, 'wind_pressures', None, max_examples)
    edge_attr_mean, edge_attr_std =\
        data_processing.dataset_statistics.compute_mean_and_std(
        dataset, 'edge_features', None, max_examples)
    x_mean, x_std = data_processing.dataset_statistics.compute_mean_and_std(
        dataset, 'node_features', None, max_examples)
    return {
        'x_mean': torch.tensor([x_mean], dtype=torch.float32),
        'x_std': torch.tensor([x_std], dtype=torch.float32),
        'edge_attr_mean': torch.tensor([edge_attr_mean], dtype=torch.float32),
        'edge_attr_std': torch.tensor([edge_attr_std], dtype=torch.float32),
        'y_mean': torch.tensor([y_mean], dtype=torch.float32),
        'y_std': torch.tensor([y_std], dtype=torch.float32)
    }


def make_dataloader(version,
                    split_percentage,
                    split_start,
                    num_proc,
                    writer_batch_size,
                    batch_size,
                    num_workers_loader,
                    shuffle=True):
    """Make a dataloader for the wind tunnel dataset."""
    if split_start == 'beginning':
        split = f'train[:{split_percentage:.0%}]'
    elif split_start == 'end':
        split = f'train[-{split_percentage:.0%}:]'
    else:
        raise ValueError(f'Invalid split_start: {split_start}')
    dataset = datasets.load_dataset('inductiva/wind_tunnel',
                                    version=version,
                                    split=split,
                                    download_mode='force_redownload')
    logging.info('Making undirected edges.')
    dataset = dataset.map(
        lambda x: data_processing.data_mappers.to_undirected(x, 'edges'),
        num_proc=num_proc,
        writer_batch_size=writer_batch_size)
    dataset = dataset.map(data_processing.data_mappers.make_edge_features,
                          num_proc=num_proc,
                          writer_batch_size=writer_batch_size)
    dataset = dataset.map(lambda x: data_processing.data_mappers.
                          make_node_features(x, 'wind_vector'),
                          num_proc=num_proc,
                          writer_batch_size=writer_batch_size,
                          remove_columns='nodes')

    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=data_processing.torch_utils.dict_to_geometric_data,
        batch_size=batch_size,
        num_workers=num_workers_loader,
        shuffle=shuffle)
    return dataset, loader


def train_model(config):
    """Train the MeshGraphNet model given the training config, experiment config
    and datasets.
    """
    dataset_version = config['dataset_version']
    val_dataset_versions = config['val_dataset_versions']
    train_split = config['train_split']
    val_split = 1 - train_split

    # Configs for mapping over the dataset
    num_proc = config['num_proc']
    writer_batch_size = config['writer_batch_size']

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

    train_dataset, train_loader = make_dataloader(dataset_version, train_split,
                                                  'beginning', num_proc,
                                                  writer_batch_size, batch_size,
                                                  num_workers_loader)
    _, val_loader = make_dataloader(dataset_version, val_split, 'end', num_proc,
                                    writer_batch_size, batch_size,
                                    num_workers_loader)

    validation_loaders = [val_loader]
    for val_dataset_version in val_dataset_versions:
        _, loader = make_dataloader(val_dataset_version, 1, 'beginning',
                                    num_proc, writer_batch_size, batch_size,
                                    num_workers_loader)
        validation_loaders.append(loader)

    print('Getting node feature size.')
    node_feature_size = get_node_feature_size(train_loader)
    print('Getting edge feature size.')
    edge_feature_size = get_edge_feature_size(train_loader)
    print('Getting output size.')
    output_size = get_output_size(train_loader)

    print('Computing dataset statistics.')
    train_stats = compute_train_stats(train_dataset,
                                      config['num_examples_dataset_stats'])

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

    print('counting parameters')
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
    print('Starting training.')
    trainer.fit(model, train_loader, validation_loaders)
