"""Train the MeshGraphNet model on a cluster.

The `train_model` method can be called for standard training with a Ray strategy
or be used for tuning using Ray."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

import ray
from ray_lightning import RayStrategy
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from meshnets.modules.lightning_wrapper import MGNLightningWrapper
from meshnets.modules.model import MeshGraphNet
from meshnets.utils.callbacks import GeometricBatchSize
from meshnets.utils.callbacks import GPUUsage
from meshnets.utils.callbacks import GradientNorm
from meshnets.utils.callbacks import MLFlowLoggerFinalizeCheckpointer


def train_model(config, experiment_config, train_dataset, validation_dataset):
    """Train the MeshGraphNet model given the training config, experiment config
    and datasets.
    
    `config` contains parameters that can impact the training results and should
     be tuned.
    
    `experiment_config` contains parameters defining the experiment logging and
    the computing ressources to use.
    
    This method allows to run standard training with a Ray strategy, or to run
    Ray tuning. The mode indicated by `experiment_config['tuning_run']`. In the
    case of tuning, no strategy is adopted and the computating ressources for
    tuning are defined outside of the method."""

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
    tuning_run = experiment_config['tuning_run']
    use_gpu = experiment_config['use_gpu']
    if not tuning_run:
        num_workers_ray = experiment_config['num_workers_ray']
        num_cpus_per_worker = experiment_config['num_cpus_per_worker']

    # Trainer config
    max_epochs = experiment_config['max_epochs']
    log_every_n_steps = experiment_config['log_every_n_steps']
    # Checkpoint config
    save_top_k = experiment_config['save_top_k']

    # Compute the training dataset stats for normalization
    train_stats = train_dataset.dataset[train_dataset.indices].get_stats()

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

    # Define the wrapper and the model
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
        learning_rate=learning_rate)

    num_params = sum(p.numel() for p in lightning_wrapper.model.parameters())

    # The logger creates a new MLFlow run automatically
    # Checkpoints are logged as artifacts at the end of training
    mlf_logger = MLFlowLoggerFinalizeCheckpointer(
        experiment_name=experiment_name)

    mlf_logger.log_hyperparams({'batch_size': batch_size})
    mlf_logger.log_hyperparams({'num_params': num_params})

    callbacks = []

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          save_top_k=save_top_k)
    callbacks.append(checkpoint_callback)

    gpu_callback = GPUUsage(log_freq=log_every_n_steps)
    callbacks.append(gpu_callback)

    gradient_callback = GradientNorm(log_freq=log_every_n_steps)
    callbacks.append(gradient_callback)

    batch_size_callback = GeometricBatchSize(log_freq=log_every_n_steps)
    callbacks.append(batch_size_callback)

    # If this is not a tuning run we launch ray and define a ray strategy
    if not tuning_run:
        ray.init()
        strategy = RayStrategy(num_workers=num_workers_ray,
                               num_cpus_per_worker=num_cpus_per_worker,
                               use_gpu=use_gpu)
        accelerator = None
    # Otherwise we add the TuneReportCallback and define no strategy
    else:
        tune_report_callback = TuneReportCallback(['val_loss'],
                                                  on='validation_end')
        callbacks.append(tune_report_callback)
        strategy = None
        accelerator = 'gpu' if use_gpu else 'cpu'

    # Instanciate a Lightning trainer with logger, callbacks and strategy
    trainer = pl.Trainer(logger=mlf_logger,
                         callbacks=callbacks,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         log_every_n_steps=log_every_n_steps,
                         accelerator=accelerator)

    trainer.fit(lightning_wrapper, train_loader, validation_loader)
