import os
import numpy as np
import pyvista as pv

from tqdm import tqdm

from meshnets.utils import data_visualization

from pytorch_lightning.callbacks import ModelCheckpoint

from absl import flags, app
import torch
from torch.utils.data import random_split

import pytorch_lightning as pl

from torch_geometric.loader import DataLoader

import meshio

from meshnets.utils import model_training
from meshnets.utils.datasets import FromDiskGeometricDataset
from meshnets.utils import data_processing
from meshnets.modules.model import MeshGraphNet
from meshnets.modules.lightning_wrapper import MGNLightningWrapper

from meshnets.utils import model_loading

from torch_geometric.data import Batch

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

# Random seed flag
flags.DEFINE_integer('random_seed', 21,
                     'The seed to initialize the random number generator.')

# Data path flags
flags.DEFINE_string('data_dir', os.path.join('data', '0_5k_simulations'),
                    'Path to the folder containing the mesh files.')

# Dataset split flags
flags.DEFINE_float('train_split', 0.9,
                   'The fraction of the dataset used for traing.')
flags.DEFINE_float('validation_split', 0.1,
                   'The fraction of the dataset used for validation.')


def main(_):

    random_seed = FLAGS.random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    dataset = FromDiskGeometricDataset(FLAGS.data_dir)
    train_dataset, validation_dataset = random_split(
        dataset, [FLAGS.train_split, FLAGS.validation_split])

    data_stats = dataset.get_stats()

    wrapper = MGNLightningWrapper.load_from_checkpoint(
        'mlruns/303814507696829307/cbabdedc296c4abb804abb2919d00da7/checkpoints/epoch=4-step=70.ckpt'
    )

    print(wrapper.model.edge_attr_mean)
    exit()

    sample_idx = np.random.choice(dataset.indices())

    mesh_path = dataset.get_mesh_path(sample_idx)
    graph = data_processing.mesh_file_to_graph_data(mesh_path, (10, 0, 0),
                                                    load_pressure=True)

    groundtruth = graph.y.detach().numpy()

    plt.hist(groundtruth, bins=100)
    plt.show()
    data_visualization.plot_mesh(mesh_path,
                                 clim=[groundtruth.min(),
                                       groundtruth.max()])

    normalized_groundtruth = wrapper.normalize_labels(graph).y.detach().numpy()
    plt.hist(normalized_groundtruth, bins=100)
    plt.show()
    data_visualization.plot_mesh_with_scalars(
        mesh_path,
        normalized_groundtruth.flatten(),
        clim=[normalized_groundtruth.min(),
              normalized_groundtruth.max()])


if __name__ == '__main__':
    app.run(main)
