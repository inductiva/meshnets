"""The main file for result visualization."""

import os

from absl import app
from absl import flags
import numpy as np
import pyvista as pv
import torch
from torch.utils.data import random_split

from meshnets.utils.datasets import FromDiskGeometricDataset
from meshnets.utils import data_visualization
from meshnets.utils import model_loading

FLAGS = flags.FLAGS

flags.DEFINE_integer('random_seed', 21,
                     'The seed to initialize the random number generator.')

flags.DEFINE_string('data_dir', os.path.join('data', '0_5k_simulations'),
                    'Path to the folder containing the mesh files.')

flags.DEFINE_float('train_split', 0.9,
                   'The fraction of the dataset used for traing.')
flags.DEFINE_float('validation_split', 0.1,
                   'The fraction of the dataset used for validation.')

flags.DEFINE_string('tracking_uri', None,
                    'The tracking URI for the MLFlow experiments.')

flags.DEFINE_string('run_id', None, 'The run id of the experiment to load.')

flags.DEFINE_integer('checkpoint', 0,
                     'The checkpoint to load from the experiment.')

flags.DEFINE_bool('start_xvfb', False, 'Whether to start xvfb or not.')

flags.DEFINE_string('output_file_path', os.path.join('imgs', 'plot.png'),
                    'File path to save the plot.')


def main(_):

    random_seed = FLAGS.random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    dataset = FromDiskGeometricDataset(FLAGS.data_dir)
    _, validation_dataset = random_split(
        dataset, [FLAGS.train_split, FLAGS.validation_split])

    wrapper = model_loading.load_model_from_mlflow(FLAGS.tracking_uri,
                                                   FLAGS.run_id,
                                                   FLAGS.checkpoint)

    sample_idx = np.random.choice(validation_dataset.indices)

    mesh_path = dataset.get_mesh_path(sample_idx)
    graph_path = dataset.get_graph_path(sample_idx)

    with torch.no_grad():
        graph = torch.load(graph_path)
        graph = wrapper.normalize_labels(graph)
        normalized_groundtruth = graph.y

        prediction = wrapper(graph)

    if FLAGS.start_xvfb:
        pv.start_xvfb()

    data_visualization.plot_mesh_comparison(mesh_path,
                                            normalized_groundtruth,
                                            prediction,
                                            clim=None,
                                            rot_z=180,
                                            off_screen=False,
                                            screenshot=FLAGS.output_file_path)


if __name__ == '__main__':
    app.run(main)
