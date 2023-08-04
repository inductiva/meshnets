"""The main file for data processing."""

import os

from absl import app
from absl import flags

from meshnets.utils.datasets import FromDiskGeometricDataset

FLAGS = flags.FLAGS

# Path flags
flags.DEFINE_string('data_dir', os.path.join('data', 'dataset'),
                    'Path to the folder containing the mesh files.')


def main(_):
    """Process mesh data files to encoded graph data files."""

    dataset = FromDiskGeometricDataset(FLAGS.data_dir)
    dataset.convert_mesh_to_graph_data()


if __name__ == '__main__':
    app.run(main)
