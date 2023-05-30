"""The main file to run our code

Currently, it loads mesh data from a .vtk file, defines a wind vector,
and creates a Graph object from them
"""

import os

from absl import app, flags, logging
import numpy as np

from meshnets.utils import data_loading
from meshnets.utils import data_processing

FLAGS = flags.FLAGS
flags.DEFINE_string('input_object', os.path.join('data', 'blob.vtk'),
                    'File path of the object to be visualized')

# Wind vector associated with the simulation
WIND_VECTOR = (10, 0, 0)


def main(_):

    logging.info('Loading the mesh data from %s', FLAGS.input_object)
    mesh = data_loading.load_edge_mesh_pv(FLAGS.input_object,
                                          get_pressure=True,
                                          verbose=False)
    nodes, edges, pressure = mesh[0], mesh[1], mesh[2]

    # node features for each node are the wind vector
    node_features = np.tile(WIND_VECTOR, (len(nodes), 1))

    logging.info('Building graph from the mesh data')
    graph = data_processing.edge_mesh_to_graph(nodes, node_features, edges,
                                               pressure)

    logging.info('Node features shape : %s', graph.x.shape)
    logging.info('Edge index shape : %s', graph.edge_index.shape)
    logging.info('Edge features shape : %s', graph.edge_attr.shape)
    if graph.y is not None:
        logging.info('Pressure label shape : %s', graph.y.shape)


if __name__ == '__main__':
    app.run(main)
