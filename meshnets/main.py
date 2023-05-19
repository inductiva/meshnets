"""The main file to run our code

Currently, it loads a mesh from an .obj file, defines a dummy wind vector
and create a Graph object from them
"""

from absl import app, flags, logging
import numpy as np

from meshnets.utils import data_loading
from meshnets.utils import data_processing

FLAGS = flags.FLAGS

flags.DEFINE_string('input_object', 'data/motorBike.obj',
                    'File path of the object to be visualized')


def main(_):

    logging.info('Loading the object from %s', FLAGS.input_object)
    nodes, cells = data_loading.load_mesh_from_obj(FLAGS.input_object,
                                                   verbose=True)

    wind_vector = (10, 0, 0)  # dummy wind vector along the X-axis
    # node features for each node are the wind vector
    node_features = np.tile(wind_vector, (len(nodes), 1))
    null_pressure = np.zeros(shape=(len(nodes), 1))

    logging.info('Building the graph from the mesh')
    graph = data_processing.triangle_mesh_to_graph(nodes, node_features, cells,
                                                   null_pressure)

    logging.info('Node features shape : %s', graph.x.shape)
    logging.info('Edge index shape : %s', graph.edge_index.shape)
    logging.info('Edge features shape : %s', graph.edge_attr.shape)
    logging.info('Pressure label shape : %s', graph.y.shape)


if __name__ == '__main__':
    app.run(main)
