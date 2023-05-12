"""The main file to run our code"""
from absl import app, flags, logging
import numpy as np

from utils import data_loading
from utils import data_processing

FLAGS = flags.FLAGS

flags.DEFINE_string('input_object', 'data/motorBike.obj',
                    'File path of the object to be visualized')

def main(_):

    logging.info(f'Loading the mesh from {FLAGS.input_object}')
    nodes, cells = data_loading.load_mesh_from_obj(FLAGS.input_object,
                                                   verbose=True)

    wind_vector = (10, 0, 0) # dummy wind vector along the X-axis
    # node features for each node are the wind vector
    node_features = np.tile(wind_vector, (len(nodes), 1))

    logging.info('Building the graph from the mesh')
    graph = data_processing.mesh_to_graph(nodes, node_features, cells)

    null_pressure = np.zeros(shape=(len(nodes), 1))

    logging.info(f'Node features shape : {graph.node_features.shape}')
    logging.info(f'Edge features shape : {graph.edge_set.features}')
    logging.info(f'Pressure label shape : {null_pressure.shape}')


if __name__ == '__main__':
    app.run(main)
