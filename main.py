"""The main file to run our code"""
from absl import app, flags, logging
import numpy as np
import pyvista as pv

from utils import data_loading
from utils import data_processing

FLAGS = flags.FLAGS

flags.DEFINE_string("input_object", "data/motorBike.obj",
                    "File path of the object to be visualized")

def main(_):

    logging.info("Loading the mesh from %s", FLAGS.input_object)
    nodes, cells = data_loading.load_mesh_from_obj(FLAGS.input_object, verbose=True)

    wind_vector = (10, 0, 0) # dummy wind vector along the X-axis
    node_features = np.tile(wind_vector, (len(nodes), 1)) # node features are the wind vector for each node

    logging.info("Building the graph from the mesh")
    graph = data_processing.mesh_to_graph(nodes, node_features, cells)

    null_pressure = np.zeros(shape=(len(nodes), 1))


if __name__ == "__main__":
    app.run(main)
