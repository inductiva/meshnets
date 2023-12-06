"""The main file for result visualization."""
from absl import app
from absl import flags

import numpy as np
import torch
import torch_geometric

import datasets

from meshnets.data_processing import data_mappers
from meshnets.utils import data_visualization, model_loading

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_version", None,
                    "The version of the dataset to use.")
flags.DEFINE_integer("random_seed", 21,
                     "The seed to initialize the random number generator.")

flags.DEFINE_float("train_split", 0.9,
                   "The fraction of the dataset used for traing.")

flags.DEFINE_string("tracking_uri", None,
                    "The tracking URI for the MLFlow experiments.")

flags.DEFINE_string("run_id", None, "The run id of the experiment to load.")

flags.DEFINE_integer("checkpoint", 0,
                     "The checkpoint to load from the experiment.")

flags.DEFINE_float("point_size", 20, "The size of the points in the plot.")

flags.DEFINE_string("output_file_path", None, "File path to save the plot.")

flags.mark_flag_as_required("dataset_version")


def main(_):

    random_seed = FLAGS.random_seed
    if random_seed is not None:
        torch.manual_seed(random_seed)

    dataset = datasets.load_dataset(
        "inductiva/wind_tunnel",
        version=FLAGS.dataset_version,
        split=f"train[-{1 - FLAGS.train_split:.0%}:]",
        download_mode="force_redownload")
    len_dataset = len(dataset)
    random_example = dataset[np.random.randint(len_dataset)]
    random_example = data_mappers.to_undirected(random_example)
    random_example = data_mappers.make_edge_features(random_example)
    random_example = data_mappers.make_node_features(random_example)

    wrapper = model_loading.load_model_from_mlflow(FLAGS.tracking_uri,
                                                   FLAGS.run_id,
                                                   FLAGS.checkpoint)
    graph = torch_geometric.data.Data(
        x=torch.tensor(random_example["node_features"], dtype=torch.float32),
        edge_index=torch.tensor(random_example["edges"]).T,
        edge_attr=torch.tensor(random_example["edge_features"],
                               dtype=torch.float32),
        dtype=torch.float)

    with torch.no_grad():
        prediction = wrapper(graph)

    data_visualization.plot_3d_graph_and_predictions(
        random_example,
        prediction.numpy(),
        point_size=FLAGS.point_size,
        save_path=FLAGS.output_file_path)


if __name__ == "__main__":
    app.run(main)
