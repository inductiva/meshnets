"""File containing dataset classes."""
import os
import warnings

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

# TODO(victor): the wind vector should be
# obtained individually for each mesh
WIND_VECTOR = (10, 0, 0)


class FromDiskGeometricDataset(Dataset):
    """Reads a torch_geometric dataset from a given directory.
    
    The data is loaded from disk each time it is accessed to reduce
    RAM requirements. The data directory is assumed to be structured
    as follow:

    data/
        sample_1/
            mesh.vtk
            graph.pt
        sample_2/
            mesh.vtk
            graph.pt
        sample_3/
            mesh.vtk
            graph.pt

    This class inherits from torch_geometric Dataset and implements its
    abstract methods `len` and `get`. This offers access to the torch_geometric
    Dataset attributes when the .pt files retrieved from disk are
    torch_geometric Data.

    It also offer a new property called 'num_label_features' following
    the implementation of 'num_node_features' and 'num_edge_features'
    in torch_geometric Dataset.
    """

    def __init__(self,
                 data_dir: str,
                 *args,
                 mesh_file_name: str = 'pressure_field.vtk',
                 graph_file_name: str = 'pressure_field.pt',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.root_dir = data_dir

        self.mesh_file_name = mesh_file_name
        self.graph_file_name = graph_file_name

        samples = [
            sample for sample in os.listdir(data_dir)
            if os.path.isdir(os.path.join(self.root_dir, sample))
        ]

        # Filter out samples that do not have a `.pt` file.
        for sample in samples:
            graph_path = os.path.join(self.root_dir, sample,
                                      self.graph_file_name)
            if not os.path.isfile(graph_path):
                warnings.warn(f"Sample '{sample}' does not have a graph file.")
                samples.remove(sample)

        self.samples = sorted(samples)

    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def get_sample_path(self, idx: int) -> str:
        """Get the path to the sample directory at the given index."""
        return os.path.join(self.root_dir, self.samples[idx])

    def get_mesh_path(self, idx: int) -> str:
        """Get the path to the mesh file at the given index."""

        sample_path = self.get_sample_path(idx)
        mesh_path = os.path.join(sample_path, self.mesh_file_name)

        if not os.path.isfile(mesh_path):
            warnings.warn(f"File '{mesh_path}' does not exist.")

        return mesh_path

    def get_graph_path(self, idx: int) -> str:
        """Get the path to the graph file at the given index."""

        sample_path = self.get_sample_path(idx)
        graph_path = os.path.join(sample_path, self.graph_file_name)

        if not os.path.isfile(graph_path):
            warnings.warn(f"File '{graph_path}' does not exist.")

        return graph_path

    def get(self, idx: int) -> Data:
        """Get an element from the dataset at the given index.
        
        The only things that exist in RAM at this point are the paths 
        to the samples. This method loads and returns the sample from disk."""

        graph_path = self.get_graph_path(idx)
        return torch.load(graph_path)

    def get_stats(self):
        """Compute and return the dataset statistics.
        
        It returns a dictionnary containing the following fields:
            'x_mean'
            'x_std'
            'edge_attr_mean'
            'eadge_attr_std'
            'y_mean'
            'y_std'
        """

        # Get the first dataset sample to initialize stat tensors with its shape
        sample = self[0]

        # Small value to avoid zero variance
        eps = torch.tensor(1e-8)

        x_mean = torch.zeros(sample.x.shape[1:])
        x_std = torch.zeros(sample.x.shape[1:])
        x_num = 0

        edge_attr_mean = torch.zeros(sample.edge_attr.shape[1:])
        edge_attr_std = torch.zeros(sample.edge_attr.shape[1:])
        edge_attr_num = 0

        y_mean = torch.zeros(sample.y.shape[1:])
        y_std = torch.zeros(sample.y.shape[1:])
        y_num = 0

        # Iterate over the samples in the dataset to compute the means
        for sample in self:

            x_mean += torch.sum(sample.x, dim=0)
            x_num += sample.x.shape[0]

            edge_attr_mean += torch.sum(sample.edge_attr, dim=0)
            edge_attr_num += sample.edge_attr.shape[0]

            y_mean += torch.sum(sample.y, dim=0)
            y_num += sample.y.shape[0]

        x_mean = x_mean / x_num
        edge_attr_mean = edge_attr_mean / edge_attr_num
        y_mean = y_mean / y_num

        # Iterate over the samples in the dataset to compute the STDs
        for sample in self:
            x_std += torch.sum((sample.x - x_mean)**2, dim=0)

            edge_attr_std += torch.sum((sample.edge_attr - edge_attr_mean)**2,
                                       dim=0)

            y_std += torch.sum((sample.y - y_mean)**2, dim=0)

        x_std = torch.maximum(torch.sqrt(x_std / x_num), eps)
        edge_attr_std = torch.maximum(torch.sqrt(edge_attr_std / edge_attr_num),
                                      eps)
        y_std = torch.maximum(torch.sqrt(y_std / y_num), eps)

        stats = {
            'x_mean': x_mean,
            'x_std': x_std,
            'edge_attr_mean': edge_attr_mean,
            'edge_attr_std': edge_attr_std,
            'y_mean': y_mean,
            'y_std': y_std
        }

        return stats

    @property
    def num_label_features(self) -> int:
        """Return the number of features per label in the dataset."""

        # Following torch_geometric.data.Dataset implementation
        data = self[0]
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(self, '_data_list') and self._data_list is not None:
            self._data_list[0] = None
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, 'y'):
            # Following torch_geometric.data.storage.BaseStorage implementation
            if 'y' in data and isinstance(data.y, (torch.Tensor, np.ndarray)):
                return 1 if data.y.ndim == 1 else data.y.shape[-1]
            return 0
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'y'")
