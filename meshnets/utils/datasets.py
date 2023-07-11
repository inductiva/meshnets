"""File containing dataset classes."""
import os
import warnings

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from meshnets.utils import data_processing

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

    Note that the directory, file names, and file extensions can be different.
    `process_data` allows to produce the graph files upon instantiating the 
    dataset class from a folder containing only the mesh files.

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
        samples = [
            sample for sample in os.listdir(data_dir)
            if os.path.isdir(os.path.join(self.root_dir, sample))
        ]
        self.samples = sorted(samples)

        self.mesh_file_name = mesh_file_name
        self.graph_file_name = graph_file_name

    def convert_mesh_to_graph_data(self) -> None:
        """Create graph files from the mesh files in the data directory.
        
        This method saves each graph file in the same sample directory as
        its corresponding mesh file. The same file name is used."""

        for i in self.indices():
            mesh_file_path = self.get_mesh_path(i)

            processed_graph = data_processing.mesh_file_to_graph_data(
                mesh_file_path, WIND_VECTOR, load_pressure=True)

            processed_file_path = os.path.join(self.get_sample_path(i),
                                               self.graph_file_name)

            torch.save(processed_graph, processed_file_path)

    def remove_files(self, file_name_to_delete: str) -> None:
        """Remove the required file in each sample directory."""

        for i in range(self.len()):
            file_path_to_delete = os.path.join(self.get_sample_path(i),
                                               file_name_to_delete)
            os.remove(file_path_to_delete)

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
            #raise FileNotFoundError(mesh_path)
            warnings.warn(f"File '{mesh_path}' does not exist.")

        return mesh_path

    def get_graph_path(self, idx: int) -> str:
        """Get the path to the graph file at the given index."""

        sample_path = self.get_sample_path(idx)
        graph_path = os.path.join(sample_path, self.graph_file_name)

        if not os.path.isfile(graph_path):
            #raise FileNotFoundError(graph_path)
            warnings.warn(f"File '{graph_path}' does not exist.")

        return graph_path

    def get(self, idx: int) -> Data:
        """Get an element from the dataset at the given index.
        
        The only things that exist in RAM at this point are the paths 
        to the samples. This method loads and returns the sample from disk."""

        graph_path = self.get_graph_path(idx)
        return torch.load(graph_path)

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
