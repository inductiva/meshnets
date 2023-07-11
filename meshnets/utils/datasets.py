"""File containing dataset classes."""
import os
import glob
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from meshnets.utils import data_processing

# TODO(victor): the wind vector should be
# obtained individually for each mesh
WIND_VECTOR = (10, 0, 0)


class FromDiskGeometricDataset(Dataset):
    """Reads a torch_geometric dataset from a given directory.
    
    The data is loaded from disk each time it as accessed to reduce
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
    `process_data` allows to produce the graph files upon instanciating the 
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
                 process_data: bool = False,
                 mesh_file_ext: str = '.vtk',
                 graph_file_ext: str = '.pt',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.root_dir = data_dir
        self.samples = sorted(os.listdir(data_dir))

        self.mesh_file_ext = mesh_file_ext
        self.graph_file_ext = graph_file_ext

        if process_data:
            self.process_data()

    def process_data(self) -> None:
        """Process the mesh files in the data directory to their graph
        representation. Save the graph file in the same sample directory."""
        for i in range(self.len()):
            mesh_file_path = self.get_mesh_path(i)

            processed_graph = data_processing.mesh_file_to_graph_data(
                mesh_file_path, WIND_VECTOR, load_pressure=True)

            processed_file_path = Path(mesh_file_path).with_suffix(
                self.graph_file_ext)

            torch.save(processed_graph, processed_file_path)

    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def get_sample_path(self, idx: int) -> str:
        """Get the path to the sample directory at the given index."""
        return os.path.join(self.root_dir, self.samples[idx])

    def get_mesh_path(self, idx: int) -> str:
        """Get the path to the mesh file at the given index."""
        sample_path = self.get_sample_path(idx)
        mesh_path = glob.glob(
            os.path.join(sample_path, f'*{self.mesh_file_ext}'))[0]
        return mesh_path

    def get_graph_path(self, idx: int) -> str:
        """Get the path to the graph file at the given index."""
        sample_path = self.get_sample_path(idx)
        graph_path = glob.glob(
            os.path.join(sample_path, f'*{self.graph_file_ext}'))[0]
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
