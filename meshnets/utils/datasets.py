"""File containing dataset classes."""
import os

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class FromDiskDataset(torch.utils.data.Dataset):
    """Reads a dataset from a given directory.

    The examples are stored in `data_dir` as .pt files. The .pt
    extension is what is commonly used to save pytorch objects such as
    tensors, graphs, models, etc. To save an pytorch object we can use
    ``torch.save(obj, 'path.pt')``. And then, to load it back into
    memory we can use ``torch.load('path.pt')``.

    A possible directory could look like this:

    data_dir/
        training_example_0.pt
        training_example_1.pt
        training_example_0.pt
        training_example_1.pt

    Note that the files do not have to be named like this.

    """

    def __init__(self, data_dir):
        super().__init__()
        self.root_dir = data_dir
        self.files = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """The get is simply a method to retrieve an element from the
        dataset. The only thing that exists in RAM at this point are
        the paths to the examples. Later, this Dataset class is
        wrapped by pytorch DataLoaders. These dataloaders will be
        responsible for batching the data and feeding the batches to
        the model.

        Example:
            >>> dataset = FromDiskDataset('data_dir')
            >>> dataloader = DataLoader(dataset, batch_size=32)
            >>> for batch in dataloader:
            >>>     x, y = batch, batch.y
        """
        data_path = os.path.join(self.root_dir, self.files[idx])
        return torch.load(data_path)


class FromDiskGeometricDataset(Dataset):
    """Reads a torch_geometric dataset from a given directory.
    
    This class inherits from torch_geometric Dataset and implements its
    abstract methods `len` and `get` from the FromDiskDataset class.
    This offers acces to the torch_geometric Dataset attributes when the .pt
    files retrieved from disk are torch_geometric Data.

    It also offer a new property called 'num_label_features' following
    the implementation of 'num_node_features' and 'num_edge_features'
    in torch_geometric Dataset.
    """

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_disk_dataset = FromDiskDataset(data_dir=data_dir)

    def len(self) -> int:
        """Returns the number of graphs stored in the dataset.
        
        This is the number of files in the dataset from disk."""
        return len(self.from_disk_dataset)

    def get(self, idx: int) -> Data:
        """Gets the data object at index :obj:`idx`.
        
        This is the data object at index :obj:`idx` in the dataset from disk."""
        return self.from_disk_dataset[idx]

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
