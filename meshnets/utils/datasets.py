"""File containing dataset classes"""
import os

import torch
from torch_geometric.data import Dataset


class FromDiskGraphDataset(Dataset):
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
        super().__init__(None, None, None)
        self.root_dir = data_dir
        self.files = sorted(os.listdir(data_dir))

    def len(self):
        return len(self.files)

    def get(self, idx):
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
