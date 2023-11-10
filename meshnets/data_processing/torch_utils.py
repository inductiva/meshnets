"""Utility functions for usage with pytorch"""
import torch
import torch_geometric


def collate_fn(examples,
               x_key='node_features',
               edge_index_key='edges',
               edge_attr_key='edge_features',
               y_key='wind_pressures'):
    graphs = [
        torch_geometric.data.Data(x=torch.tensor(e[x_key]),
                                  edge_index=torch.tensor(e[edge_index_key]).T,
                                  edge_attr=torch.tensor(e[edge_attr_key]),
                                  y=torch.tensor(e[y_key])) for e in examples
    ]
    return torch_geometric.data.Batch.from_data_list(graphs)
