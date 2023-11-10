"""Utility functions for usage with pytorch"""
import torch
import torch_geometric


def collate_fn(examples,
               x_key='node_features',
               edge_index_key='edges',
               edge_attr_key='edge_features',
               y_key='wind_pressures'):
    """Collate function for pytorch dataloader

    Args:
        examples (list): List of examples from the dataset
        x_key (str, optional): Key for the node features. Defaults to
        'node_features'.
        edge_index_key (str, optional): Key for the edge
        indices. Defaults to 'edges'.
        edge_attr_key (str, optional): Key for the edge
        features. Defaults to 'edge_features'.
        y_key (str, optional): Key for the target values. Defaults to
        'wind_pressures'.

    This assumes examples have the structure:
    {
        x_key: torch.tensor,
        edge_index_key: torch.tensor,
        edge_attr_key: torch.tensor,
        y_key: torch.tensor
    }

    where the tensors have the following shapes:
    x_key: (num_nodes, num_node_features)
    edge_index_key: (num_edges, 2)
    edge_attr_key: (num_edges, num_edge_features)
    y_key: (num_nodes, num_target_values)

    The function wraps the examples in a torch_geometric.data.Data
    object and returns a torch_geometric.data.Batch object.

    """
    graphs = [
        torch_geometric.data.Data(
            x=torch.tensor(example[x_key]),
            edge_index=torch.tensor(example[edge_index_key]).T,
            edge_attr=torch.tensor(example[edge_attr_key]),
            y=torch.tensor(example[y_key])) for example in examples
    ]
    return torch_geometric.data.Batch.from_data_list(graphs)
