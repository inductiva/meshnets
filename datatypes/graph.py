"""Data classes used by our model"""

from dataclasses import dataclass
import numpy as np


@dataclass
class EdgeSet:
    """Set of directed edges with features, sender node indices
    and receiver node indices"""
    features: np.ndarray
    senders: np.ndarray
    receiver: np.ndarray


@dataclass
class Graph:
    """Graph with node features and a set of directed edges
    defined over the nodes"""
    node_features: np.ndarray
    edge_set: EdgeSet
