from dataclasses import dataclass
import numpy as np

@dataclass
class EdgeSet:
    features: np.ndarray
    senders: np.ndarray
    receiver: np.ndarray

@dataclass
class Graph:
    node_features: np.ndarray
    edge_set: EdgeSet