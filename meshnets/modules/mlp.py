""""Define an MLP class."""

from typing import List

import torch
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential


class MLP(torch.nn.Module):
    """MLP class.
    
    Define an MLP with ReLU activation from a list of layer widths.
    """

    def __init__(self,
                 widths: List[int],
                 layer_norm: bool = False,
                 activate_final: bool = False):
        """Initialize an MLP module given a list of layer widths, with
        a normalization layer and activated output if required."""
        super().__init__()

        self.mlp = self._make_mlp(widths, layer_norm, activate_final)

    def _make_mlp(self, widths: List[int], layer_norm: bool,
                  activate_final: bool):
        "Construct the MLP with ReLU activation."

        num_layers = len(widths)

        layers = []
        for i in range(num_layers - 1):
            layers.append(Linear(widths[i], widths[i + 1]))
            if i < (num_layers - 2) or activate_final:
                layers.append(ReLU())

        if layer_norm:
            layers.append(LayerNorm(widths[-1]))

        return Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to the input."""

        return self.mlp(x)
