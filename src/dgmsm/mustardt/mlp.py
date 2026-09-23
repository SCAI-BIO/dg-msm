import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layers: List[int],
        out_features: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Simple MLP:
          Linear -> ReLU -> Dropout (repeated for hidden_layers) -> Linear(out)
        """
        super().__init__()

        layers: list[nn.Module] = []
        dim = in_features

        for h in hidden_layers:
            layers.append(nn.Linear(dim, h, bias=bias))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = h

        layers.append(nn.Linear(dim, out_features, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)