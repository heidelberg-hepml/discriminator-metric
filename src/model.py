import torch
import torch.nn as nn

from .vblinear import VBLinear

class Discriminator(nn.Module):
    """
    Class that implements a simple fully connected network, optionally as a Bayesian network
    """

    def __init__(self, input_dim: int, params: dict):
        """
        Constructs the network using the given parameters.

        Args:
            input_dim: Dimension of the input data
            params: Dictionary with the architecture hyperparameters
        """
        super().__init__()
        self.bayesian_layers = []

        if params["bayesian"]:
            layer_class = VBLinear
            layer_kwargs = {
                "prior_prec": params.get("prior_prec", 1.0),
                "std_init": params.get("std_init", -9)
            }
        else:
            layer_class = nn.Linear
            layer_kwargs = {}

        activation = {
            "relu": nn.ReLU,
            "leaky_relu": lambda: nn.LeakyReLU(params.get("negative_slope", 0.01)),
            "elu": nn.ELU
        }[params["activation"]]
        dropout = params.get("dropout", 0)

        layers = []
        layer_size = input_dim
        for i in range(params["layers"] - 1):
            hidden_size = params["hidden_size"]
            layer = layer_class(layer_size, hidden_size, **layer_kwargs)
            if params["bayesian"]:
                self.bayesian_layers.append(layer)
            layers.append(layer)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(activation())
            layer_size = hidden_size
        layer = layer_class(layer_size, 1, **layer_kwargs)
        if params["bayesian"]:
            self.bayesian_layers.append(layer)
        layers.append(layer)
        self.layers = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the network forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.layers(x)
