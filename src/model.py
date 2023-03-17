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
            def make_bayesian_layer(n_in, n_out):
                layer = VBLinear(
                    n_in,
                    n_out,
                    prior_prec = params.get("prior_prec", 1.0),
                    std_init = params.get("std_init", -9)
                )
                self.bayesian_layers.append(layer)
                return layer

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
            if params["bayesian"] and i >= params.get("skip_bayesian_layers", 0):
                layer = make_bayesian_layer(layer_size, hidden_size)
            else:
                layer = nn.Linear(layer_size, hidden_size)
            layers.append(layer)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(activation())
            layer_size = hidden_size
        if params["bayesian"]:
            layer = make_bayesian_layer(layer_size, 1)
        else:
            layer = nn.Linear(layer_size, 1)
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
