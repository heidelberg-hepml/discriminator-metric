import torch
import torch.nn as nn

from .vblinear import VBLinear

class Discriminator(nn.Module):
    def __init__(self, input_dim: int, params: dict):
        super().__init__()
        self.bayesian_layers = []

        if params["bayesian"]:
            layer_class = nn.Linear
            layer_kwargs = {}
        else:
            layer_class = VBLinear
            layer_kwargs = {
                "prior_prec": params.get("prior_prec", 1.0),
                "std_init": params.get("std_init", -9)
            }

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
                self.layers.append(nn.Dropout(p=dropout))
            layers.append(activation())
            layer_size = hidden_size
        layer = layer_class(layer_size, 1, **layer_kwargs)
        if params["bayesian"]:
            self.bayesian_layers.append(layer)
        layers.append(layer)
        self.layers = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor):
        return self.layers(x)
