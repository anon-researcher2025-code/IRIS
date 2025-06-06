from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ExU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
        output = F.relu(output)
        # print('n', n)
        # output = torch.clamp(output, 0, n)

        # print('output: ', torch.min(output), torch.max(output))

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

class LinReLU(torch.nn.Module):
    __constants__ = ['bias']

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(LinReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = (inputs - self.bias) @ self.weights
        output = F.relu(output)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


class FeatureNN(torch.nn.Module):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        input_shape: int,
        num_units: int,
        dropout: float,
        hidden_sizes: list = [64, 32],
        activation: str = 'relu'
    ) -> None:
        """Initializes FeatureNN hyperparameters.

        Args:
          input_shape: Dimensionality of input data.
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          hidden_sizes: List of hidden dimensions for each layer.
          activation: Activation function of first layer (relu or exu).
        """
        super(FeatureNN, self).__init__()
        self.input_shape = input_shape
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        all_hidden_sizes = [self.num_units] + self.hidden_sizes

        layers = []

        self.dropout = nn.Dropout(p=dropout)

        ## First layer is ExU
        if self.activation == "exu":
            layers.append(ExU(in_features=input_shape, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_shape, out_features=num_units))

        ## Hidden Layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())

        ## Last Linear Layer
        layers.append(nn.Linear(in_features=all_hidden_sizes[-1], out_features=1, bias=False))

        self.model = nn.ModuleList(layers)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        outputs = inputs.unsqueeze(1)
        for layer in self.model:
            outputs = self.dropout(layer(outputs))
        return outputs

class FIM(torch.nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        activation: str = 'relu'
    ) -> None:
        super(FIM, self).__init__()
        print('num_units: ', len(num_units), 'num_inputs', num_inputs)
        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.activation = activation

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1, 
                num_units=self.num_units[i], 
                dropout=self.dropout, 
                hidden_sizes=self.hidden_sizes,
                activation=self.activation
            )
            for i in range(num_inputs)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1), requires_grad=True)

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print('inside FIM forward')
        individual_outputs = self.calc_outputs(inputs)
        # print('individual_outputs: ', len(individual_outputs))
        # print('individual_outputs[0]: ', individual_outputs[0].shape)
        # exit(0)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out).unsqueeze(1)

        # print('dropout_out: ', dropout_out.shape)

        # check_result = torch.sum(torch.sum(dropout_out, dim=-1) + self._bias).item()
        # print('check_result:', check_result)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out
