"""Torch model definitons for the Deep Clustering Survival Machines model

This includes definitons for the Deep Clustering Survival Machines module.
The main interface is the InterpretableRiskClusteringIntelligence class which inherits
from torch.nn.Module.

"""

import torch.nn as nn
import torch
import numpy as np
from .fim import FIM

class InterpretableRiskClusteringIntelligenceTorch(nn.Module):
    """A Torch implementation of Deep Clustering Survival Machines model.

  This is an implementation of Deep Clustering Survival Machines model in torch.
  It inherits from the torch.nn.Module class and includes references to the
  representation learning MLP, the parameters of the underlying distributions
  and the forward function which is called whenver data is passed to the
  module. Each of the parameters belongs to nn.Parameters and torch automatically
  keeps track and computes gradients for them.

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
  activation: str
      Choice of activation function for the MLP representation.
      One of 'ReLU6', 'ReLU' or 'SeLU'.
      Default is 'ReLU6'.
  dist: str
      Choice of the underlying survival distributions.
      One of 'Weibull', 'LogNormal'.
      Default is 'Weibull'.
  temp: float
      The logits for the alpha are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.

  """

    def _init_iris_layers(self, lastdim):

        if self.is_seed:  # if is_seed is true, means we use the random seed to fix the initialization
            print('random seed for torch model initialization is: ', self.random_state)
            torch.manual_seed(self.random_state)  # fix the initialization
        if self.dist in ['Weibull']:
            self.act = nn.SELU()
            print('self.act', self.act)
            if self.fix:  # means using fixed base distribution
                self.shape = nn.ParameterDict({str(r + 1): nn.Parameter(torch.randn(self.k, requires_grad=True))
                                               for r in range(self.risks)})  # .cuda()
                self.scale = nn.ParameterDict({str(r + 1): nn.Parameter(torch.randn(self.k, requires_grad=True))
                                               for r in range(self.risks)})  # .cuda()                       
            else:
                self.shape = nn.ParameterDict({str(r + 1): nn.Parameter(-torch.ones(self.k))
                                               for r in range(self.risks)})  # .cuda()
                self.scale = nn.ParameterDict({str(r + 1): nn.Parameter(-torch.ones(self.k))
                                               for r in range(self.risks)})  # .cuda()
               
        else:
            raise NotImplementedError('Distribution: ' + self.dist + ' not implemented' +
                                      ' yet.')

        hidden_dims = [50] # [50]
        dropout = 0 # 0.1
        feature_dropout = 0 # 0.05

        self.alpha = nn.ModuleDict({
            str(r + 1): nn.ModuleList([
                FIM(
                    num_inputs=self.inputdim,
                    hidden_sizes=hidden_dims,
                    activation='exu', # exu
                    dropout=dropout,
                    feature_dropout=feature_dropout,
                    num_units=self.num_units
                )
                for _ in range(self.k)
            ])
            for r in range(self.risks)
        })

        if self.fix == False:  
            self.scaleg = nn.ModuleDict({str(r + 1): nn.Sequential(
                nn.Linear(lastdim, self.k, bias=True)
            ) for r in range(self.risks)})  # .cuda()

            self.shapeg = nn.ModuleDict({str(r + 1): nn.Sequential(
                nn.Linear(lastdim, self.k, bias=True)
            ) for r in range(self.risks)})  # .cuda()

    def __init__(self, inputdim, k, layers=None, dist='Weibull',
                 temp=1000., discount=1.0, optimizer='Adam',
                 risks=1, random_state=42, num_units=[], fix=False, is_seed=False):
        super(InterpretableRiskClusteringIntelligenceTorch, self).__init__()

        self.k = k
        self.dist = dist
        self.temp = 1 # 1000
        self.discount = float(discount)
        self.optimizer = optimizer
        self.risks = risks
        self.num_units = num_units # size of the hidden layers for each feature
        self.inputdim = inputdim

        if layers is None: layers = []
        self.layers = layers

        if len(layers) == 0:
            lastdim = inputdim
        else:
            lastdim = layers[-1]

        self.random_state = random_state
        self.fix = fix
        self.is_seed = is_seed

        self._init_iris_layers(lastdim)

    def forward(self, x, risk='1'):
        """The forward function that is called when data is passed through IRIS.

    Args:
      x:
        a torch.tensor of the input features.

    """

        dim = x.shape[0]

        logits = []

        for k in range(self.k):
            alpha = self.alpha[risk][k](x)[0] / self.temp
            logits.append(alpha)

        logits = torch.cat(logits, dim=1)

        if self.fix:  # means using fixed base distributions
            return (self.shape[risk].expand(dim, -1).cuda(),
                    self.scale[risk].expand(dim, -1).cuda(),
                    logits)
        else:
            return (self.act(self.shapeg[risk](x)) + self.shape[risk].expand(dim, -1),
                    self.act(self.scaleg[risk](x)) + self.scale[risk].expand(dim, -1),
                    logits)

    def get_shape_scale(self, risk='1'):
        return self.shape[risk], self.scale[risk]