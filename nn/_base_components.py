import collections
from typing import Callable, Iterable, List, Optional

import torch
from torch import nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from scSVAE.hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform
from scvi.nn import FCLayers


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def reparameterize_vonmises(z_mean, z_var):
    q_z = VonMisesFisher(z_mean, z_var)
    return q_z.rsample()


def identity(x):
    return x


# Encoder
class EncoderSVAE(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        When `None`, defaults to `torch.exp`.
    **kwargs
        Keyword args for :class:`~scvi.module._base.FCLayers`
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            distribution: str = "normal",
            var_eps: float = 1e-4,
            var_activation: Optional[Callable] = None,
            activation: torch.nn.functional = F.relu,
            **kwargs,
    ):
        super().__init__()

        self.activation = activation

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity
        self.var_activation = torch.exp if var_activation is None else var_activation

        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(n_input, n_hidden * 2)
        self.fc_e1 = nn.Linear(n_hidden * 2, n_hidden)

        # compute mean and concentration of the von Mises-Fisher
        self.fc_mean = nn.Linear(n_hidden, n_output)
        self.fc_var = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))

        # Parameters for latent distribution

        # compute mean and concentration of the von Mises-Fisher
        z_mean = self.fc_mean(x)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(self.fc_var(x)) + 1

        latent = self.z_transformation(reparameterize_vonmises(z_mean, z_var))

        return z_mean, z_var, latent


# Decoder
