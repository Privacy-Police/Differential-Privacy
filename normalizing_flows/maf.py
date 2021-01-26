from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from made import MADE

# This implementation of MAF is copied from: https://github.com/e-hulten/maf

class MAFLayer(nn.Module):
    def __init__(self, dim: int, hidden_dims: List[int], reverse: bool):
        super(MAFLayer, self).__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims, gaussian=True)
        self.reverse = reverse

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.made(x.float())
        mu, logp = torch.chunk(out, 2, dim=1)
        u = (x - mu) * torch.exp(0.5 * logp)
        u = u.flip(dims=(1,)) if self.reverse else u
        log_det = 0.5 * torch.sum(logp, dim=1)
        return u, log_det

    def backward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        u = u.flip(dims=(1,)) if self.reverse else u
        x = torch.zeros_like(u)
        for dim in range(self.dim):
            out = self.made(x)
            mu, logp = torch.chunk(out, 2, dim=1)
            mod_logp = torch.clamp(-0.5 * logp, max=10)
            x[:, dim] = mu[:, dim] + u[:, dim] * torch.exp(mod_logp[:, dim])
        log_det = torch.sum(mod_logp, axis=1)
        return x, log_det


class BatchNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps  # torch.mean((x - m) ** 2, axis=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean.clone()
            v = self.batch_var.clone()

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v))
        return x_hat, log_det

    def backward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean
            v = self.batch_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det

    def set_batch_stats_func(self, x):
        print("setting batch stats for validation")
        self.batch_mean = x.mean(dim=0)
        self.batch_var = x.var(dim=0) + self.eps



class MAF(nn.Module):
    def __init__(
        self, dim: int, n_layers: int, hidden_dims: List[int], use_reverse: bool = True
    ):
        """
        Args:
            dim: Dimension of input. E.g.: dim = 784 when using MNIST.
            n_layers: Number of layers in the MAF (= number of stacked MADEs).
            hidden_dims: List with of sizes of the hidden layers in each MADE.
            use_reverse: Whether to reverse the input vector in each MADE.
        """
        super().__init__()
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse))
            self.layers.append(BatchNormLayer(dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0])
        # Forward pass.
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det

        return x, log_det_sum

    def backward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        log_det_sum = torch.zeros(x.shape[0])
        # Backward pass.
        for layer in reversed(self.layers):
            x, log_det = layer.backward(x)
            log_det_sum += log_det

        return x, log_det_sum

