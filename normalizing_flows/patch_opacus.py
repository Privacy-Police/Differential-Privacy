from typing import Union

import numpy as np
import torch
from opacus.layers.dp_lstm import LSTMLinear
from opacus.layers.dp_multihead_attention import SequenceBias
from torch import nn
from torch.functional import F

from opacus.supported_layers_grad_samplers import _supported_layers_grad_samplers


def _compute_masked_linear_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    #print('A.shape', A.shape)
    #print('B.shape', B.shape)

    gs = torch.einsum("n...i,n...j->n...ij", B, A)
    #print('gs.shape', gs.shape)
    _create_or_extend_grad_sample(
        layer.linear.weight, torch.einsum("n...ij->nij", gs), batch_dim
    )
    #print('gs_after.shape', torch.einsum("n...ij->nij", gs).shape)
    if layer.linear.bias is not None:

        _create_or_extend_grad_sample(
            layer.linear.bias,
            torch.einsum("n...k->nk", B),
            batch_dim,  # pyre-ignore[6] We know layer.bias is not None
        )


# Patching Opacus to support MaskedLinear layer
_supported_layers_grad_samplers['MaskedLinear'] = _compute_masked_linear_grad_sample
