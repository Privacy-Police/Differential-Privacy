from typing import Union, Tuple
from functools import partial
import numpy as np
import torch
from opacus.layers.dp_lstm import LSTMLinear
from opacus.layers.dp_multihead_attention import SequenceBias
from torch import nn
from torch.functional import F

from opacus.supported_layers_grad_samplers import _supported_layers_grad_samplers, _create_or_extend_grad_sample
from opacus.utils.module_inspection import get_layer_type, requires_grad
import opacus.autograd_grad_sample as autograd_grad_sample


def _compute_masked_linear_grad_sample(layer, A, B, batch_dim=0) -> None:
    """
    Computes per sample gradients for ``nn.MaskedLinear`` layer

    Args:
        layer: Layer (nn.MaskedLinear)
        A: Activations (Tensor)
        B: Backpropagations (Tensor)
        batch_dim: Batch dimension position (int)
    """
    gs = torch.einsum("n...i,n...j->n...ij", B, A)
    _create_or_extend_grad_sample(
        layer.linear.weight, torch.einsum("n...ij->nij", gs), batch_dim
    )
    if layer.linear.bias is not None:
        _create_or_extend_grad_sample(
            layer.linear.bias,
            torch.einsum("n...k->nk", B),
            batch_dim,  # pyre-ignore[6] We know layer.bias is not None
        )

def _capture_activations(
    layer: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]
):
    r"""Forward hook handler captures and saves activations flowing into the
    ``layer`` in ``layer.activations`` during forward pass.

    Args:
        layer: Layer to capture the activations in.
        inputs: Inputs to the ``layer``.
        outputs: Outputs of the ``layer``.
    """
    layer_type = get_layer_type(layer)
    if (
        not requires_grad(layer, recurse=True) # <- Patched
        or layer_type not in _supported_layers_grad_samplers.keys()
        or not layer.training
    ):
        return

    if autograd_grad_sample._hooks_disabled:
        return

    if get_layer_type(layer) not in _supported_layers_grad_samplers.keys():
        raise ValueError("Hook installed on unsupported layer")

    if not hasattr(layer, "activations"):
        layer.activations = []

    layer.activations.append(inputs[0].detach())

def _compute_grad_sample(
        layer: nn.Module, backprops: torch.Tensor, loss_reduction: str, batch_first: bool
):
    r"""Computes per-sample gradients with respect to the parameters of the
    ``layer`` (if supported), and saves them in ``param.grad_sample``.

    Args:
        layer: Layer to capture per-sample gradients in.
        backprops: Back propagated gradients captured by the backward hook.
        loss_reduction: Indicates if the loss reduction (for aggregating the
            gradients) is a sum or a mean operation. Can take values ``sum``
            or ``mean``.
        batch_first: Flag to indicate if the input tensor to the corresponding
            module has the first dimension represent the batch, for example of
            shape ``[batch_size, ..., ...]``. Set to True if batch appears in
            first dimension else set to False (``batch_first=False`` implies
            that the batch is always in the second dimension).
    """
    layer_type = get_layer_type(layer)
    if (
        not requires_grad(layer, recurse=True) # <- Patched
        or layer_type not in _supported_layers_grad_samplers.keys()
        or not layer.training
    ):
        return

    if not hasattr(layer, "activations"):
        raise ValueError(
            f"No activations detected for {type(layer)},"
            " run forward after add_hooks(model)"
        )

    # Outside of the LSTM there is "batch_first" but not for the Linear inside the LSTM
    batch_dim = 0 if batch_first or type(layer) is LSTMLinear else 1

    if isinstance(layer.activations, list):
        A = layer.activations.pop()
    else:
        A = layer.activations

    n = A.shape[batch_dim]
    if loss_reduction == "mean":
        B = backprops * n
    elif loss_reduction == "sum":
        B = backprops
    else:
        raise ValueError(
            f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported"
        )

    # rearrange the blob dimensions
    if batch_dim != 0:
        A = A.permute([batch_dim] + [x for x in range(A.dim()) if x != batch_dim])
        B = B.permute([batch_dim] + [x for x in range(B.dim()) if x != batch_dim])
    # compute grad sample for  individual layers
    compute_layer_grad_sample = _supported_layers_grad_samplers.get(
        get_layer_type(layer)
    )
    compute_layer_grad_sample(layer, A, B)


# Patching Opacus to support MaskedLinear layer
_supported_layers_grad_samplers['MaskedLinear'] = _compute_masked_linear_grad_sample

# Patching Opacus functions to recurse into the layers
autograd_grad_sample._capture_activations = _capture_activations
autograd_grad_sample._compute_grad_sample = _compute_grad_sample

print('Opacus patch is done!')
