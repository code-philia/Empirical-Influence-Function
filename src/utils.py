#! /usr/bin/env python3
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math
import pickle
import torch
from torch.autograd import grad
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Tuple, Union, Callable

@torch.inference_mode()
def calc_loss(model: nn.Module,
              criterion: nn.Module,
              dl: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    ) -> np.ndarray:
    '''
    Compute L for each sample in the dataset

    Arguments:
        model: torch.nn.Module
        criterion: loss function (output, target) -> scalar loss
        dl: dataloader or a tuple of (input, target)
        param_filter_fn: Optional function to select subset of parameters
                         e.g. lambda name, param: 'last_layer' in name
    Returns:
        loss_all: np.ndarray – loss of selected params
    '''
    loss_all = []
    model.eval()

    if isinstance(dl, tuple) and len(dl) == 2:
        iterable = [dl]
    else:
        iterable = dl

    with torch.no_grad():
        for inputs, targets in tqdm(iterable, desc="Computing losses"):
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)

            outputs = model(inputs)
            losses = criterion(outputs, targets)  # must be reduction='none'
            loss_all.extend(losses.detach().cpu().numpy())

    return np.array(loss_all)

def grad_loss(model: nn.Module,
              criterion: nn.Module,
              dl: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]],
              param_filter_fn: Callable[[str, nn.Parameter], bool] = None
    ) -> List[List[torch.Tensor]]:
    '''
    Compute dL/dθ for each sample in the dataset

    Arguments:
        model: torch.nn.Module
        criterion: loss function (output, target) -> scalar loss
        dl: dataloader or a tuple of (input, target)
        param_filter_fn: Optional function to select subset of parameters
                         e.g. lambda name, param: 'last_layer' in name

    Returns:
        grad_all: List[List[Tensor]] – each inner list contains gradients of selected params for one sample
    '''
    grad_all = []
    model.eval()

    # Get parameters to differentiate
    named_params = list(model.named_parameters())
    if param_filter_fn is not None:
        selected_params = [p for n, p in named_params if param_filter_fn(n, p) and p.requires_grad]
    else:
        selected_params = [p for _, p in named_params if p.requires_grad]

    # Prepare iterable: DataLoader or single (input, target) pair
    if isinstance(dl, tuple) and len(dl) == 2:
        iterable = [dl]
    else:
        iterable = dl

    for inputs, targets in tqdm(iterable, desc="Computing gradients"):
        inputs = inputs.to(next(model.parameters()).device)
        targets = targets.to(next(model.parameters()).device)

        model.zero_grad()
        with torch.set_grad_enabled(True):
            output = model(inputs)
            loss = criterion(output, targets).sum()
            grad_this = grad(loss, selected_params, create_graph=False)
            grad_all.append([g.detach().cpu() for g in grad_this])

    return grad_all


def inverse_hessian_product(model: nn.Module,
                            criterion:nn.Module,
                            v: List[torch.Tensor],
                            dl_tr: DataLoader,
                            param_filter_fn: Callable[[str, nn.Parameter], bool] = None,
                            scale=500,
                            damping=0.01,
    ) -> List[torch.Tensor]:
    """
        Get grad(test) H^-1. v is grad(test)
        Arguments:
            model: torch NN, model used to evaluate the dataset
            criterion: loss function
            v: vector you want to multiply with H-1
            dl_tr: torch Dataloader, can load the training dataset
            damping: float, dampening factor "chosen to be roughly the size of the most negative eigenvalue of the empirical Hessian (so that it becomes PSD)."
            scale: float, scaling factor, "the scale parameter scales the maximum eigenvalue to < 1 so that our Taylor approximation converges, otherwise h_estimate get NaN"
        Returns:
            h_estimate: List of tensor, s_test
    """
    model.eval()

    # Select parameters
    named_params = list(model.named_parameters())
    if param_filter_fn:
        selected_params = [p for n, p in named_params if param_filter_fn(n, p) and p.requires_grad]
    else:
        selected_params = [p for _, p in named_params if p.requires_grad]

    assert len(selected_params) == len(v), "Length of v must match number of selected parameters"

    cur_estimate = [ve.clone() for ve in v]

    for x, t in tqdm(dl_tr, desc="Estimating inverse hessian vector product"):
        x, t = x.to(next(model.parameters()).device), t.to(next(model.parameters()).device)

        model.zero_grad()
        output = model(x)
        loss = criterion(output, t).sum()

        hv = hessian_vector_product(loss, selected_params, cur_estimate)  # H @ estimate
        # LiSSA update rule
        cur_estimate = [
            v_i + (1 - damping) * ce_i - hv_i.detach().cpu() / scale
            for v_i, ce_i, hv_i in zip(v, cur_estimate, hv)
        ]

    # Final correction for scale
    inverse_hvp = [est.detach().cpu() / scale for est in cur_estimate]
    return inverse_hvp


def hessian_vector_product(y: torch.Tensor,
                           x: List[torch.Tensor],
                           v: List[torch.Tensor]
                           ) -> Tuple[torch.Tensor]:
    """
    Efficient computation of Hessian-vector product H·v using double backprop.

    Args:
        y: scalar tensor (e.g. loss)
        x: list of tensors – parameters to compute Hessian w.r.t.
        v: list of tensors – same shape as x, to multiply with Hessian
    Returns:
        Hessian-vector product
    """
    if len(x) != len(v):
        raise ValueError("x and v must have the same length.")
    if not isinstance(x, (list, tuple)) or not isinstance(v, (list, tuple)):
        raise TypeError("x and v must be lists or tuples of tensors.")

    grads = grad(y, x, retain_graph=True, create_graph=True)

    # ⟨∇y, v⟩ scalar
    prod = sum(torch.sum(g * v_i.to(g.device).detach()) for g, v_i in zip(grads, v))

    # ∇⟨∇y, v⟩ = H·v
    hvp = grad(prod, x, retain_graph=False)

    return hvp



