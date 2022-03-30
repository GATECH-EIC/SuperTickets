from pruners import *
from models import masked_layers

from tqdm import tqdm
import torch
import numpy as np

def load_pruner(method):
    prune_methods = {
        'rand' : Rand,
        'mag' : Mag,
        'snip' : SNIP,
        'grasp': GraSP,
        'synflow' : SynFlow,
    }
    return prune_methods[method]

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def prunable(module, batchnorm):
    r"""Returns boolean whether a module is prunable.
    """
    isprunable = isinstance(module, (masked_layers.Linear, masked_layers.Conv2d))
    if batchnorm:
        isprunable |= isinstance(module, (masked_layers.BatchNorm1d, masked_layers.BatchNorm2d))
    return isprunable

def masked_parameters(model, bias=False, batchnorm=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm), model.modules()):
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
            if param is not module.bias or bias is True:
                yield mask, param


def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False, return_stats=False, set_pruned_params_to_zero=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)

    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Set pruned params to zero
    if set_pruned_params_to_zero:
        pruner.set_zeros()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    print('Prune at {}% sparsity level: Total Params: {}; Remaining Params: {}'.format(round((1 - sparsity)*100), total_params, int(remaining_params)))
    if np.abs(remaining_params - total_params*sparsity) >= 30:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()

    if return_stats:
        remaining_params, total_params = pruner.meta_stats()
        return remaining_params, total_params

    return int(remaining_params), total_params