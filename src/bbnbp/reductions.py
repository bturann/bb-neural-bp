"""Deterministic gather/reduce primitives for backends without atomic-scatter support."""

import numpy as np
import torch


def inverse_slots(indices, n_inputs, active=None):
    ids = np.asarray(indices, dtype=np.int64).ravel()
    keep = np.ones(len(ids), dtype=bool) if active is None else np.asarray(active, dtype=bool).ravel()
    positions = np.flatnonzero(keep)
    positions = positions[np.argsort(ids[positions], kind="stable")]
    counts = np.bincount(ids[positions], minlength=n_inputs)
    width = max(1, int(counts.max(initial=0)))
    result = np.full((n_inputs, width), len(ids), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)])
    for i in np.flatnonzero(counts):
        result[i, :counts[i]] = positions[starts[i]:starts[i+1]]
    return torch.from_numpy(result)


class DeterministicGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, inverse):
        ctx.save_for_backward(inverse)
        return values.index_select(1, indices)

    @staticmethod
    def backward(ctx, gradient):
        inverse, = ctx.saved_tensors
        padded = torch.cat([gradient, gradient.new_zeros((len(gradient), 1))], dim=1)
        gathered = padded.index_select(1, inverse.reshape(-1)).reshape(len(gradient), *inverse.shape)
        return gathered.sum(-1), None, None


class DeterministicSegmentSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, inverse, indices, active):
        ctx.save_for_backward(indices, active)
        padded = torch.cat([values, values.new_zeros((len(values), 1))], dim=1)
        gathered = padded.index_select(1, inverse.reshape(-1)).reshape(len(values), *inverse.shape)
        return gathered.sum(-1)

    @staticmethod
    def backward(ctx, gradient):
        indices, active = ctx.saved_tensors
        return gradient.index_select(1, indices)*active, None, None, None
