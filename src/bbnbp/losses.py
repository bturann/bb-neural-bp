"""Degeneracy-aware parity loss with stable products and exact zero-factor gradients."""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


class _ParityProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, factors, rows, columns, n_rows):
        values = factors[:, columns]
        zero = values == 0
        log_values = torch.log(torch.where(zero, torch.ones_like(values), values.abs()))
        shape = (len(factors), n_rows)
        logs = factors.new_zeros(shape).index_add(1, rows, log_values)
        negatives = factors.new_zeros(shape).index_add(1, rows, (values < 0).to(factors.dtype))
        zeros = factors.new_zeros(shape).index_add(1, rows, zero.to(factors.dtype))
        signs = 1-2*(negatives % 2)
        nonzero_product = signs*logs.exp()
        ctx.save_for_backward(factors, rows, columns, logs, signs, zeros)
        return torch.where(zeros > 0, torch.zeros_like(nonzero_product), nonzero_product)

    @staticmethod
    def backward(ctx, grad_output):
        factors, rows, columns, logs, signs, zeros = ctx.saved_tensors
        values = factors[:, columns]
        local_zero = values == 0
        local_log = torch.log(torch.where(local_zero, torch.ones_like(values), values.abs()))
        local_sign = torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))
        # Compute excluded products in log space, including the one-zero case.
        derivative = signs[:, rows]*local_sign*(logs[:, rows]-local_log).exp()
        active = (zeros[:, rows] == 0) | ((zeros[:, rows] == 1) & local_zero)
        derivative = torch.where(active, derivative, torch.zeros_like(derivative))
        gradient = torch.zeros_like(factors).index_add(1, columns, grad_output[:, rows]*derivative)
        return gradient, None, None, None


class ParityLoss(nn.Module):
    def __init__(self, problem, beta_logical=5.0, eps=1e-7, prob_floor=1e-12):
        super().__init__()
        if beta_logical < 0 or not 0 < eps < 0.5 or not 0 < prob_floor < 0.5:
            raise ValueError("Invalid parity loss parameters")
        matrix = sp.vstack([problem.H, problem.L], format="csr")
        self.n_checks = problem.H.shape[0]
        self.n_rows, self.n_vars = matrix.shape
        self.beta_logical, self.eps, self.prob_floor = beta_logical, eps, prob_floor
        self.register_buffer("rows", torch.from_numpy(np.repeat(np.arange(self.n_rows), np.diff(matrix.indptr))))
        self.register_buffer("columns", torch.from_numpy(matrix.indices.astype(np.int64)))

    def forward(self, probabilities, faults):
        """Return the parity objective for each shot, without reducing the batch."""
        if probabilities.ndim != 2 or probabilities.shape != faults.shape or probabilities.shape[1] != self.n_vars:
            raise ValueError("Loss inputs must have matching [batch, variables] shapes")
        faults = faults.to(dtype=probabilities.dtype)
        q = (probabilities+faults-2*probabilities*faults).clamp(self.eps, 1-self.eps)
        products = _ParityProduct.apply(1-2*q, self.rows, self.columns, self.n_rows)
        even = ((1+products)/2).clamp_min(self.prob_floor)
        row_loss = -even.log()
        return row_loss[:, :self.n_checks].sum(1)+self.beta_logical*row_loss[:, self.n_checks:].sum(1)


def bitwise_bce(probabilities, faults, eps=1e-7):
    """Per-shot bitwise BCE ablation, using the same batch reduction as parity loss."""
    if probabilities.shape != faults.shape or probabilities.ndim != 2:
        raise ValueError("BCE inputs must have matching [batch, variables] shapes")
    return F.binary_cross_entropy(probabilities.clamp(eps, 1-eps), faults.to(probabilities.dtype),
                                  reduction="none").mean(1)


def multi_loss(outputs, faults, per_shot_loss, sample_weights=None):
    if not outputs:
        raise ValueError("Multi-loss requires at least one unrolled iteration")
    total = outputs[0].new_zeros(len(faults))
    for iteration, probabilities in enumerate(outputs, start=1):
        total = total+iteration*per_shot_loss(probabilities, faults)
    total = total/(len(outputs)*(len(outputs)+1)/2)
    if sample_weights is None:
        return total.mean()
    if sample_weights.shape != total.shape or not (sample_weights > 0).all():
        raise ValueError("Sample weights must be a positive vector matching the batch")
    return (sample_weights*total).sum()/sample_weights.sum()
