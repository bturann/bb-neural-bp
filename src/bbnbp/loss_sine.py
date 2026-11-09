"""Auxiliary sine parity surrogate adapted from Liu and Poulin (2019), Eq. 8."""

import math

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


class SineParityLoss(nn.Module):
    def __init__(self,problem,beta_logical=5.0):
        super().__init__()
        if beta_logical<0:raise ValueError("Logical weight must be nonnegative")
        matrix=sp.vstack([problem.H,problem.L],format="csr")
        self.n_checks=problem.H.shape[0]
        self.n_rows,self.n_vars=matrix.shape
        self.beta_logical=beta_logical
        self.register_buffer("rows",torch.from_numpy(np.repeat(np.arange(self.n_rows),np.diff(matrix.indptr))))
        self.register_buffer("columns",torch.from_numpy(matrix.indices.astype(np.int64)))

    def forward(self,probabilities,faults):
        if probabilities.ndim!=2 or probabilities.shape!=faults.shape or probabilities.shape[1]!=self.n_vars:
            raise ValueError("Sine loss requires matching [batch, variables] inputs")
        values=probabilities+faults.to(probabilities.dtype)
        totals=probabilities.new_zeros((len(values),self.n_rows)).index_add(1,self.rows,values[:,self.columns])
        row_loss=torch.sin((math.pi/2)*totals).abs()
        return row_loss[:,:self.n_checks].sum(1)+self.beta_logical*row_loss[:,self.n_checks:].sum(1)
