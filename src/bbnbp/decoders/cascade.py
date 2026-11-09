"""An auxiliary two-pass Stage-1 decoder that preserves first-pass convergence."""

import numpy as np

from .base import validate_syndromes
from ..noise.base import parity


class CascadeDecoder:
    def __init__(self,first,second):
        self.first,self.second=first,second
        self.problem=first.problem
        other=second.problem
        for name in ("H","L"):
            a,b=getattr(self.problem,name),getattr(other,name)
            if a.shape!=b.shape or (a!=b).nnz:raise ValueError("Cascade passes must share H and L")
        if not np.array_equal(self.problem.priors,other.priors):raise ValueError("Cascade passes must share priors")
        self.T=first.T+second.T
        self.name=f"{first.name}_then_{second.name}"
        self.last_iterations=np.empty(0,dtype=np.int64)
        self.last_used_second=np.empty(0,dtype=bool)
        self.last_first_iterations=np.empty(0,dtype=np.int64)
        self.last_second_iterations=np.empty(0,dtype=np.int64)

    def decode_batch(self,s):
        s=validate_syndromes(s,self.problem.H.shape[0])
        first_hat,converged=self.first.decode_batch(s)
        if not np.array_equal((parity(self.problem.H,first_hat)==s).all(1),converged):
            raise AssertionError("First cascade pass reported incorrect convergence")
        out=first_hat.copy()
        converged=converged.copy()
        self.last_first_iterations=self.first.last_iterations.copy()
        self.last_second_iterations=np.zeros(len(s),dtype=np.int64)
        self.last_used_second=~converged
        if self.last_used_second.any():
            indices=self.last_used_second
            second_hat,second_conv=self.second.decode_batch(s[indices])
            if not np.array_equal((parity(self.problem.H,second_hat)==s[indices]).all(1),second_conv):
                raise AssertionError("Second cascade pass reported incorrect convergence")
            out[indices]=second_hat
            converged[indices]=second_conv
            self.last_second_iterations[indices]=self.second.last_iterations
        self.last_iterations=self.last_first_iterations+self.last_second_iterations
        return out,converged
