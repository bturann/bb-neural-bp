"""Differentiable flooding min-sum, with first-convergence freezing at inference."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .._construction import CODES
from ..codes import edge_orbits
from ..graph import build_tanner, weight_map
from ..reductions import DeterministicGather, DeterministicSegmentSum, inverse_slots
from .base import validate_syndromes


class NeuralMinSum(nn.Module):
    name = "nbp"

    def __init__(self, problem, T: int, sharing: str, ms_scaling: float = 0.625,
                 msg_clip: float = 20.0, *, orbits=None, dtype=torch.float32):
        super().__init__()
        if type(T) is not int or T < 1:
            raise ValueError("T must be a positive integer")
        if not math.isfinite(ms_scaling) or not 0 < ms_scaling <= 1:
            raise ValueError("Initial scaling must be in (0, 1]")
        if math.isnan(msg_clip) or msg_clip <= 0:
            raise ValueError("Message clip must be positive (infinity is an explicit diagnostic mode)")
        self.problem, self.T = problem, T
        self.sharing, self.msg_clip = sharing, float(msg_clip)
        if sharing == "orbit" and orbits is None:
            code = problem.meta.get("code")
            if problem.meta.get("noise_model") != "code_capacity" or code not in CODES:
                raise ValueError("Exact orbit tying is supported for code-capacity BB problems only")
            l, m, _ = CODES[code]
            orbits = edge_orbits(problem.H.toarray(), l, m)
        graph = build_tanner(problem.H, detector_rounds=problem.meta.get("detector_rounds"))
        mapping = weight_map(graph, sharing, orbits)
        self.struct_features = graph.struct_features
        for name, value in (("chk_idx", graph.chk_idx), ("chk_mask", graph.chk_mask),
                            ("edge_ids", graph.edge_ids), ("w_map", mapping)):
            self.register_buffer(name, value)
        self.register_buffer("llr", torch.tensor(problem.llr, dtype=torch.float64))
        n_weights = int(mapping[graph.chk_mask].max())+1 if graph.chk_mask.any() else 1
        self.w = nn.Parameter(torch.full((T, n_weights), float(ms_scaling), dtype=dtype))
        self.alpha = nn.Parameter(torch.ones(T, dtype=dtype))
        self.register_buffer("var_inverse", inverse_slots(graph.chk_idx.numpy(), graph.n_vars,
                                                           graph.chk_mask.numpy()), persistent=False)
        self.register_buffer("weight_inverse", inverse_slots(mapping.numpy(), n_weights,
                                                              graph.chk_mask.numpy()), persistent=False)
        self.last_hard_decisions = []
        self.last_iterations = np.empty(0, dtype=np.int64)

    def _apply(self, fn, recurse=True):
        # MPS cannot transfer float64 buffers. Recreate the current problem's
        # exact CPU/CUDA prior or its float32 MPS representation after moving.
        prior = self._buffers.pop("llr")
        try:
            result = super()._apply(fn, recurse=recurse)
        except BaseException:
            self.register_buffer("llr", prior)
            raise
        prior_dtype = torch.float32 if self.w.device.type == "mps" else torch.float64
        self.register_buffer("llr", torch.tensor(self.problem.llr, dtype=prior_dtype, device=self.w.device))
        return result

    def forward(self, s, llr=None, early_stop=False, readout="full_unroll"):
        if readout not in {"full_unroll", "first_converged"}:
            raise ValueError("Unknown neural training readout")
        if early_stop and torch.is_grad_enabled():
            raise ValueError("Early stopping is for inference; use no_grad or decode_batch")
        device, dtype = self.w.device, self.w.dtype
        syndrome = torch.as_tensor(s, device=device)
        if syndrome.ndim != 2 or syndrome.shape[1] != self.chk_idx.shape[0]:
            raise ValueError("Syndrome shape does not match the Tanner graph")
        if not torch.all((syndrome == 0) | (syndrome == 1)):
            raise ValueError("Syndromes must be binary")
        syndrome = syndrome.to(dtype=dtype)
        prior = torch.as_tensor(self.llr if llr is None else llr, device=device, dtype=dtype)
        if prior.shape != (self.problem.H.shape[1],) or not torch.isfinite(prior).all():
            raise ValueError("Prior LLRs must be a finite vector with one entry per variable")
        batch, checks = syndrome.shape
        n = self.problem.H.shape[1]
        messages = torch.zeros((batch, *self.chk_idx.shape), device=device, dtype=dtype)
        total = torch.zeros((batch, n), device=device, dtype=dtype)
        sign_s = (1-2*syndrome).unsqueeze(-1)
        variables = self.chk_idx[self.chk_mask]
        track_convergence = early_stop or readout == "first_converged"
        converged = ~syndrome.bool().any(dim=1) if track_convergence else torch.zeros(batch, device=device, dtype=torch.bool)
        frozen = torch.zeros((batch, n), dtype=torch.bool, device=device)
        iterations = torch.full((batch,), self.T, dtype=torch.long, device=device)
        if track_convergence:
            iterations[converged] = 0
        frozen_probabilities = torch.sigmoid(-prior).expand(batch, n) if readout == "first_converged" else None
        outputs, decisions = [], []
        hard = frozen
        for t in range(self.T):
            if early_stop and bool(converged.all()):
                break
            variable_values = self.alpha[t]*prior+total
            if device.type == "mps":
                gathered = DeterministicGather.apply(variable_values, self.chk_idx.reshape(-1), self.var_inverse)
                mu = gathered.reshape_as(messages)-messages
            else:
                mu = variable_values[:, self.chk_idx]-messages
            mu = torch.where(self.chk_mask, mu, torch.full_like(mu, 1e18))
            signs = torch.where(mu < 0, -torch.ones_like(mu), torch.ones_like(mu))
            magnitudes = mu.abs()
            minima = magnitudes.topk(2, dim=-1, largest=False).values
            extrinsic = torch.where(magnitudes == minima[..., :1], minima[..., 1:], minima[..., :1])
            if device.type == "mps":
                weights = DeterministicGather.apply(self.w[t:t+1], self.w_map.reshape(-1), self.weight_inverse)
                weights = weights.reshape(1, *self.w_map.shape)
            else:
                weights = self.w[t][self.w_map]
            messages = weights*sign_s*signs.prod(dim=-1, keepdim=True)*signs*extrinsic
            messages = messages.clamp(-self.msg_clip, self.msg_clip)*self.chk_mask
            if device.type == "mps":
                total = DeterministicSegmentSum.apply(messages.flatten(1), self.var_inverse,
                                                     self.chk_idx.reshape(-1), self.chk_mask.reshape(-1))
            else:
                total = torch.zeros_like(total).index_add(1, variables, messages[:, self.chk_mask])
            posterior = prior+total  # posterior always uses the unweighted prior
            hard = posterior < 0
            outputs.append(torch.sigmoid(-posterior))
            # Hard decisions use the LLR sign directly, avoiding sigmoid rounding at 0.5.
            decisions.append(hard.detach())
            if track_convergence:
                parity = (hard[:, self.chk_idx] & self.chk_mask).sum(dim=-1) % 2
                now = (parity == syndrome).all(dim=1)
                newly = now & ~converged
                frozen = torch.where(newly[:, None], hard, frozen)
                iterations = torch.where(newly, t+1, iterations)
                converged = converged | now
                if readout == "first_converged":
                    # Keep the first soft prediction attached to its gradient path.
                    # The parity objective still checks its logical correctness.
                    frozen_probabilities = torch.where(newly[:, None], outputs[-1], frozen_probabilities)
                    outputs[-1] = torch.where(converged[:, None], frozen_probabilities, outputs[-1])
        if not track_convergence:
            parity = (hard[:, self.chk_idx] & self.chk_mask).sum(dim=-1) % 2
            converged = (parity == syndrome).all(dim=1)
            frozen = hard
        self.last_hard_decisions = decisions
        self.last_e_hat = torch.where(converged[:, None], frozen, hard).detach()
        self.last_converged = converged.detach()
        self.last_iterations = iterations.detach().cpu().numpy().copy()
        return outputs

    def decode_batch(self, s):
        s = validate_syndromes(s, self.problem.H.shape[0])
        with torch.inference_mode():
            self.forward(s, early_stop=True)
        return self.last_e_hat.to(dtype=torch.uint8).cpu().numpy().copy(), self.last_converged.cpu().numpy().copy()

    def export_cpu(self):
        from .fast_nbp import FastNeuralBP
        return FastNeuralBP(self)
