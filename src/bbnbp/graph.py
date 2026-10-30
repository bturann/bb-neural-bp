"""Padded Tanner tensors and explicit message-weight sharing maps."""

from dataclasses import dataclass

import numpy as np
import torch

from .noise.base import binary_csr


@dataclass
class TannerTensors:
    chk_idx: torch.Tensor
    chk_mask: torch.Tensor
    edge_ids: torch.Tensor
    n_vars: int
    struct_ids: torch.Tensor | None = None
    struct_features: tuple[tuple[int, int, int], ...] | None = None


def build_tanner(H, *, detector_rounds=None) -> TannerTensors:
    h = binary_csr(H)
    degrees = np.diff(h.indptr)
    width = max(2, int(degrees.max(initial=0)))
    index = np.zeros((h.shape[0], width), dtype=np.int64)
    mask = np.zeros_like(index, dtype=bool)
    edges = np.full_like(index, -1)
    for c in range(h.shape[0]):
        start, end = h.indptr[c:c+2]
        index[c, :end-start] = h.indices[start:end]
        mask[c, :end-start] = True
        edges[c, :end-start] = np.arange(start, end)
    result = TannerTensors(torch.from_numpy(index), torch.from_numpy(mask),
                           torch.from_numpy(edges), h.shape[1])
    if detector_rounds is not None:
        rounds = np.asarray(detector_rounds)
        if rounds.shape != (h.shape[0],) or not np.issubdtype(rounds.dtype, np.integer):
            raise ValueError("Detector rounds must be an integer per check")
        ci = np.repeat(np.arange(h.shape[0]), degrees)
        vi = h.indices
        var_degree = np.bincount(vi, minlength=h.shape[1])
        anchor = np.full(h.shape[1], int(rounds.max(initial=0))+1, dtype=np.int64)
        np.minimum.at(anchor, vi, rounds[ci])
        # DEM mechanisms may combine multiple physical fault locations. Their
        # earliest affected detector gives a reproducible temporal anchor, not
        # a claimed physical gate time or an exact code automorphism orbit.
        features = np.column_stack((degrees[ci], var_degree[vi], rounds[ci]-anchor[vi]))
        unique, ids = np.unique(features, axis=0, return_inverse=True)
        padded = np.zeros_like(index)
        padded[mask] = ids
        result.struct_ids = torch.from_numpy(padded)
        result.struct_features = tuple(tuple(map(int, row)) for row in unique)
    return result


def weight_map(tt: TannerTensors, sharing: str, orbits=None) -> torch.Tensor:
    if sharing == "scalar":
        return torch.zeros_like(tt.edge_ids)
    if sharing == "full":
        return tt.edge_ids.clamp_min(0)
    if sharing == "struct":
        if tt.struct_ids is None:
            raise ValueError("Structural tying requires explicit detector round metadata")
        return tt.struct_ids.clone()
    if sharing == "orbit":
        ids = np.asarray(orbits)
        count = int(tt.chk_mask.sum())
        if ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
            raise ValueError("Orbit IDs must be a nonnegative integer per Tanner edge")
        if not np.array_equal(np.unique(ids), np.arange(ids.max(initial=-1)+1)):
            raise ValueError("Orbit IDs must be contiguous from zero")
        result = torch.zeros_like(tt.edge_ids)
        result[tt.chk_mask] = torch.from_numpy(ids.astype(np.int64))
        return result
    raise ValueError(f"Unknown weight sharing {sharing!r}")
