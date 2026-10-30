import numpy as np
import pytest
import torch

from bbnbp.codes import bb_matrices, edge_orbits
from bbnbp.graph import build_tanner, weight_map
from bbnbp.noise.circuit_level import CircuitLevelSampler


def test_padding_and_weight_maps():
    h = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    graph = build_tanner(h)
    np.testing.assert_array_equal(graph.chk_mask.numpy(), [[1, 1], [1, 0], [0, 0]])
    np.testing.assert_array_equal(graph.chk_idx[graph.chk_mask].numpy(), np.nonzero(h)[1])
    np.testing.assert_array_equal(graph.edge_ids[graph.chk_mask].numpy(), np.arange(3))
    assert not weight_map(graph, "scalar").any()
    np.testing.assert_array_equal(weight_map(graph, "full")[graph.chk_mask].numpy(), [0, 1, 2])
    with pytest.raises(ValueError, match="round metadata"):
        weight_map(graph, "struct")


def test_orbit_weight_count_and_padding_map():
    h, _ = bb_matrices("bb72")
    graph = build_tanner(h)
    mapping = weight_map(graph, "orbit", edge_orbits(h, 6, 6))
    assert torch.unique(mapping[graph.chk_mask]).numel() == 6
    with pytest.raises(ValueError, match="per Tanner edge"):
        weight_map(graph, "orbit", np.zeros(10, dtype=int))


def test_circuit_structure_uses_explicit_temporal_anchors():
    sampler = CircuitLevelSampler("bb72", 2, 0.001)
    problem = sampler.problem
    rounds = np.asarray(problem.meta["detector_rounds"])
    graph = build_tanner(problem.H, detector_rounds=rounds)
    mapping = weight_map(graph, "struct")[graph.chk_mask].numpy()
    h = problem.H
    ci = np.repeat(np.arange(h.shape[0]), np.diff(h.indptr))
    vd = np.bincount(h.indices, minlength=h.shape[1])
    columns = h.tocsc()
    for edge in range(h.nnz):
        c, v = ci[edge], h.indices[edge]
        support = columns.indices[columns.indptr[v]:columns.indptr[v+1]]
        expected = (int(np.diff(h.indptr)[c]), int(vd[v]), int(rounds[c]-rounds[support].min()))
        assert graph.struct_features[mapping[edge]] == expected
