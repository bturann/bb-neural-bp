import numpy as np
import pytest
import torch

from bbnbp.losses import ParityLoss, _ParityProduct, bitwise_bce, multi_loss
from bbnbp.noise.base import DecodingProblem
from bbnbp.noise.code_capacity import CodeCapacitySampler
from bbnbp.codes import bb_matrices, logicals


def test_degenerate_equivalence_and_logical_error():
    sampler = CodeCapacitySampler("bb72", 0.04)
    hx, hz = bb_matrices("bb72")
    _, lz = logicals("bb72")
    objective = ParityLoss(sampler.problem).double()
    predictions = torch.zeros((3, 72), dtype=torch.float64)
    faults = torch.from_numpy(np.stack([np.zeros(72, dtype=np.uint8), hz[0], lz[0]]))
    values = objective(predictions, faults)
    assert values[0] < 1e-3 and values[1] < 1e-3
    assert values[2] > 10


def test_parity_loss_float64_gradcheck():
    problem = DecodingProblem("toy", [[1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                                      [0, 0, 1, 1, 1, 0, 1, 0, 0, 0]],
                               [[0, 1, 0, 0, 0, 1, 0, 1, 0, 1]], [0.1]*10, {})
    loss = ParityLoss(problem).double()
    generator = torch.Generator().manual_seed(64)
    p = (0.1+0.3*torch.rand((2, 10), generator=generator, dtype=torch.float64)).requires_grad_()
    e = torch.randint(0, 2, (2, 10), generator=generator)
    assert torch.autograd.gradcheck(lambda x: loss(x, e), (p,), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("values", [[0.0, 0.3, -0.7], [0.0, 0.0, 0.7], [0.2, -0.4, 0.7]])
def test_zero_factor_gradients_match_direct_polynomial(values):
    first = torch.tensor([values], dtype=torch.float64, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    sparse = _ParityProduct.apply(first, torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]), 1)
    direct = second.prod(1, keepdim=True)
    sparse.sum().backward()
    direct.sum().backward()
    torch.testing.assert_close(sparse, direct, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(first.grad, second.grad, rtol=1e-12, atol=1e-12)


def test_bce_and_iteration_weighting():
    p = torch.tensor([[0.2, 0.8], [0.4, 0.6]], requires_grad=True)
    e = torch.tensor([[0, 1], [1, 0]])
    first, second = bitwise_bce(p, e), bitwise_bce(p*0.9, e)
    weights = torch.tensor([1., 10.])
    expected = (weights*(first+2*second)/3).sum()/weights.sum()
    actual = multi_loss([p, p*0.9], e, bitwise_bce, weights)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(p.grad).all()
