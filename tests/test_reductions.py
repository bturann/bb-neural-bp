import numpy as np
import torch

from bbnbp.reductions import DeterministicGather, DeterministicSegmentSum, inverse_slots


def test_gather_gradients_with_repeated_indices_and_padding():
    indices = torch.tensor([0, 1, 0, 2, 0])
    active = torch.tensor([True, True, True, True, False])
    inverse = inverse_slots(indices.numpy(), 3, active.numpy())
    a = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    custom = DeterministicGather.apply(a, indices, inverse)*active
    reference = b[:, indices]*active
    scale = torch.arange(1, 6, dtype=torch.float64)
    (custom*scale).sum().backward(); (reference*scale).sum().backward()
    torch.testing.assert_close(custom, reference, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_segment_sum_gradients_match_index_add():
    indices = torch.tensor([0, 1, 0, 2, 0])
    active = torch.tensor([True, True, True, True, False])
    inverse = inverse_slots(indices.numpy(), 3, active.numpy())
    generator = torch.Generator().manual_seed(818)
    a = torch.randn((2, 5), dtype=torch.float64, generator=generator, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    custom = DeterministicSegmentSum.apply(a, inverse, indices, active)
    reference = b.new_zeros((2, 3)).index_add(1, indices, b*active)
    (custom.square()).sum().backward(); (reference.square()).sum().backward()
    torch.testing.assert_close(custom, reference, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-14, atol=1e-14)
