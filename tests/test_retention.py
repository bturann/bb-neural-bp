import runpy

import numpy as np
import torch

from bbnbp.losses import multi_loss

HELPERS=runpy.run_path("scripts/m6_retention.py")


def test_hard_control_excludes_easy_gradients_and_retention_includes_them():
    a=torch.tensor([[.2],[.4],[.6],[.8]],requires_grad=True)
    e=torch.zeros_like(a)
    objective=lambda p,e:(p-e).square().sum(1)
    weights=torch.tensor([1.,1.,0.,0.])
    loss=HELPERS["weighted_objective"]([a],e,objective,weights)
    torch.testing.assert_close(loss,multi_loss([a[:2]],e[:2],objective))
    loss.backward()
    assert a.grad[:2].abs().sum()>0 and a.grad[2:].abs().sum()==0
    b=a.detach().clone().requires_grad_()
    HELPERS["weighted_objective"]([b],e,objective,torch.ones(4)).backward()
    assert b.grad[2:].abs().sum()>0


def test_shared_batch_plan_balances_each_group_and_is_seeded():
    settings=dict(shuffle_seed_entropy=[1234,913],hard_cases=8,easy_cases=8,hard_per_batch=2,easy_per_batch=2)
    plan=HELPERS["batch_plan"](settings,8)
    np.testing.assert_array_equal(plan,HELPERS["batch_plan"](settings,8))
    assert (plan[:,:2]<8).all() and (plan[:,2:]>=8).all()
    for epoch in (plan[:4],plan[4:]):
        np.testing.assert_array_equal(np.sort(epoch[:,:2].ravel()),np.arange(8))
        np.testing.assert_array_equal(np.sort(epoch[:,2:].ravel()),np.arange(8,16))
