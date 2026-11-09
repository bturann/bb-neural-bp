from dataclasses import replace

import numpy as np
import torch

from bbnbp.config import load_config
from bbnbp.losses import ParityLoss,multi_loss
from bbnbp.noise.factory import make_sampler
from bbnbp.train import make_model


def test_capacity_factorial_changes_only_sharing_and_scaled_learning_rate():
    reference=load_config("configs/m6/cpu_capacity_struct_standard.yaml")
    for arm in ("struct_standard","struct_low","full_standard","full_low"):
        c=load_config(f"configs/m6/cpu_capacity_{arm}.yaml")
        assert c.train.min_lr==c.train.lr*.01
        normalized=replace(c,model=replace(c.model,sharing=reference.model.sharing),
            train=replace(c.train,lr=reference.train.lr,min_lr=reference.train.min_lr))
        assert normalized==reference


def test_structural_and_full_start_identically_and_both_have_finite_updates():
    reference=load_config("configs/m6/cpu_capacity_struct_standard.yaml")
    sampler=make_sampler(reference,.001)
    s,_,e=sampler.sample(16,np.random.default_rng(69911))
    decisions=[]
    for arm in ("struct_standard","struct_low","full_standard","full_low"):
        c=load_config(f"configs/m6/cpu_capacity_{arm}.yaml")
        model=make_model(sampler.problem,c)
        decisions.append(model.export_cpu().decode_batch(s))
        objective=ParityLoss(sampler.problem,c.train.beta_logical)
        optimizer=torch.optim.Adam(model.parameters(),lr=c.train.lr)
        loss=multi_loss(model(s[:4]),torch.from_numpy(e[:4]),objective)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        assert torch.isfinite(norm) and norm>0
        optimizer.step()
        assert any((p.detach()!=value).any() for p,value in ((model.w,.95),(model.alpha,1.)))
        assert all(torch.isfinite(p).all() for p in model.parameters())
    for correction,converged in decisions[1:]:
        np.testing.assert_array_equal(correction,decisions[0][0])
        np.testing.assert_array_equal(converged,decisions[0][1])
