import numpy as np
import torch

from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.losses import ParityLoss
from bbnbp.noise.base import DecodingProblem, parity
from bbnbp.noise.code_capacity import CodeCapacitySampler


def test_training_readout_matches_first_convergence_probabilities_and_export():
    sampler=CodeCapacitySampler("bb72",0.06)
    s,_,_=sampler.sample(100,np.random.default_rng(191))
    s[0]=0
    raw=NeuralMinSum(sampler.problem,15,"orbit",dtype=torch.float64)
    frozen=NeuralMinSum(sampler.problem,15,"orbit",dtype=torch.float64)
    with torch.no_grad():
        outputs=raw(s)
    first=np.full(len(s),16)
    first[~s.any(1)]=0
    for t,hard in enumerate(raw.last_hard_decisions,1):
        valid=(parity(sampler.problem.H,hard.numpy().astype(np.uint8))==s).all(1)
        first[(first==16)&valid]=t
    readout=frozen(s,readout="first_converged")
    for t,actual in enumerate(readout,1):
        expected=outputs[t-1].clone()
        expected[first==0]=torch.sigmoid(-raw.llr)
        for step in range(1,t+1):
            expected[first==step]=outputs[step-1][first==step]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    correction,conv=frozen.export_cpu().decode_batch(s)
    np.testing.assert_array_equal(frozen.last_e_hat.numpy(),correction)
    np.testing.assert_array_equal(frozen.last_converged.numpy(),conv)
    np.testing.assert_array_equal(frozen.last_iterations,np.where(first<=15,first,15))
    sum(p.mean() for p in readout).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in frozen.parameters())


def test_frozen_wrong_logical_readout_keeps_penalty_and_gradient():
    problem=DecodingProblem("logical-toy",[[1,1]],[[1,0]],[0.1,0.2],{})
    model=NeuralMinSum(problem,4,"scalar",0.95,dtype=torch.float64)
    outputs=model(np.array([[1]],dtype=np.uint8),readout="first_converged")
    assert model.last_iterations[0]==1
    np.testing.assert_array_equal(model.last_e_hat.numpy(),[[0,1]])
    for p in outputs[1:]:
        torch.testing.assert_close(p,outputs[0],rtol=0,atol=0)
    objective=ParityLoss(problem).double()
    wrong=objective(outputs[-1],torch.tensor([[1,0]]))
    correct=objective(outputs[-1],torch.tensor([[0,1]]))
    assert wrong.item()>correct.item()+1
    wrong.sum().backward()
    assert model.w.grad[0].abs().sum()>0
    assert model.w.grad[1:].abs().sum()==0


def test_zero_syndrome_training_readout_has_safe_zero_gradients():
    problem=DecodingProblem("zero-toy",[[1,1]],[[1,0]],[0.1,0.2],{})
    model=NeuralMinSum(problem,3,"scalar",dtype=torch.float64)
    outputs=model(np.zeros((2,1),dtype=np.uint8),readout="first_converged")
    for output in outputs:
        torch.testing.assert_close(output,torch.sigmoid(-model.llr).expand(2,2),rtol=0,atol=0)
    sum(output.sum() for output in outputs).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()==0
