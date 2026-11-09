import numpy as np
import torch

from bbnbp.loss_sine import SineParityLoss
from bbnbp.losses import ParityLoss
from bbnbp.noise.base import DecodingProblem


def test_sine_surrogate_accepts_degenerate_corrections_and_penalizes_logicals():
    problem=DecodingProblem("toy",[[1,1,0,0]],[[0,0,1,1]],[.1]*4,{})
    loss=SineParityLoss(problem,5.).double()
    faults=torch.tensor([[1.,0.,1.,0.]],dtype=torch.float64)
    assert loss(torch.tensor([[0.,1.,0.,1.]],dtype=torch.float64),faults).item()<1e-12
    assert loss(torch.tensor([[0.,0.,1.,0.]],dtype=torch.float64),faults).item()>.99
    assert loss(torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64),faults).item()>4.99


def test_sine_surrogate_gradcheck_and_high_degree_probability_gradient():
    problem=DecodingProblem("toy",[[1,1,0,0]],[[0,0,1,1]],[.1]*4,{})
    p=torch.tensor([[.13,.24,.37,.42]],dtype=torch.float64,requires_grad=True)
    e=torch.tensor([[1,0,0,1]])
    assert torch.autograd.gradcheck(lambda x:SineParityLoss(problem).double()(x,e).sum(),(p,))
    dense=DecodingProblem("high-degree",np.ones((1,200),dtype=np.uint8),np.zeros((1,200),dtype=np.uint8),[.1]*200,{})
    p=torch.full((1,200),.251,dtype=torch.float64,requires_grad=True)
    e=torch.zeros_like(p)
    sine,=torch.autograd.grad(SineParityLoss(dense).double()(p,e).sum(),p)
    product,=torch.autograd.grad(ParityLoss(dense).double()(p,e).sum(),p)
    assert sine.norm()>1 and product.norm()<1e-20
