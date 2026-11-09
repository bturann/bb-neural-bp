from dataclasses import replace

import numpy as np
import pytest

from bbnbp.decoders.cascade import CascadeDecoder
from bbnbp.noise.base import DecodingProblem


class ToyDecoder:
    name="toy"
    T=3
    def __init__(self,problem,corrections,converged,iterations):
        self.problem=problem
        self.corrections=np.asarray(corrections,dtype=np.uint8)
        self.converged=np.asarray(converged,dtype=bool)
        self.iterations=np.asarray(iterations)
        self.calls=[]
    def decode_batch(self,s):
        self.calls.append(s.copy())
        self.last_iterations=self.iterations.copy()
        return self.corrections.copy(),self.converged.copy()


def test_cascade_only_routes_failures_and_preserves_first_wrong_logical():
    problem=DecodingProblem("toy",[[1,1]],[[1,0]],[.1,.2],{})
    first=ToyDecoder(problem,[[0,1],[0,0]],[True,False],[1,3])
    second=ToyDecoder(problem,[[1,0]],[True],[2])
    decoder=CascadeDecoder(first,second)
    out,conv=decoder.decode_batch(np.array([[1],[1]],dtype=np.uint8))
    np.testing.assert_array_equal(out,[[0,1],[1,0]])
    np.testing.assert_array_equal(conv,[True,True])
    np.testing.assert_array_equal(decoder.last_iterations,[1,5])
    np.testing.assert_array_equal(decoder.last_used_second,[False,True])
    assert second.calls[0].shape==(1,1)


def test_cascade_skips_second_pass_and_rejects_wrong_priors_or_flags():
    problem=DecodingProblem("toy",[[1,1]],[[1,0]],[.1,.2],{})
    first=ToyDecoder(problem,[[0,1]],[True],[1])
    second=ToyDecoder(problem,[],[],[])
    decoder=CascadeDecoder(first,second)
    decoder.decode_batch(np.array([[1]],dtype=np.uint8))
    assert not second.calls
    with pytest.raises(ValueError,match="priors"):
        CascadeDecoder(first,ToyDecoder(replace(problem,priors=np.array([.2,.2])),[],[],[]))
    first.converged[:]=False
    with pytest.raises(AssertionError,match="incorrect convergence"):
        decoder.decode_batch(np.array([[1]],dtype=np.uint8))
