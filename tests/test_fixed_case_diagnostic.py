import runpy

import numpy as np
import pytest

from bbnbp.config import load_config


HELPERS=runpy.run_path("scripts/m6_fixed_case_diagnostic.py")
SETTINGS=dict(training_min_correct_gain=.10,loss_min_relative_reduction=.10,
              unseen_min_correct_gain=.02,unseen_min_fallback_reduction=.02,alpha=.05)


def test_diagnostic_requires_decoding_gain_and_keeps_budget_stops_incomplete():
    initial=dict(logical_failure_rate=1.,mean_loss=100.)
    loss_only=dict(logical_failure_rate=.99,mean_loss=50.)
    classify=HELPERS["classify"]
    assert classify(initial,loss_only,{"useful":False},SETTINGS,True)["decision"]=="loss_reduction_without_useful_training_decoding"
    learned=dict(logical_failure_rate=.5,mean_loss=80.)
    assert classify(initial,learned,{"useful":False},SETTINGS,True)["decision"]=="training_learning_generalization_unconfirmed"
    assert classify(initial,learned,{"useful":True},SETTINGS,True)["decision"]=="learning_and_local_generalization"
    assert classify(initial,learned,{"useful":True},SETTINGS,False)["decision"]=="budget_limited"
    assert classify(initial,initial,{"useful":False},SETTINGS,True)["decision"]=="no_useful_learning_under_this_budget"


def test_syndrome_convergence_with_wrong_logicals_does_not_pass_unseen_screen():
    failed=np.ones(100,dtype=bool)
    initial=dict(failed=failed,logical_failed=failed)
    final=dict(failed=np.zeros(100,dtype=bool),logical_failed=failed)
    result=HELPERS["paired_screen"](initial,final,SETTINGS)
    assert result["fallback_reduction"]==1.
    assert result["correct_gain"]==0.
    assert not result["useful"]
    final["logical_failed"]=final["failed"]
    assert HELPERS["paired_screen"](initial,final,SETTINGS)["useful"]


def test_hard_case_selection_uses_initial_failures_and_unique_syndromes():
    class Sampler:
        def sample(self,n,rng):
            assert n==4
            return np.array([[0,0],[1,0],[1,0],[0,1]],dtype=np.uint8),np.zeros((4,1),dtype=np.uint8),np.zeros((4,2),dtype=np.uint8)
    class Initial:
        def decode_batch(self,s):
            return np.zeros_like(s),np.array([True,False,False,False])
    train,pool,indices,_=HELPERS["hard_training_cases"](Sampler(),Initial(),dict(training_pool_seed=5,
        training_cases=2,training_pool_batch=4,training_pool_max_shots=4))
    np.testing.assert_array_equal(indices,[1,3])
    assert pool["failed"][indices].all()
    assert len(np.unique(train["syndromes"],axis=0))==2


def test_reused_validation_seed_requires_the_explicit_source_directory(tmp_path,monkeypatch):
    c=load_config("configs/m6/cpu_fixed_case.yaml")
    monkeypatch.chdir(tmp_path)
    parent=tmp_path/"results"/"approved"
    parent.mkdir(parents=True)
    (parent/"tuning.csv").write_text("seed\n42234\n")
    settings=dict(training_pool_seed=48234,unseen_seed=42234)
    with pytest.raises(ValueError,match="previously used"):
        HELPERS["seed_guard"](settings,c)
    settings["validation_source"]=str(parent/"cases.npz")
    assert HELPERS["seed_guard"](settings,c)==1
    other=tmp_path/"results"/"not_approved"
    other.mkdir()
    (other/"test.csv").write_text("seed\n42234\n")
    with pytest.raises(ValueError,match="previously used"):
        HELPERS["seed_guard"](settings,c)
