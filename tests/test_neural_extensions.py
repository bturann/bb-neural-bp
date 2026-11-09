import runpy

import numpy as np
import torch

from bbnbp.checkpoints import checkpoint_payload,save_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.noise.code_capacity import CodeCapacitySampler


def test_continuation_preserves_prefix_and_explicit_parameter_schedule(tmp_path):
    sampler=CodeCapacitySampler("bb72",.04)
    base=NeuralMinSum(sampler.problem,3,"orbit",dtype=torch.float64)
    with torch.no_grad():
        base.w[0].fill_(.7);base.w[1].fill_(.8);base.w[2].fill_(.9)
    path=tmp_path/"source.pt"
    save_checkpoint(path,checkpoint_payload(base,load_config("configs/base.yaml"),9))
    extend=runpy.run_path("scripts/m6_neural_extensions.py")["continued_model"]
    s,_,_=sampler.sample(40,np.random.default_rng(581))
    with torch.no_grad():original=base(s)
    for mode,index in (("cycle",[0,1,2,0,1,2]),("last",[0,1,2,2,2,2])):
        model=extend(path,sampler.problem,6,mode)
        torch.testing.assert_close(model.w,base.w[index],rtol=0,atol=0)
        with torch.no_grad():outputs=model(s)
        for a,b in zip(original,outputs[:3]):torch.testing.assert_close(a,b,rtol=0,atol=0)
        expected,conv=model.decode_batch(s)
        actual,actual_conv=model.export_cpu().decode_batch(s)
        np.testing.assert_array_equal(expected,actual)
        np.testing.assert_array_equal(conv,actual_conv)
        assert model.extension_provenance["additional_optimizer_updates"]==0
