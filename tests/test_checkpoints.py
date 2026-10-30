import numpy as np
import pytest
import torch

from bbnbp.checkpoints import checkpoint_payload, load_model, save_checkpoint
from bbnbp.config import load_config
from bbnbp.decoders.neural_bp import NeuralMinSum
from bbnbp.noise.code_capacity import CodeCapacitySampler


def test_restore_changes_priors_without_changing_learned_weights(tmp_path):
    config = load_config("configs/base.yaml")
    train = CodeCapacitySampler("bb72", 0.08)
    evaluate = CodeCapacitySampler("bb72", 0.02)
    model = NeuralMinSum(train.problem, 15, "orbit")
    with torch.no_grad():
        model.w.fill_(0.7)
        model.alpha.fill_(0.9)
    path = tmp_path/"best.pt"
    save_checkpoint(path, checkpoint_payload(model, config, 8))
    restored = load_model(path, evaluate.problem)
    torch.testing.assert_close(model.w, restored.w)
    torch.testing.assert_close(model.alpha, restored.alpha)
    np.testing.assert_array_equal(restored.llr.numpy(), evaluate.problem.llr)
    assert not np.array_equal(restored.llr.numpy(), train.problem.llr)
    with pytest.raises(ValueError, match="topology"):
        load_model(path, CodeCapacitySampler("bb144", 0.02).problem)
