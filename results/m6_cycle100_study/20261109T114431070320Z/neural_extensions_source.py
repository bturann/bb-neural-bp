"""Explicit, parameter-only continuation of a trained unroll for E5-style validation."""

import hashlib
from pathlib import Path

import torch

from bbnbp.checkpoints import load_model,read_checkpoint
from bbnbp.decoders.neural_bp import NeuralMinSum


def continued_model(path,problem,iterations,extension):
    payload=read_checkpoint(path)
    original=load_model(path,problem)
    if type(iterations) is not int or iterations<original.T or extension not in {"cycle","last"}:
        raise ValueError("Continuation requires a valid larger unroll and cycle/last rule")
    kwargs=dict(payload["model_config"],T=iterations)
    model=NeuralMinSum(problem,**kwargs,dtype=original.w.dtype)
    indices=torch.arange(iterations)
    indices=indices%original.T if extension=="cycle" else indices.clamp_max(original.T-1)
    with torch.no_grad():
        model.w.copy_(original.w[indices])
        model.alpha.copy_(original.alpha[indices])
    model.extension_provenance=dict(source_checkpoint=str(path),source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        source_iterations=original.T,inference_iterations=iterations,extension=extension,additional_optimizer_updates=0)
    return model.eval()
