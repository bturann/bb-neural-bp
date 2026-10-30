"""Portable parameter-only model restoration; evaluation priors are never overwritten."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .decoders.neural_bp import NeuralMinSum


def topology_hash(problem):
    digest = hashlib.sha256()
    for matrix in (problem.H, problem.L):
        digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
        digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
        digest.update(matrix.data.tobytes())
    return digest.hexdigest()


def checkpoint_payload(model, config, step, *, training_state=None):
    return dict(format_version=1, step=step, config=config.to_dict(), config_hash=config.config_hash,
                model_config=dict(T=model.T, sharing=model.sharing, ms_scaling=config.model.ms_scaling,
                                  msg_clip=model.msg_clip),
                dtype=str(model.w.dtype), topology_hash=topology_hash(model.problem),
                map_hash=hashlib.sha256(model.w_map.cpu().numpy().tobytes()).hexdigest(),
                parameters={"w": model.w.detach().cpu().clone(), "alpha": model.alpha.detach().cpu().clone()},
                training_state=training_state)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return payload


def restore_parameters(model, payload):
    if topology_hash(model.problem) != payload["topology_hash"]:
        raise ValueError("Checkpoint H/L topology does not match the decoding problem")
    mapping_hash = hashlib.sha256(model.w_map.cpu().numpy().tobytes()).hexdigest()
    if mapping_hash != payload["map_hash"]:
        raise ValueError("Checkpoint weight sharing map does not match the model")
    with torch.no_grad():
        for name in ("w", "alpha"):
            target, source = getattr(model, name), payload["parameters"][name]
            if target.shape != source.shape or not torch.isfinite(source).all():
                raise ValueError(f"Invalid checkpoint parameter {name}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def load_model(path, problem, *, device="cpu"):
    payload = read_checkpoint(path)
    dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
    if payload["dtype"] not in dtypes:
        raise ValueError("Unsupported checkpoint precision")
    model = NeuralMinSum(problem, **payload["model_config"], dtype=dtypes[payload["dtype"]])
    restore_parameters(model, payload)
    # Only learned parameters are restored. LLRs and Tanner buffers come from
    # the supplied evaluation problem, including its current probability p.
    return model.to(device).eval()
