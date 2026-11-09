"""Frozen builders for auxiliary same-budget decoder comparisons."""

import runpy
from pathlib import Path

import torch

from bbnbp.checkpoints import load_model,read_checkpoint
from bbnbp.decoders.bp_ldpc import LdpcBP
from bbnbp.decoders.cascade import CascadeDecoder
from bbnbp.decoders.neural_bp import NeuralMinSum


def neural_model(checkpoint,problem,design,initial=False):
    payload=read_checkpoint(checkpoint)
    kwargs=dict(payload["model_config"])
    if initial:
        kwargs["T"]=15 if design=="cascade15" else 30
        return NeuralMinSum(problem,**kwargs,dtype={"torch.float32":torch.float32,"torch.float64":torch.float64}[payload["dtype"]]).eval()
    if design=="cascade15":return load_model(checkpoint,problem)
    local=Path(__file__).with_name("neural_extensions_source.py")
    helper=runpy.run_path(str(local if local.exists() else Path("scripts/m6_neural_extensions.py")))
    return helper["continued_model"](checkpoint,problem,30,{"cycle30":"cycle","last30":"last"}[design])


def make_decoders(problem,checkpoint,design,controls):
    cpp,native=controls["cpp"],controls["native"]
    first_factor=cpp["first_factor"]
    decoder=lambda T,f:LdpcBP(problem,T,f)
    native_decoder=lambda T,f:NeuralMinSum(problem,T,"scalar",f,20.,dtype=torch.float32).export_cpu()
    result=dict(
        bp15=decoder(15,first_factor),
        bp30=decoder(30,cpp["selected"]["longer"]["scaling"]),
        two_bp=CascadeDecoder(decoder(15,first_factor),decoder(15,cpp["selected"]["two_pass"]["scaling"])),
        native30=native_decoder(30,native["selected"]["native30"]["scaling"]),
        native_second=CascadeDecoder(decoder(15,first_factor),native_decoder(15,native["selected"]["native_second"]["scaling"])))
    for name,initial in (("selected",False),("matched_init",True)):
        model=neural_model(checkpoint,problem,design,initial=initial)
        node=model.export_cpu()
        result[name]=CascadeDecoder(decoder(15,first_factor),node) if design=="cascade15" else node
    result["ordinary"]=result[controls["best_ordinary"]["name"]]
    return result
