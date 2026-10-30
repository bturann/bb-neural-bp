"""Instrument a runtime copy of the immutable NumPy reference for per-step checks."""

import inspect

import numpy as np
from ref_impl import minsum_bp


def reference_trace(H, s, llr, T, alpha=0.625, *, msg_clip=None):
    source = inspect.getsource(minsum_bp)
    loop = "    for _ in range(max_iter):"
    stop = "        if conv.all(): break"
    result = "    return e_hat, conv"
    assert source.count(loop) == source.count(stop) == source.count(result) == 1
    source = source.replace(loop, "    history = []\n"+loop)
    source = source.replace(stop, "        history.append(e_hat.copy())")
    source = source.replace(result, "    return history")
    if msg_clip is not None:
        update = "        m_cv = alpha * sign_s * sprod_c[:, ci] * sg * ext"
        assert source.count(update) == 1
        source = source.replace(update, update+"\n        m_cv = np.clip(m_cv, -msg_clip, msg_clip)")
    namespace = {"np": np, "msg_clip": msg_clip}
    exec(compile(source, "<instrumented_reference_copy>", "exec"), namespace)
    return namespace["minsum_bp"](H, s, llr, max_iter=T, alpha=alpha)
