from typing import Protocol

import numpy as np


class Decoder(Protocol):
    name: str

    def decode_batch(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return uint8 corrections [B,N] and boolean syndrome-match flags [B]."""


def validate_syndromes(s, checks):
    s = np.asarray(s)
    if s.ndim != 2 or s.shape[1] != checks or not np.isin(s, [0, 1]).all():
        raise ValueError(f"Expected binary syndromes with shape [B, {checks}]")
    return np.ascontiguousarray(s, dtype=np.uint8)
