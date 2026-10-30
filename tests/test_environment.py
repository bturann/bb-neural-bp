"""Minimal M0 checks; the full executable API audit is scripts/m0_verify_apis.py."""

import sys

import numpy as np
import stim
from ldpc import BpDecoder, BpOsdDecoder


def test_python_version():
    assert sys.version_info[:2] == (3, 11)


def test_stim_fault_labels():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")
    det, obs, err = dem.compile_sampler(seed=123).sample(shots=100, return_errors=True)
    np.testing.assert_array_equal(det, err)
    np.testing.assert_array_equal(obs, err)


def test_ldpc_syndrome_api():
    h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    for cls in (BpDecoder, BpOsdDecoder):
        decoder = cls(h, error_channel=[0.01]*3, max_iter=15, bp_method="minimum_sum",
                      ms_scaling_factor=0.625, schedule="parallel", omp_thread_count=1)
        correction = decoder.decode(s)
        np.testing.assert_array_equal((h @ correction) % 2, s)
