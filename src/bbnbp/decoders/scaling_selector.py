"""Auxiliary syndrome-only factor prediction followed by exactly one BP decode."""

from pathlib import Path

import numpy as np

from .base import validate_syndromes
from .bp_ldpc import LdpcBP


def syndrome_features(s, p):
    s = np.asarray(s, dtype=np.float32)
    if s.ndim != 2 or not 0 < p < .5 or not np.isin(s, [0, 1]).all():
        raise ValueError('Require binary syndrome matrix and known noise rate')
    return np.concatenate((s, s.mean(1, keepdims=True), np.full((len(s), 1), p/.003, np.float32)), axis=1)


def choose_factors(scores, fixed, margin):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all() or not 0 <= fixed < scores.shape[1] or not np.isfinite(margin) or margin < 0:
        raise ValueError('Invalid selector scores, fixed factor or switching margin')
    choice = scores.argmax(1)
    gain = scores[np.arange(len(scores)), choice]-scores[:, fixed]
    # Exact ties retain the independently tuned fixed factor.
    return np.where(gain > margin, choice, fixed)


class DensePredictor:
    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as saved:
            self.mean, self.scale = saved['mean'], saved['scale']
            self.weights = [saved[f'w{i}'] for i in range(int(saved['layers']))]
            self.biases = [saved[f'b{i}'] for i in range(len(self.weights))]

    def scores(self, s, p):
        x = (syndrome_features(s, p)-self.mean)/self.scale
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i+1 < len(self.weights):
                x = np.maximum(x, 0)
        # Numerically stable sigmoid, with bounded inputs to avoid overflow.
        return 1/(1+np.exp(-np.clip(x, -80, 80)))


class CountPredictor:
    def __init__(self, path):
        with np.load(Path(path), allow_pickle=False) as saved:
            self.edges, self.rates = saved['edges'], saved['rates']

    def scores(self, s, p):
        counts = np.asarray(s).sum(1)
        return self.rates[np.searchsorted(self.edges, counts, side='right')]


class ScalingSelector:
    name = 'scaling_selector'

    def __init__(self, problem, p, factors, fixed, margin, predictor, iterations=30):
        self.problem, self.p = problem, p
        self.factors = np.asarray(factors, dtype=float)
        self.fixed, self.margin, self.predictor = fixed, margin, predictor
        self.T = iterations
        self.decoders = [LdpcBP(problem, iterations, float(factor)) for factor in self.factors]
        self.last_iterations = np.empty(0, np.int64)
        self.last_choices = np.empty(0, np.int64)

    def choices(self, s):
        s = validate_syndromes(s, self.problem.H.shape[0])
        return choose_factors(self.predictor.scores(s, self.p), self.fixed, self.margin)

    def decode_batch(self, s):
        s = validate_syndromes(s, self.problem.H.shape[0])
        chosen = self.choices(s)
        out = np.empty((len(s), self.problem.H.shape[1]), np.uint8)
        flags = np.empty(len(s), bool)
        self.last_iterations = np.empty(len(s), np.int64)
        self.last_choices = chosen.copy()
        for index in np.unique(chosen):
            mask = chosen == index
            # A shot enters exactly one group, so it receives exactly one BP pass.
            out[mask], flags[mask] = self.decoders[index].decode_batch(s[mask])
            self.last_iterations[mask] = self.decoders[index].last_iterations
        return out, flags
