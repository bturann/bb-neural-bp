from .code_capacity import CodeCapacitySampler


def make_sampler(config, p):
    if config.noise.model == "code_capacity":
        return CodeCapacitySampler(config.code.name, p, config.noise.basis)
    if config.noise.model == "circuit_level":
        from .circuit_level import CircuitLevelSampler
        return CircuitLevelSampler(config.code.name, config.noise.rounds, p, config.noise.basis)
    raise ValueError(f"Noise model {config.noise.model!r} is not implemented")
