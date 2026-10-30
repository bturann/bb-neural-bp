"""Strict, immutable experiment configuration with canonical content hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CodeConfig:
    name: str


@dataclass(frozen=True)
class CircuitConfig:
    strategy: str
    idle_noise: bool
    get_all_detectors: bool
    noisy_zeroth_round: bool
    noisy_final_meas: bool
    decompose_errors: bool
    total_rounds_include_zeroth: bool


CIRCUIT_POLICY = CircuitConfig(
    strategy="custom", idle_noise=True, get_all_detectors=True,
    noisy_zeroth_round=True, noisy_final_meas=True, decompose_errors=False,
    total_rounds_include_zeroth=True,
)


@dataclass(frozen=True)
class NoiseConfig:
    model: str
    rounds: int | None
    basis: str
    p_grid: tuple[float, ...]
    circuit: CircuitConfig | None = None


@dataclass(frozen=True)
class ModelConfig:
    T: int
    sharing: str
    ms_scaling: float
    msg_clip: float


@dataclass(frozen=True)
class CurriculumConfig:
    p_start: float
    p_end: float
    steps: int


@dataclass(frozen=True)
class TrainingRuntimeConfig:
    threads: int
    val_batch: int
    validation_seed_offset: int
    training_stream: int
    torch_seed_offset: int
    log_every: int
    loss_eps: float
    loss_floor: float
    gate_min_points: int
    tensorboard: bool
    max_gpu_hours: float
    baseline_table: str
    purpose: str
    m3_gate_report: str


@dataclass(frozen=True)
class TrainConfig:
    batch: int
    max_steps: int
    lr: float
    min_lr: float
    grad_clip: float
    loss: str
    beta_logical: float
    hard_mining: str
    hard_weight: float
    mining_max_draws: int
    curriculum: CurriculumConfig
    val_every: int
    val_shots: int
    patience: int
    device: str
    runtime: TrainingRuntimeConfig | None = None


@dataclass(frozen=True)
class OsdConfig:
    method: str
    order: int
    bp_iters: int
    ms_scaling: float


@dataclass(frozen=True)
class EvalConfig:
    stop_failures: int
    max_shots: int
    batch: int
    stage1_iters: int
    osd: OsdConfig
    latency_shots: int
    latency_warmup: int
    threads: int
    seed_offset: int
    purpose: str
    sample_offset: int = 0


@dataclass(frozen=True)
class SweepConfig:
    factors: tuple[float, ...]
    iterations: tuple[int, ...]
    shots: int
    seed_offset: int


@dataclass(frozen=True)
class Config:
    seed: int
    code: CodeConfig
    noise: NoiseConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    sweep: SweepConfig

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha1(canonical.encode()).hexdigest()

    def to_dict(self):
        raw = asdict(self)
        # Keep historical code-capacity config hashes stable as circuit options are added.
        if self.noise.circuit is None:
            del raw["noise"]["circuit"]
        if self.eval.sample_offset == 0:
            del raw["eval"]["sample_offset"]
        if self.train.runtime is None:
            del raw["train"]["runtime"]
        return raw

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # JSON roundtrip converts tuples to safe YAML lists.
        path.write_text(yaml.safe_dump(json.loads(json.dumps(self.to_dict())), sort_keys=True))

    def validate(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.code.name not in {"bb72", "bb144"}:
            raise ValueError("code.name must be bb72 or bb144")
        if self.noise.model not in {"code_capacity", "phenomenological", "circuit_level"}:
            raise ValueError("Unknown noise model")
        if self.noise.basis not in {"X", "Z"}:
            raise ValueError("noise.basis must be X or Z")
        if self.noise.model == "code_capacity":
            if self.noise.rounds is not None:
                raise ValueError("Code capacity has no rounds")
        elif type(self.noise.rounds) is not int or self.noise.rounds < 1:
            raise ValueError("Noisy measurement models require positive integer rounds")
        if self.noise.model == "circuit_level":
            if self.noise.circuit != CIRCUIT_POLICY:
                raise ValueError("Circuit noise options must explicitly match the supported uniform-noise policy")
            if any(type(v) is not bool for k, v in asdict(self.noise.circuit).items() if k != "strategy"):
                raise ValueError("Circuit policy toggles must be booleans")
        elif self.noise.circuit is not None:
            raise ValueError("Circuit options require circuit_level noise")
        if not self.noise.p_grid or any(not 0 < p < 0.5 for p in self.noise.p_grid):
            raise ValueError("p_grid must be nonempty with probabilities in (0, 0.5)")
        if len(set(self.noise.p_grid)) != len(self.noise.p_grid):
            raise ValueError("p_grid contains duplicate points")
        for name, value in {
            "model.T": self.model.T, "train.batch": self.train.batch,
            "train.max_steps": self.train.max_steps, "train.val_every": self.train.val_every,
            "train.val_shots": self.train.val_shots, "train.patience": self.train.patience,
            "eval.stop_failures": self.eval.stop_failures, "eval.max_shots": self.eval.max_shots,
            "eval.batch": self.eval.batch, "eval.stage1_iters": self.eval.stage1_iters,
            "osd.bp_iters": self.eval.osd.bp_iters, "sweep.shots": self.sweep.shots,
            "train.mining_max_draws": self.train.mining_max_draws,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.model.sharing not in {"scalar", "orbit", "struct", "full"}:
            raise ValueError("Unknown weight sharing")
        if self.train.loss not in {"parity", "bitwise"}:
            raise ValueError("Unknown training loss")
        if self.train.hard_mining not in {"off", "filter", "upweight"}:
            raise ValueError('hard_mining must be a string; quote "off" in YAML')
        if self.eval.osd.method not in {"osd0", "osd_cs"} or self.eval.osd.order < 0:
            raise ValueError("Invalid OSD settings")
        if type(self.eval.osd.order) is not int:
            raise ValueError("OSD order must be an integer")
        if self.eval.osd.method == "osd0" and self.eval.osd.order != 0:
            raise ValueError("osd0 requires order 0")
        if self.eval.threads != 1:
            raise ValueError("Primary CPU evaluation requires a single thread")
        if self.eval.purpose not in {"development", "final"}:
            raise ValueError("eval.purpose must be development or final")
        if not self.sweep.factors or any(not 0 < x <= 1 for x in self.sweep.factors):
            raise ValueError("Invalid scaling sweep")
        if not self.sweep.iterations or any(type(t) is not int or t < 1 for t in self.sweep.iterations):
            raise ValueError("Invalid sweep iteration budgets")
        for name, value in {"msg_clip": self.model.msg_clip, "ms_scaling": self.model.ms_scaling,
                            "osd.ms_scaling": self.eval.osd.ms_scaling, "lr": self.train.lr,
                            "min_lr": self.train.min_lr, "grad_clip": self.train.grad_clip,
                            "hard_weight": self.train.hard_weight}.items():
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if self.train.beta_logical < 0 or self.train.curriculum.steps < 0:
            raise ValueError("Loss weight and curriculum steps must be nonnegative")
        if any(not 0 < p < 0.5 for p in (self.train.curriculum.p_start, self.train.curriculum.p_end)):
            raise ValueError("Invalid curriculum probability")
        if not 0 < self.model.ms_scaling <= 1 or not 0 < self.eval.osd.ms_scaling <= 1:
            raise ValueError("Min-sum scaling must be in (0, 1]")
        if self.train.min_lr > self.train.lr:
            raise ValueError("Minimum learning rate exceeds the initial learning rate")
        if any(type(x) is not int or x < 0 for x in (self.eval.latency_shots, self.eval.latency_warmup)):
            raise ValueError("Latency counts must be nonnegative integers")
        if any(type(x) is not int or x < 0 for x in (self.eval.seed_offset, self.sweep.seed_offset)):
            raise ValueError("Seed offsets must be nonnegative integers")
        if abs(self.eval.seed_offset-self.sweep.seed_offset) < len(self.noise.p_grid):
            raise ValueError("Sweep and evaluation seed offsets must differ")
        if type(self.eval.sample_offset) is not int or self.eval.sample_offset < 0:
            raise ValueError("Sample offset must be a nonnegative integer")
        if self.eval.sample_offset % self.eval.batch:
            raise ValueError("Sample offset must be a multiple of the sampling batch")
        runtime = self.train.runtime
        if runtime is not None:
            if runtime.purpose not in {"smoke", "m5_gate"}:
                raise ValueError("Training purpose must be smoke or m5_gate")
            for field in ("threads", "val_batch", "log_every", "gate_min_points"):
                if type(getattr(runtime, field)) is not int or getattr(runtime, field) < 1:
                    raise ValueError(f"train.runtime.{field} must be a positive integer")
            for field in ("validation_seed_offset", "training_stream", "torch_seed_offset"):
                if type(getattr(runtime, field)) is not int or getattr(runtime, field) < 0:
                    raise ValueError(f"train.runtime.{field} must be a nonnegative integer")
            if not 0 < runtime.loss_eps < 0.5 or not 0 < runtime.loss_floor < 0.5:
                raise ValueError("Training loss clamps must be in (0, 0.5)")
            if runtime.gate_min_points > len(self.noise.p_grid):
                raise ValueError("Training gate requires more points than the configured grid")
            if not 0 < runtime.max_gpu_hours <= 4:
                raise ValueError("Training runs are capped at four GPU-hours; larger runs require human agreement")
            if type(runtime.tensorboard) is not bool:
                raise ValueError("TensorBoard toggle must be boolean")
            val_seeds = set(range(self.seed+runtime.validation_seed_offset,
                                  self.seed+runtime.validation_seed_offset+len(self.noise.p_grid)))
            for offset in (self.eval.seed_offset, self.sweep.seed_offset, 10000):
                if val_seeds & set(range(self.seed+offset, self.seed+offset+len(self.noise.p_grid))):
                    raise ValueError("Training validation seeds overlap tuning/evaluation/final seeds")
            if type(self.train.curriculum.steps) is not int or self.train.curriculum.steps < 0:
                raise ValueError("Curriculum steps must be a nonnegative integer")
            if runtime.purpose == "m5_gate":
                expected_grid = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
                if (self.code.name != "bb72" or self.noise.model != "code_capacity"
                        or self.model.sharing != "orbit" or self.noise.p_grid != expected_grid
                        or runtime.gate_min_points != 4 or self.train.val_shots < 20000):
                    raise ValueError("M5 acceptance requires bb72 code capacity, orbit sharing, seven p values, "
                                     "20k validation shots per p and wins at four points")
        # Reject NaN and non-serializable values before running an experiment.
        _ = self.config_hash
        return self


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")
    raw = dict(raw)
    try:
        noise = dict(raw["noise"])
        noise["p_grid"] = tuple(noise["p_grid"])
        if "circuit" in noise:
            noise["circuit"] = CircuitConfig(**noise["circuit"])
        train = dict(raw["train"])
        train["curriculum"] = CurriculumConfig(**train["curriculum"])
        if "runtime" in train:
            train["runtime"] = TrainingRuntimeConfig(**train["runtime"])
        ev = dict(raw["eval"])
        ev["osd"] = OsdConfig(**ev["osd"])
        sweep = dict(raw["sweep"])
        sweep["factors"] = tuple(sweep["factors"])
        sweep["iterations"] = tuple(sweep["iterations"])
        raw.update(code=CodeConfig(**raw["code"]), noise=NoiseConfig(**noise),
                   model=ModelConfig(**raw["model"]), train=TrainConfig(**train),
                   eval=EvalConfig(**ev), sweep=SweepConfig(**sweep))
        return Config(**raw).validate()
    except (TypeError, KeyError) as exc:
        raise ValueError(f"Invalid or missing config field: {exc}") from exc
