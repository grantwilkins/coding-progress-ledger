from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    name: str
    beta_bytes_per_tok: float
    eta_bytes_per_tok: float
    prefill_tok_s: float
    published_crossover_gbps: float

    @property
    def crossover_gbps(self) -> float:
        return 8.0 * self.prefill_tok_s * (self.eta_bytes_per_tok - self.beta_bytes_per_tok) / 1e9


# TODO: replace with shared model catalog import once the early-experiment package is importable.
MODELS = (
    ModelParams("DeepSeek-V4-Pro", 4.0, 9_900.0, 13_900.0, 1.10),
    ModelParams("GLM-5", 4.0, 89_900.0, 8_300.0, 5.93),
    ModelParams("Qwen3-Next-80B-A3B", 4.0, 24_600.0, 175_000.0, 34.41),
)


def catalog_models() -> tuple[ModelParams, ...]:
    return MODELS


def get_model(name: str) -> ModelParams:
    for model in MODELS:
        if model.name == name:
            return model
    raise KeyError(name)
