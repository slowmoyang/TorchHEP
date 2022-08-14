from dataclasses import dataclass

import torchhep.optuna.hyperparameter as hp
from torchhep.utils.config import ConfigBase

@dataclass
class TransformerEncoderConfig(hp.HyperparameterConfig):
    dim_head: int = hp.integer(default=16, low=4, high=64, step=4)
    num_heads: int = hp.integer(default=2, low=1, high=8)
    dim_feedforward: int = hp.integer(default=128, low=32, high=1024, step=32)
    num_layers: int = hp.integer(default=2, low=1, high=12)
    dropout: float = hp.uniform(default=0.05, low=0.0, high=0.95)

    @property
    def dim_model(self) -> int:
        return self.dim_head * self.num_heads

@dataclass
class TransformerIOConfig(ConfigBase):
    dim_input: int
    dim_output: int

@dataclass
class TransformerConfig(hp.HyperparameterConfig):
    io: TransformerIOConfig
    encoder: TransformerEncoderConfig
