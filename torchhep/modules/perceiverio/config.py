from dataclasses import dataclass
from dataclasses import field

from torchhep.utils.config import ConfigBase
import torchhep.optuna.hyperparameter as hp

@dataclass
class IOConfig(ConfigBase):
    dim_input: int
    dim_output: int

@dataclass
class LatentConfig(hp.HyperparameterConfig):
    r"""docstring for LatentConfig TODO
    """
    length: int = hp.integer(default=10, low=5, high=70, step=5)
    dimension: int = hp.integer(default=512, low=64, high=1024, step=64)

@dataclass
class EncoderSelfAttentionConfig(hp.HyperparameterConfig):
    r"""docstring for EncoderSelfAttentionConfig TODO
    """
    num_heads: int = hp.integer(default=8, low=1, high=16)
    widening_factor: int = hp.integer(default=1, low=1, high=3)

@dataclass
class EncoderCrossAttentionConfig(hp.HyperparameterConfig):
    num_heads: int = hp.integer(default=1, low=1, high=8)
    widening_factor: int = hp.integer(default=1, low=1, high=3)
    use_query_residual: bool = hp.boolean(default=True)

@dataclass
class EncoderConfig(hp.HyperparameterConfig):
    self_attn: EncoderSelfAttentionConfig
    cross_attn: EncoderCrossAttentionConfig
    num_blocks: int = hp.integer(default=1, low=1, high=10)
    num_layers_per_block: int = hp.integer(default=2, low=1, high=10)
    dropout: float = hp.uniform(default=0.05, low=0, high=0.2)

@dataclass
class DecoderConfig(hp.HyperparameterConfig):
    use_query_residual: bool = hp.boolean(default=False)
    num_heads: int = hp.integer(default=1, low=1, high=2)

@dataclass
class PerceiverIOConfig(hp.HyperparameterConfig):
    io: IOConfig
    latent: LatentConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
