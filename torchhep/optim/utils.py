from dataclasses import dataclass
import torch
import torch.optim as optim
from torch._six import inf
import torchhep.optuna.hyperparameter as hp

DEFAULT_WHITELIST = (
    torch.nn.Linear,
    torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
    torch.nn.MultiheadAttention,
)

DEFAULT_BLACKLIST = (
    torch.nn.LayerNorm,
    torch.nn.Embedding,
)

@dataclass
class OptimizerConfig(hp.HyperparameterConfig):
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1

def configure_optimizer(
        model,
        config,
        whitelist=DEFAULT_WHITELIST,
        blacklist=DEFAULT_BLACKLIST
):
    """
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134-L178
    """

    decay = set()
    no_decay = set()

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():
             # full param name
            full_param_name = f'{module_name}.{param_name}' if module_name else param_name

            if param_name.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(full_param_name)
            elif param_name.endswith('weight'):
                if isinstance(module, whitelist):
                    # weights of whitelist modules will be weight decayed
                    decay.add(full_param_name)
                elif isinstance(module, blacklist):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(full_param_name)
            else:
                # FIXME for latents of PerceiverIO
                no_decay.add(full_param_name)

    # validate that we considered every parameter
    param_dict = dict(model.named_parameters())
    inter_params = decay & no_decay
    union_params = decay | no_decay

    if len(inter_params) != 0:
        raise RuntimeError(f"parameters {inter_params} made it into both decay/"
                           "no_decay sets!")

    if len(param_dict.keys() - union_params) != 0:
        raise RuntimeError(f"parameters {param_dict.keys() - union_params} were"
                            "not separated into either decay/no_decay set!")

    params = [
        {
            "params": [param_dict[each] for each in sorted(list(decay))],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [param_dict[each] for each in sorted(list(no_decay))],
            "weight_decay": 0.0
        },
    ]

    optimizer = optim.AdamW(
        params=params,
        lr=config.learning_rate,
        betas=config.betas
    )
    return optimizer
