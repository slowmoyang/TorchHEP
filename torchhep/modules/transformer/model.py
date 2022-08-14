from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn

from .modules import TransformerEncoder
from ..objwise import Objwise

class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.input_projection = Objwise(
            nn.Linear(config.io.dim_input, config.encoder.dim_model),
            nn.GELU())

        self.encoder = TransformerEncoder(config.encoder)

        self.output_projection = Objwise(
            nn.Linear(config.encoder.dim_model, config.io.dim_output))

    def forward(self,
                input: Tensor,
                data_mask: Tensor
    ) -> Tensor:
        # CMSSW/ONNXRuntime
        if data_mask.is_floating_point():
            data_mask = data_mask.not_equal(0)
        x = self.input_projection(input, data_mask)
        x = self.encoder(x, data_mask)
        x = self.output_projection(x, data_mask)
        if self.dim_output == 1:
            x = x.squeeze(dim=2)
        return x
