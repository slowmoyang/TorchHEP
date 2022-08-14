from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

from .modules import PerceiverIOEncoder
from .modules import PerceiverIOSegmentationDecoder
from .modules import PerceiverIOBase
import torchhep.optuna.hyperparameter as hp



class PerceiverIO(PerceiverIOBase):

    def __init__(self, config) -> None:
        self.config = config

        encoder = PerceiverIOEncoder(
            dim_input=config.io.dim_input,
            dim_latent=config.latent_dimension,
            num_layers_per_block=config.encoder_num_layers_per_block,
            num_blocks=config.encoder_num_blocks,
            self_attn_num_heads=config.encoder_self_attn_num_heads,
            self_attn_widening_factor=config.encoder_self_attn_widening_factor,
            cross_attn_num_heads=config.encoder_cross_attn_num_heads,
            cross_attn_widening_factor=config.encoder_cross_attn_widening_factor,
            cross_attn_use_query_residual=config.encoder_cross_attn_use_query_residual,
            dropout=config.encoder_dropout)

        decoder = PerceiverIOSegmentationDecoder(
            dim_output_query=config.io.dim_input,
            dim_latent=config.latent_dimension,
            num_classes=config.io.dim_output)

        super().__init__(encoder,
                         decoder,
                         config.latent_length,
                         config.latent_dimension)

    def forward(self,
                input: Tensor,
                input_data_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return super().forward(input=input,
                               output_query=input,
                               data_mask=data_mask,
                               output_query_data_mask=data_mask)
