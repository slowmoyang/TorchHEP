from typing import Optional
from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones

from ..objwise import Objwise
from ..utils import make_self_attention_mask


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 dim_model: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 batch_first: bool = True,
    ) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first)

        self.dropout = Objwise(nn.Dropout(dropout))

        # pre-norm ffnn
        # merge layer norm, FFNN and dropout into a single objwise
        self.residual_function = Objwise(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_model),
            nn.Dropout(dropout)
        )
        self.norm = Objwise(nn.LayerNorm(dim_model))

    def forward(self,
                x: Tensor,
                data_mask: Tensor,
                pad_mask: Tensor,
                attn_mask: Tensor,
                # data_mask: Optional[Tensor] = None,
                # pad_mask: Optional[Tensor] = None,
                # attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        residual, _ = self.attention(query=x, key=x, value=x,
                                     key_padding_mask=pad_mask,
                                     attn_mask=attn_mask,
                                     need_weights=False)

        residual = self.dropout(residual, data_mask)
        x = x + residual

        residual = self.residual_function(x, data_mask)

        x = x + residual
        x = self.norm(x, data_mask)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        layer = TransformerEncoderLayer(
            config.dim_model,
            config.num_heads,
            config.dim_feedforward,
            config.dropout)

        self._layers = _get_clones(layer, config.num_layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for each in self.parameters():
            if each.dim() > 1:
                nn.init.xavier_uniform_(each)

    def forward(self,
                x: Tensor,
                data_mask: Tensor,
    ) -> Tensor:
        # FIXME
        pad_mask = data_mask.bitwise_not()
        attn_mask = make_self_attention_mask(
            pad_mask=pad_mask,
            num_heads=self.config.num_heads)
        # if data_mask is not None:
            # NOTE onnx with opset v11 does not support logical_not
            # key_padding_mask = mask.logical_not()

        for layer in self._layers:
            x = layer(x, data_mask, pad_mask, attn_mask)
        return x
