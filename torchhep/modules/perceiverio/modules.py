r"""NOTE: This implementation uses nn.MultiheadAttention and so has fewer
hyperparameters than [the original](https://github.com/deepmind/deepmind-research/blob/198cf845de758fe966b2b7b1a95797df01e60011/perceiver/perceiver.py)

qk_channels, v_channels, output_channels
"""
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

from ..objwise import Objwise
from ..utils import make_attention_mask


class ObjwisePreNormMLP(Objwise):
    def __init__(self,
                 dim_input: int,
                 widening_factor: int = 4,
                 dropout: float = 0.0,
                 dim_output: Optional[int] = None,
    ) -> None:
        r"""
        """
        dim_hidden = widening_factor * dim_input
        dim_output = dim_output or dim_input

        super().__init__(
            nn.LayerNorm(dim_input),
            nn.Linear(in_features=dim_input, out_features=dim_hidden),
            nn.GELU(),
            nn.Linear(in_features=dim_hidden, out_features=dim_output),
            nn.Dropout(p=dropout))


class SelfAttentionBlock(nn.Module):

    def __init__(self,
                 dim_embed: int,
                 num_heads: float = 8,
                 widening_factor: int = 4,
                 dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attention = Objwise(nn.LayerNorm(dim_embed))

        self.attention = nn.MultiheadAttention(
            embed_dim=dim_embed,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)

        self.dropout_attention = Objwise(nn.Dropout(dropout))

        self.pre_norm_mlp = ObjwisePreNormMLP(
            dim_input=dim_embed,
            widening_factor=widening_factor,
            dropout=dropout)

    def forward(self,
                x: Tensor,
                data_mask: Optional[Tensor] = None,
                pad_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        """
        if pad_mask is None and data_mask is not None:
            pad_mask = data_mask.bitwise_not()

        residual = self.norm_attention(x, data_mask=data_mask)

        residual, _ = self.attention(
            query=residual,
            key=residual,
            value=residual,
            key_padding_mask=pad_mask,
            attn_mask=attention_mask,
            need_weights=False)
        # TODO
        if pad_mask is not None:
            residual = residual.masked_fill(pad_mask.unsqueeze(2), 0)

        residual = self.dropout_attention(residual, data_mask=data_mask)
        x = x + residual

        residual = self.pre_norm_mlp(residual, data_mask=data_mask)
        x = x + residual
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self,
                 dim_embed: int,
                 dim_kv: int,
                 num_heads: int = 8,
                 widening_factor: int = 1,
                 dropout: float = 0.0,
                 use_query_residual: bool = True,
    ) -> None:
        r"""
        """
        super().__init__()
        # attributes
        self._use_query_residual = use_query_residual

        # modules
        self.norm_q = Objwise(nn.LayerNorm(normalized_shape=dim_embed))
        self.norm_kv = Objwise(nn.LayerNorm(normalized_shape=dim_kv))

        self.attention = nn.MultiheadAttention(
            embed_dim=dim_embed,
            num_heads=num_heads,
            dropout=dropout,
            kdim=dim_kv,
            vdim=dim_kv,
            batch_first=True)

        self.dropout_attention = nn.Dropout(p=dropout)
        self.pre_norm_mlp = ObjwisePreNormMLP(
            dim_input=dim_embed,
            widening_factor=widening_factor,
            dropout=dropout)

    def forward(self,
                x_q: Tensor,
                x_kv: Tensor,
                q_mask: Optional[Tensor] = None,
                kv_mask: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x_q: FloatTensor of shape (batch_size, L, query_dim)
            x_kv: FloatTensor of shape (N, S, dim_kv)
            q_mask: BoolTensor of shape (N, L)
            kv_mask: BoolTensor of shape (N, S)
            attention_mask: BoolTensor of shape (N, L, S)
        Returns:
            output: FloatTensor of shape (N, )
        """
        x_kv = self.norm_kv(x_kv, data_mask=kv_mask)

        if kv_mask is None:
            key_padding_mask = None
        else:
            key_padding_mask = kv_mask.bitwise_not()

        x, _ = self.attention(
            query=self.norm_q(x_q, data_mask=q_mask),
            key=x_kv,
            value=x_kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
            need_weights=False)

        # TODO
        if q_mask is not None:
            x = x.masked_fill(q_mask.bitwise_not().unsqueeze(2), 0)

        x = self.dropout_attention(x)

        if self._use_query_residual:
            x = x + x_q

        residual = self.pre_norm_mlp(x, data_mask=q_mask)
        x = x + residual
        return x


class PerceiverIOEncoder(nn.Module):

    def __init__(self,
                 dim_input: int,
                 dim_latent: int,
                 num_blocks: int = 8,
                 num_layers_per_block: int = 6,
                 self_attn_num_heads: int = 8,
                 self_attn_widening_factor: int = 1,
                 cross_attn_num_heads: int = 1,
                 cross_attn_widening_factor: int = 1,
                 cross_attn_use_query_residual: bool = True,
                 dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # attributes
        self.num_blocks = num_blocks
        self.cross_attn_num_heads = cross_attn_num_heads

        # modules
        self.cross_attention = CrossAttentionBlock(
            dim_embed=dim_latent,
            dim_kv=dim_input,
            widening_factor=cross_attn_widening_factor,
            dropout=dropout,
            num_heads=cross_attn_num_heads,
            use_query_residual=cross_attn_use_query_residual)

        self.block = nn.ModuleList()
        for _ in range(num_layers_per_block):
            layer = SelfAttentionBlock(
                dim_embed=dim_latent,
                num_heads=self_attn_num_heads,
                widening_factor=self_attn_widening_factor,
                dropout=dropout)
            self.block.append(layer)

    def forward(self,
                input: Tensor,
                latent: Tensor,
                input_data_mask: Optional[Tensor] = None
    ) -> Tensor:
        if input_data_mask is None:
            attention_mask = None
        else:
            attention_mask = make_attention_mask(
                key_pad_mask=input_data_mask.bitwise_not(),
                num_heads=self.cross_attn_num_heads,
                query_length=latent.size(1))

        latent = self.cross_attention(x_q=latent,
                                      x_kv=input,
                                      q_mask=None,
                                      kv_mask=input_data_mask,
                                      attention_mask=attention_mask)
        for _ in range(self.num_blocks):
            for layer in self.block:
                latent = layer(latent)
        return latent


# FIXME rename ClassificationDecoder
class PerceiverIOSegmentationDecoder(nn.Module):

    def __init__(self,
                 dim_output_query: int,
                 dim_latent: int,
                 num_classes: int,
                 use_query_residual: bool = False,
                 num_heads: int = 1,
    ) -> None:
        super().__init__()

        self.cross_attention = CrossAttentionBlock(
            dim_embed=dim_output_query,
            dim_kv=dim_latent,
            num_heads=num_heads,
            widening_factor=1,
            dropout=0.0,
            use_query_residual=use_query_residual)

        self.segmentation_head = Objwise(
            nn.Linear(in_features=dim_output_query,
                      out_features=num_classes))

    def forward(self,
                query: Tensor,
                latent: Tensor,
                query_mask: Optional[Tensor] = None
    ) -> Tensor:
        output = self.cross_attention(
            x_q=query,
            x_kv=latent,
            q_mask=query_mask,
            kv_mask=None,
            attention_mask=None)
        output = self.segmentation_head(output, data_mask=query_mask)
        return output

class PerceiverIOBase(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 latent_length: int = 6,
                 latent_dimension: int = 32,
                 latent_init_scale: float = 0.02
    ) -> None:
        super().__init__()
        self.latent_init_scale = latent_init_scale

        self.encoder = encoder
        self.decoder = decoder
        self.latent = nn.Parameter(torch.empty(latent_length, latent_dimension))
        self.reset_parameters()


    def reset_parameters(self) -> None:
        r"""
        https://github.com/deepmind/deepmind-research/blob/198cf845de758fe966b2b7b1a95797df01e60011/perceiver/perceiver.py
        z_pos_enc_init_scale=0.02

        https://github.com/deepmind/deepmind-research/blob/2c7c401024c42c4fb1aa20a8b0471d2e6b480906/perceiver/position_encoding.py#L107
        pos_embs = hk.get_parameter(
            'pos_embs',
            [self._index_dim, self._num_channels],
            init=hk.initializers.TruncatedNormal(stddev=self._init_scale)
        )
        """
        torch.nn.init.trunc_normal_(tensor=self.latent, mean=0, std=self.latent_init_scale)

    # TODO
    def forward(self,
                input: Tensor,
                output_query: Optional[Tensor],
                data_mask: Optional[Tensor] = None,
                output_query_data_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Args:
            input: FloatTensor of shape ()
            data_mask: BoolTensor of shape ()
        Returns:
            output: FloatTensor of shape
        """
        latent = self.latent.repeat(input.size(0), 1, 1)
        latent = self.encoder(input=input, latent=latent,
                              input_data_mask=data_mask)
        output = self.decoder(query=output_query, latent=latent,
                              query_mask=output_query_data_mask)
        output = output.squeeze(dim=2)
        return output
