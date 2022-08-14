import torch
from torch import Tensor

# FIXME
# @torch.jit.script
def make_attention_mask(key_pad_mask: Tensor,
                        query_length: int,
                        num_heads: int,
) -> Tensor:
    attn_mask = key_pad_mask.unsqueeze(1).expand(-1, query_length, -1)
    attn_mask = attn_mask.unsqueeze(1)
    attn_mask = attn_mask.repeat(1, num_heads, 1, 1)
    # attn_mask = attn_mask.reshape(key_pad_mask.size(0), query_length, key_pad_mask.size(1))
    attn_mask = attn_mask.reshape(-1, query_length, key_pad_mask.size(1))
    return attn_mask


@torch.jit.script # type: ignore
def make_self_attention_mask(pad_mask: Tensor,
                             num_heads: int,
) -> Tensor:
    r"""docstring for make_self_attention_mask
    """
    attn_mask = pad_mask.unsqueeze(1).expand(-1, pad_mask.size(1), -1)
    attn_mask = attn_mask.unsqueeze(1)
    attn_mask = attn_mask.repeat(1, num_heads, 1, 1)
    attn_mask = attn_mask.reshape(-1, pad_mask.size(1), pad_mask.size(1))
    return attn_mask
