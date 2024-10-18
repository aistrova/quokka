# Built on top of https://github.com/kyegomez/zeta

# Additional features in Quokka:
# - Hugging Face transformers v4 config.json compatibility
# - SwiGLU activation
# - clearer Grouped Query Attention
# - text generation with RoPE
# - (optional) Attention with linear complexity
#    - kernel feature map approximation
#    - memory-efficient softmax operations
# - (optional) Zigzag Ring Attention with Flash Attention

from functools import partial, wraps
from dataclasses import dataclass
from inspect import isfunction
from random import random

import torch
import torch.nn.functional as F
from collections import namedtuple
from einops import rearrange, repeat
from packaging import version
from torch import Tensor, einsum, nn
from typing import Optional, List
import math

DEFAULT_DIM_HEAD = 64

EfficientAttentionConfig = namedtuple(
    "EfficientAttentionConfig",
    ["enable_flash", "enable_math", "enable_mem_efficient"],
)


@dataclass
class Intermediates:
    qk_similarities: Optional[Tensor] = None
    pre_softmax_attn: Optional[Tensor] = None
    post_softmax_attn: Optional[Tensor] = None

    def to_tuple(self):
        return (
            self.qk_similarities,
            self.pre_softmax_attn,
            self.post_softmax_attn,
        )

@dataclass
class LayerIntermediates:
    hiddens: Optional[List[Tensor]] = None
    attn_intermediates: Optional[List[Intermediates]] = None
    layer_hiddens: Optional[List[Tensor]] = None
    attn_z_loss: Optional[Tensor] = None


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner

print_once = once(print)

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)


def create_causal_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)


def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device=device)
    causal_mask = rearrange(r, "i -> i 1") < rearrange(r, "j -> 1 j")
    causal_mask = F.pad(causal_mask, (j - i, 0), value=False)
    return causal_mask


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner



class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class not_equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val



# token shifting


def shift(t, amount, mask=None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.0)

    return pad_at_dim(t, (amount, -amount), dim=-2, value=0.0)


class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get("mask", None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(
            map(
                lambda args: shift(*args, mask=mask),
                zip(segments_to_shift, shifts),
            )
        )
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)



# feedforward


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out, mult_bias=False):
        super().__init__()
        # project to 2 * dim_out, then split into two
        self.proj = nn.Linear(dim_in, dim_out * 2)
        # multiplicative bias
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.0

    def forward(self, x):
        # split the projection into two halves
        x, gate = self.proj(x).chunk(2, dim=-1)
        # apply the SiLU (Swish) activation to the gate
        gate = F.silu(gate)
        # return the element-wise product
        return x * gate * self.mult_bias


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        inner_dim=None,
        mult=4,
        glu=False,
        glu_mult_bias=False,
        swish=False,
        relu_squared=False,
        post_act_ln=False,
        dropout=0.0,
        no_bias=False,
        zero_init_output=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult) if inner_dim is None else inner_dim
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = SwiGLU(
                dim, inner_dim, activation, mult_bias=glu_mult_bias
            )
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias=not no_bias), activation
            )

        self.ff = Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias),
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)



class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.0,
        base=10000,
        base_rescale_factor=1.0,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        t = t / self.interpolation_factor

        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if not exists(self.scale):
            return freqs, 1.0

        power = (
            torch.arange(seq_len, device=device) - (seq_len // 2)
        ) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs, scale=1, seq_start_pos=None):
    seq_len = t.shape[-2]
    
    # default to selecting the last `seq_len` rows, as in the original code
    if seq_start_pos is None:
        freqs = freqs[-seq_len:, :]
    else:
        freqs = freqs[seq_start_pos:seq_start_pos + seq_len, :]

    return (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)


# activations

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


# positional embeddings

class RelativePositionBias(nn.Module):
    def __init__(
        self, scale, causal=False, num_buckets=32, max_distance=128, heads=8
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return bias * self.scale


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance=False, norm=False):
        super().__init__()
        assert (
            depth >= 1
        ), "depth for dynamic position bias MLP must be greater or equal to 1"
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(
            Sequential(
                nn.Linear(1, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU(),
            )
        )

        for _ in range(depth - 1):
            self.mlp.append(
                Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim) if norm else None,
                    nn.SiLU(),
                )
            )

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device=device)
        context_arange = torch.arange(n, device=device)
        indices = rearrange(seq_arange, "i -> i 1") - rearrange(
            context_arange, "j -> 1 j"
        )
        indices += n - 1

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device=device).float()
        pos = rearrange(pos, "... -> ... 1")

        if self.log_distance:
            # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases
        bias = pos[indices]
        bias = rearrange(bias, "i j h -> h i j")
        return bias


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            rearrange(j_arange, "j -> 1 1 j")
            - rearrange(i_arange, "i -> 1 i 1")
        )
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if (
            exists(self.bias)
            and self.bias.shape[-1] >= j
            and self.bias.shape[-2] >= i
        ):
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer("bias", bias, persistent=False)

        return self.bias


# norms

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        def scale_fn(t):
            return t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim**-0.5))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale



# residual and residual gates

class Residual(nn.Module):
    def __init__(self, dim, scale_residual=False, scale_residual_constant=1.0):
        super().__init__()
        self.residual_scale = (
            nn.Parameter(torch.ones(dim)) if scale_residual else None
        )
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual


class GRUGating(nn.Module):
    def __init__(self, dim, scale_residual=False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = (
            nn.Parameter(torch.ones(dim)) if scale_residual else None
        )

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, "b n d -> (b n) d"),
            rearrange(residual, "b n d -> (b n) d"),
        )

        return gated_output.reshape_as(x)


# tensor helpers


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head


# embedding


class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed=False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x)
        return l2norm(token_emb) if self.l2norm_embed else token_emb


# keyword argument helpers


def pick_and_pop(keys, d):
    values = list(map(d.pop, keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [{}, {}]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d
    )
    kwargs_without_prefix = dict(
        map(
            lambda x: (x[0][len(prefix) :], x[1]),
            tuple(kwargs_with_prefix.items()),
        )
    )
    return kwargs_without_prefix, kwargs


# initializations

def deepnorm_init(
    transformer, beta, module_name_match_list=[".ff.", ".to_v", ".to_out"]
):
    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        needs_beta_gain = any(
            map(lambda substr: substr in name, module_name_match_list)
        )
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain=gain)

        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)




# structured dropout, more effective than traditional attention dropouts


def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(
            seq_keep_counts, "b -> b 1"
        )

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask



# linear attention with mem-efficient softmax operations - 2020

# def split_at_index(dim, index, t):
#     pre_slices = (slice(None),) * dim
#     l = (*pre_slices, slice(None, index))
#     r = (*pre_slices, slice(index, None))
#     return t[l], t[r]

# def linear_attn(q, k, v, kv_mask = None):
#     dim = q.shape[-1]

#     if exists(kv_mask):
#         mask_value = max_neg_value(q)
#         mask = kv_mask[:, None, :, None]
#         k = k.masked_fill_(~mask, mask_value)
#         v = v.masked_fill_(~mask, 0.)
#         del mask

#     q = q.softmax(dim=-1)
#     k = k.softmax(dim=-2)

#     q = q * dim ** -0.5

#     context = einsum('bhnd,bhne->bhde', k, v)
#     attn = einsum('bhnd,bhde->bhne', q, context)
#     return attn.reshape(*q.shape)

# def causal_linear_attn(q, k, v, kv_mask = None, bucket_size = None, eps = 1e-3):
#     b, h, n, e, dtype = *q.shape, q.dtype
#     bucket_size = default(bucket_size, 64)
#     bucket_size = max(bucket_size, 1)
#     assert bucket_size == 0 or (n % bucket_size) == 0, f'sequence length {n} must be divisible by the bucket size {bucket_size} for causal linear attention'

#     q = q.softmax(dim=-1)
#     k = torch.exp(k).type(dtype).clone()

#     q = q * e ** -0.5

#     if exists(kv_mask):
#         mask = kv_mask[:, None, :, None]
#         k = k.masked_fill_(~mask, 0.)
#         v = v.masked_fill_(~mask, 0.)
#         del mask

#     bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, bucket_size, e)
#     b_q, b_k, b_v = map(bucket_fn, (q, k, v))

#     b_k_sum = b_k.sum(dim=-2)
#     b_k_cumsum = b_k_sum.cumsum(dim = -2).type(dtype)

#     context = einsum('bhund,bhune->bhude', b_k, b_v)
#     context = context.cumsum(dim = -3).type(dtype)

#     if bucket_size > 1:
#         context = F.pad(context, (0, 0, 0, 0, 1, 0), value = 0.)
#         context, _ = split_at_index(2, -1, context)

#         b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value = 0.)
#         b_k_cumsum, _ = split_at_index(2, -1, b_k_cumsum)

#     D_inv = 1. / einsum('bhud,bhund->bhun', b_k_cumsum, b_q).clamp(min = eps)
#     attn = einsum('bhund,bhude,bhun->bhune', b_q, context, D_inv)
#     return attn.reshape(*q.shape)




# zigzag ring attention - 2024

# from zigzag_ring_flash_attn import zigzag_ring_flash_attn_func



# attention is all you need - 2017
# + grouped query attention - 2023

class Attend(nn.Module):
    """
    Attend module performs attention mechanism for neural networks.

    Args:
        dropout (float): Dropout probability. Default is 0.0.
        causal (bool): Whether to use causal attention. Default is False.
        heads (int): Number of attention heads. Default is None.
        talking_heads (bool): Whether to use talking heads attention-linearprojections across the attention-heads dimension. Default is False.
        sparse_topk (int): Number of top-k values to consider for sparse attention. Default is None.
        scale (float): Scaling factor for attention scores. Default is None.
        qk_norm (bool): Whether to normalize query-key dot products. Default is False.
        flash (bool): Whether to use flash attention. Default is False.
        add_zero_kv (bool): Whether to add a key/value token composed of zeros. Default is False.
        onnxable (bool): Whether the module is ONNX compatible. Default is False.
    """

    def __init__(
        self,
        *,
        dropout=0.0,
        causal=False,
        num_attention_heads=None,
        talking_heads=False,
        sparse_topk=None,
        scale=None,
        qk_norm=False,
        flash=False,
        add_zero_kv=False,
        onnxable=False,
    ):
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm

        self.causal = causal
        self.create_causal_mask = (
            onnx_create_causal_mask if onnxable else create_causal_mask
        )

        self.attn_fn = (
            partial(F.softmax, dtype=torch.float32)
            if not qk_norm
            else F.softmax
        )

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (
            flash and talking_heads
        ), "talking heads not compatible with flash attention"

        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(
                num_attention_heads, num_attention_heads, 1, bias=False
            )
            self.post_softmax_talking_heads = nn.Conv2d(
                num_attention_heads, num_attention_heads, 1, bias=False
            )

        # sparse topk

        assert not (
            flash and sparse_topk
        ), "sparse topk not compatible with flash attention"
        self.sparse_topk = sparse_topk

        # add a key / value token composed of zeros
        # in case this helps controlling outliers, proposed by
        # https://www.evanmiller.org/attention-is-off-by-one.html

        self.add_zero_kv = add_zero_kv

        # flash attention

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), (
            "in order to use flash attention, you must be using pytorch 2.0 or"
            " above"
        )

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device("cuda")
        )

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on"
                " cuda"
            )
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention"
                " if input tensor is on cuda"
            )
            self.cuda_config = EfficientAttentionConfig(True, True, True)

        # self.global_attention = causal_linear_attn if self.causal else linear_attn

    def flash_attn(self, q, k, v, mask=None, attn_bias=None):
        """
        Perform flash attention.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor. Default is None.
            attn_bias (torch.Tensor): Attention bias tensor. Default is None.

        Returns:
            torch.Tensor: Output tensor.
            Intermediates: Intermediate values during attention computation.
        """

        batch, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # handle scale - by default they scale by dim_head ** -0.5, but need to
        # take care if using cosine sim attention

        if self.qk_norm:
            default_scale = q.shape[-1] ** -0.5
            q = q * (default_scale / self.scale)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            # manually handle causal mask, if another mask was given

            if causal:
                causal_mask = self.create_causal_mask(
                    q_len, k_len, device=device
                )
                mask = mask & ~causal_mask
                causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "h i j -> 1 h i j").expand(
                batch, heads, -1, -1
            )

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi
            # positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(
                    q_len, k_len, device=device
                )
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )
            # out = zigzag_ring_flash_attn_func(
            #     q,
            #     k,
            #     v,
            #     dropout_p=self.dropout if self.training else 0.0,
            #     causal=causal
            # )
            # if exists(mask):
            #     out = out.masked_fill(~mask, 0.0)

        return out, Intermediates()

    def forward(self, q, k, v, mask=None, attn_bias=None, prev_attn=None, seq_start_pos=None):
        """
        Perform forward pass of the Attend module.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor. Default is None.
            attn_bias (torch.Tensor): Attention bias tensor. Default is None.
            prev_attn (torch.Tensor): Previous attention tensor. Default is None.
            seq_start_pos (int): Starting position of the sequence. Default is None.

        Returns:
            torch.Tensor: Output tensor.
            Intermediates: Intermediate values during attention computation.
        """

        _n, heads, kv_heads, device = (
            q.shape[-2],
            q.shape[1],
            k.shape[1],
            q.device,
        )

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, "b 1 n d -> b n d"), (k, v))
        elif kv_heads < heads:
            k, v = map(
                lambda t: repeat(
                    t, "b kvh n d -> b (r kvh) n d", r=heads // kv_heads
                ),
                (k, v),
            )

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v = map(lambda t: F.pad(t, (0, 0, 1, 0), value=0.0), (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value=True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value=0.0)

        if self.flash:
            assert not exists(
                prev_attn
            ), "residual attention not compatible with flash attention"
            return self.flash_attn(q, k, v, mask=mask, attn_bias=attn_bias)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        dots = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        if exists(prev_attn):
            dots = dots + prev_attn

        qk_similarities = dots.clone()

        if self.talking_heads:
            dots = self.pre_softmax_talking_heads(dots)

        if exists(attn_bias):
            dots = dots + attn_bias

        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max

        if exists(self.sparse_topk) and self.sparse_topk < j:
            top_values, _ = dots.topk(self.sparse_topk, dim=-1)
            sparse_topk_mask = dots < top_values[..., -1:]
            mask = (
                (mask & sparse_topk_mask) if exists(mask) else sparse_topk_mask
            )

        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        if self.causal:
            causal_mask = self.create_causal_mask(i, j, device=device, seq_start_pos=seq_start_pos)
            dots = dots.masked_fill(causal_mask, mask_value)

        pre_softmax_attn = dots.clone()

        attn = self.attn_fn(dots, dim=-1)
        attn = attn.type(dtype)

        post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        if self.talking_heads:
            attn = self.post_softmax_talking_heads(attn)

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        intermediates = Intermediates(
            qk_similarities=qk_similarities,
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn,
        )

        return out, intermediates


class Attention(nn.Module):
    def __init__(
        self,
        in_dim,
        head_dim=DEFAULT_DIM_HEAD,
        q_heads=8,
        kv_heads=None,
        causal=False,
        flash=False,
        talking_heads=False,
        head_scale=False,
        sparse_topk=None,
        num_mem_kv=0,
        dropout=0.0,
        on_attn=False,
        gate_values=False,
        zero_init_output=False,
        max_attend_past=None,
        qk_norm=False,
        qk_norm_groups=1,
        qk_norm_scale=10,
        qk_norm_dim_scale=False,
        one_kv_head=False,
        shared_kv=False,
        value_dim_head=None,
        tensor_product=False,  # https://arxiv.org/abs/2208.06061
        cascading_heads=False,
        add_zero_kv=False,  # same as add_zero_attn in pytorch
        onnxable=False,
    ):
        super().__init__()
        self.scale = head_dim**-0.5

        self.heads = q_heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        assert not (exists(kv_heads) and one_kv_head), (
            "either attn_one_kv_head is set to True (in which case kv_heads is"
            " set to 1), or attn_kv_heads is set, but not both"
        )

        value_dim_head = default(value_dim_head, head_dim)
        kv_heads = default(kv_heads, q_heads)

        kv_heads = 1 if one_kv_head else kv_heads
        assert divisible_by(q_heads, kv_heads)

        self.kv_heads = kv_heads

        # grouped query attention
        q_dim = head_dim * q_heads
        k_dim = head_dim * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * q_heads

        self.to_q = nn.Linear(in_dim, q_dim, bias=False)
        self.to_k = nn.Linear(in_dim, k_dim, bias=False)

        # shared key / values, for further memory savings during inference
        assert not (
            shared_kv and value_dim_head != head_dim
        ), "key and value head dimensions must be equal for shared key / values"
        self.to_v = nn.Linear(in_dim, v_dim, bias=False) if not shared_kv else None

        # relations projection from tp-attention
        self.to_r = (
            nn.Linear(in_dim, v_dim, bias=False) if tensor_product else None
        )

        # add SwiGLU gating for aggregated values
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(in_dim, out_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 1)

        # cosine sim attention
        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim attention when
        # scale is equal to 1) - https://arxiv.org/abs/2302.05442
        self.qk_norm_dim_scale = qk_norm_dim_scale

        self.qk_norm_q_scale = self.qk_norm_k_scale = 1
        if qk_norm and qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.ones(head_dim))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(head_dim))

        assert (not qk_norm) or divisible_by(head_dim, qk_norm_groups), (
            "dimension per attention head must be divisible by the qk norm"
            " groups"
        )
        assert not (qk_norm and (head_dim // qk_norm_groups) <= 2), (
            "the group dimension may be too small (2 was too small in my tests,"
            " but 4 still works, surprisingly)"
        )

        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            num_attention_heads=q_heads,
            causal=causal,
            talking_heads=talking_heads,
            dropout=dropout,
            sparse_topk=sparse_topk,
            qk_norm=qk_norm,
            scale=qk_norm_scale if qk_norm else self.scale,
            add_zero_kv=add_zero_kv,
            flash=flash,
            onnxable=onnxable,
        )

        # if cascading_heads:
        #     # cascading heads - wrap the Attend logic
        #     self.attend = CascadingHeads(self.attend)

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, q_heads, 1, 1))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(q_heads, num_mem_kv, head_dim))
            self.mem_v = nn.Parameter(torch.randn(q_heads, num_mem_kv, head_dim))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = (
            nn.Sequential(nn.Linear(out_dim, in_dim * 2, bias=False), nn.GLU())
            if on_attn
            else nn.Linear(out_dim, in_dim, bias=False)
        )

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        rel_pos=None,
        rotary_pos_emb=None,
        prev_attn=None,
        mem=None,
        seq_start_pos=None,
    ):
        # einops conflicts with ruff, so noqa on next line
        b, n, _, h, kv_h, head_scale, device, has_context = (  # noqa F841
            *x.shape,
            self.heads,
            self.kv_heads,
            self.head_scale,
            x.device,
            exists(context),
        )
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input
        r_input = x

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if exists(self.to_v) else k
        r = self.to_r(r_input) if exists(self.to_r) else None

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        k, v, r = map(
            lambda t: maybe(rearrange)(t, "b n (h d) -> b h n d", h=kv_h),
            (k, v, r),
        )

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups=self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        if exists(rotary_pos_emb) and not has_context:
            freqs, xpos_scale = rotary_pos_emb
            l = freqs.shape[-1]  # noqa F741

            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale**-1.0)
                if exists(xpos_scale)
                else (1.0, 1.0)
            )
            (ql, qr), (kl, kr), (vl, vr) = map(
                lambda t: (t[..., :l], t[..., l:]), (q, k, v)
            )

            ql, kl, vl = map(
                lambda arg: apply_rotary_pos_emb(arg[0], freqs, arg[1], seq_start_pos=seq_start_pos),
                ((ql, q_xpos_scale), (kl, k_xpos_scale), (vl, k_xpos_scale)),
            )
            q, k, v = map(
                lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr))
            )

        input_mask = context_mask if has_context else mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(
                lambda t: repeat(t, "h n d -> b h n d", b=b),
                (self.mem_k, self.mem_v),
            )

            if self.qk_norm:
                mem_k = l2norm(mem_k)
                mem_k = mem_k * self.qk_norm_k_scale

            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)

            if exists(input_mask):
                input_mask = pad_at_dim(
                    input_mask, (self.num_mem_kv, 0), dim=-1, value=True
                )

        i, j = map(lambda t: t.shape[-2], (q, k))

        # determine masking

        max_neg_value(q)
        masks = []
        final_attn_mask = None

        if exists(input_mask):
            input_mask = rearrange(input_mask, "b j -> b 1 1 j")
            masks.append(~input_mask)

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, (
                "attention mask must have greater than 2 dimensions but less"
                " than or equal to 4"
            )
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, "i j -> 1 1 i j")
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, "h i j -> 1 h i j")
            masks.append(~attn_mask)

        if exists(self.max_attend_past):
            range_q = torch.arange(j - i, j, device=device)
            range_k = torch.arange(j, device=device)
            dist = rearrange(range_q, "i -> 1 1 i 1") - rearrange(
                range_k, "j -> 1 1 1 j"
            )
            max_attend_past_mask = dist > self.max_attend_past
            masks.append(max_attend_past_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        attn_bias = None
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)

        # attention is all we need

        out, intermediates = self.attend(
            q,
            k,
            v,
            mask=final_attn_mask,
            attn_bias=attn_bias,
            prev_attn=prev_attn,
            seq_start_pos=seq_start_pos,
        )

        # https://arxiv.org/abs/2208.06061 proposes to add a residual for
        # better gradients

        if exists(r):
            out = out * r + out

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")

        # alphafold2 styled gating of the values

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * gates.sigmoid()

        # combine the heads

        out = self.to_out(out)

        if exists(mask):
            mask = rearrange(mask, "b n -> b n 1")
            out = out.masked_fill(~mask, 0.0)

        return out, intermediates


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_simple_rmsnorm=False,
        alibi_pos_bias=False,
        alibi_num_heads=None,
        rel_pos_bias=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        dynamic_pos_bias=False,
        dynamic_pos_bias_log_distance=False,
        dynamic_pos_bias_mlp_depth=2,
        dynamic_pos_bias_norm=False,
        rotary_pos_emb=False,
        rotary_emb_dim=None,
        rotary_xpos=False,
        rotary_interpolation_factor=1.0,
        rotary_xpos_scale_base=512,
        rotary_base=10000,
        rotary_base_rescale_factor=1.0,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        pre_norm_has_final_norm=True,
        gate_residual=False,
        scale_residual=False,
        scale_residual_constant=1.0,
        deepnorm=False,
        shift_tokens=0,
        sandwich_norm=False,
        resi_dual=False,
        resi_dual_scale=1.0,
        zero_init_branch_output=False,
        layer_dropout=0.0,
        cross_attn_tokens_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim("attn_", kwargs)

        dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = rel_pos_bias or rotary_pos_emb

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)

        assert not (
            rotary_xpos and not causal
        ), "rotary xpos is not compatible with bidirectional attention"
        self.rotary_pos_emb = (
            RotaryEmbedding(
                rotary_emb_dim,
                use_xpos=rotary_xpos,
                base=rotary_base,
                scale_base=rotary_xpos_scale_base,
                interpolation_factor=rotary_interpolation_factor,
                base_rescale_factor=rotary_base_rescale_factor,
            )
            if rotary_pos_emb
            else None
        )

        assert not (alibi_pos_bias and rel_pos_bias), (
            "you can only choose Alibi positional bias or T5 relative"
            " positional bias, not both"
        )
        assert rel_pos_num_buckets <= rel_pos_max_distance, (
            "number of relative position buckets must be less than the relative"
            " position max distance"
        )

        # relative positional bias

        flash_attn = attn_kwargs.get("flash", False)
        assert (
            int(rel_pos_bias) + int(dynamic_pos_bias) + int(alibi_pos_bias)
        ) <= 1, (
            "you can only choose up to one of t5, alibi, or dynamic positional"
            " bias"
        )

        self.rel_pos = None
        if rel_pos_bias:
            assert (
                not flash_attn
            ), "flash attention not compatible with t5 relative positional bias"
            self.rel_pos = RelativePositionBias(
                scale=dim_head**0.5,
                causal=causal,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
        elif dynamic_pos_bias:
            assert (
                not flash_attn
            ), "flash attention not compatible with dynamic positional bias"
            self.rel_pos = DynamicPositionBias(
                dim=dim // 4,
                heads=heads,
                log_distance=dynamic_pos_bias_log_distance,
                depth=dynamic_pos_bias_mlp_depth,
                norm=dynamic_pos_bias_norm,
            )
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, (
                "number of ALiBi heads must be less than the total number of"
                " heads"
            )
            self.rel_pos = AlibiPositionalBias(
                heads=alibi_num_heads, total_heads=heads
            )

        # determine deepnorm and residual scale

        if deepnorm:
            assert scale_residual_constant == 1, (
                "scale residual constant is being overridden by deep norm"
                " settings"
            )
            pre_norm = sandwich_norm = resi_dual = False
            scale_residual = True
            scale_residual_constant = (2 * depth) ** 0.25

        assert (
            int(sandwich_norm) + int(resi_dual)
        ) <= 1, "either sandwich norm or resiDual is selected, but not both"
        assert not (
            not pre_norm and sandwich_norm
        ), "sandwich norm cannot be used when not using prenorm"

        if resi_dual:
            pre_norm = False

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.resi_dual = resi_dual
        assert 0 < resi_dual_scale <= 1.0, (
            "resiDual prenorm residual must be scaled by a factor greater than"
            " 0 and less than or equal to 1."
        )
        self.resi_dual_scale = resi_dual_scale

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (
            flash_attn and (residual_attn or cross_residual_attn)
        ), "flash attention is not compatible with residual attention"

        self.cross_attend = cross_attend

        assert (
            int(use_scalenorm) + int(use_rmsnorm) + int(use_simple_rmsnorm)
        ) <= 1, "you can only use either scalenorm, rmsnorm, or simple rmsnorm"

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        else:
            norm_class = nn.LayerNorm

        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")

        if macaron:
            default_block = ("f",) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, "zero_init_output": True}
            ff_kwargs = {**ff_kwargs, "zero_init_output": True}

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            # 2 / 3 attention layer cutoff suggested by PAR paper
            depth_cut = par_depth * 2 // 3
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert (
                len(default_block) <= par_width
            ), "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (
                par_width - len(default_block)
            )
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert (
                sandwich_coef > 0 and sandwich_coef <= depth
            ), "sandwich coefficient should be less than the depth"
            layer_types = (
                ("a",) * sandwich_coef
                + default_block * (depth - sandwich_coef)
                + ("f",) * sandwich_coef
            )
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm or resi_dual else nn.Identity()

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(
            zip(self.layer_types, shift_tokens)
        ):
            ind == (len(self.layer_types) - 1)

            if layer_type == "a":
                layer = Attention(
                    dim, q_heads=heads, causal=causal, **attn_kwargs
                )
            elif layer_type == "c":
                layer = Attention(dim, q_heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type {layer_type}")

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(
                    range(shift_range_lower, shift_range_upper), layer
                )

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(
                dim,
                scale_residual=scale_residual,
                scale_residual_constant=scale_residual_constant,
            )

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = nn.ModuleList(
                [pre_branch_norm, post_branch_norm, post_main_norm]
            )

            self.layers.append(nn.ModuleList([norms, layer, residual]))

        if deepnorm:
            init_gain = (8 * depth) ** -0.25
            deepnorm_init(self, init_gain)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        self_attn_context_mask=None,
        mems=None,
        return_hiddens=False,
        seq_start_pos=None,
        cache=None
    ):
        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_length = x.shape[1]
            if exists(seq_start_pos):
                rotary_length += seq_start_pos
            max_rotary_emb_length = max(
                list(
                    map(
                        lambda m: (m.shape[1] if exists(m) else 0) + rotary_length,
                        mems,
                    )
                )
            )
            rotary_pos_emb = self.rotary_pos_emb(
                max_rotary_emb_length, x.device
            )

        outer_residual = x * self.resi_dual_scale

        for ind, (
            layer_type,
            (norm, block, residual_fn),
            layer_dropout,
        ) in enumerate(zip(self.layer_types, self.layers, self.layer_dropouts)):
            ind == (len(self.layers) - 1)

            if (
                self.training
                and layer_dropout > 0.0
                and random() < layer_dropout
            ):
                continue

            if layer_type == "a":
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            if layer_type == "c":
                if self.training and self.cross_attn_tokens_dropout > 0.0:
                    context, context_mask = dropout_seq(
                        context, context_mask, self.cross_attn_tokens_dropout
                    )

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_norm):
                x = pre_norm(x)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    context_mask=self_attn_context_mask,
                    attn_mask=attn_mask,
                    rel_pos=self.rel_pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    mem=layer_mem,
                    seq_start_pos=seq_start_pos,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                )
            elif layer_type == "f":
                out = block(x)

            if self.resi_dual:
                outer_residual = outer_residual + out * self.resi_dual_scale

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ("a", "c") and return_hiddens:
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.resi_dual:
            x = x + self.final_norm(outer_residual)
        else:
            x = self.final_norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates,
                layer_hiddens=layer_hiddens,
            )

            return x, intermediates

        return x