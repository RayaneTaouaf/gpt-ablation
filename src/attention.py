"""
Multi-head causal self-attention, with RoPE and KV cache.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .rope import apply_rope


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal attention.

    "Causal": the token at position t can only look at positions <= t. Without
    this mask the model would see the future and the task would become trivial
    (it could just copy the next token). The loss would drop to ~0 and the model
    would be useless at generation.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # A single linear projection produces Q, K and V at once, then we split.
        # More efficient than three separate nn.Linear layers: one matrix
        # multiply instead of three.
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x, cos, sin, kv_cache=None):
        """
        x   : (B, T, C)
        cos, sin : RoPE tables sliced to the right positions
        kv_cache : None during training. At generation time, a dict
                   {"k":..., "v":...} holding the already-computed keys/values.

        Returns (output, new_cache).
        """
        B, T, C = x.shape

        # Project then split into q, k, v
        qkv = self.qkv(x)                                  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)                      # 3 x (B, T, C)

        # Reshape into heads: (B, T, C) -> (B, n_head, T, head_dim)
        # Each head looks at a different subspace of the representation.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE on q and k only, NEVER on v.
        # Position should influence WHO attends to WHOM (the attention score),
        # not the information being transported.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # --- KV cache ---
        # At generation time we produce one token at a time. Without a cache we
        # would recompute k and v for the whole sequence on every new token:
        # pointless quadratic cost. With a cache we compute only the new token
        # and concatenate. q is never cached: we only query with the current
        # token.
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            new_cache = {"k": k, "v": v}
        else:
            new_cache = None

        # Causal mask: only needed when processing more than one position. During
        # cached generation (T == 1) the current token may attend to the whole
        # past held in the cache, so no mask is required.
        is_causal = (kv_cache is None) or (T > 1)

        # PyTorch's scaled_dot_product_attention implements
        # softmax(QK^T / sqrt(d)) V with optimized kernels (Flash Attention when
        # available). The 1/sqrt(head_dim) factor keeps dot products from growing
        # with dimension and saturating the softmax.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Merge the heads back: (B, nh, T, hd) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))

        return y, new_cache


class MLP(nn.Module):
    """
    The transformer block's feed-forward network.

    4x expansion then back down: this is where most of the model's parameters
    live. Attention mixes information across positions; the MLP transforms it
    position by position.

    GELU rather than ReLU: the standard transformer choice, smoother around zero.
    """

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))
