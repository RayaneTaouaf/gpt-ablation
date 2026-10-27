"""
Rotary Positional Embeddings (RoPE).

The problem: attention is order-invariant. With no positional information,
"the cat eats the mouse" and "the mouse eats the cat" give the same result.

RoPE encodes position by ROTATING the query and key vectors by an angle
proportional to their position. The property that makes RoPE elegant: the dot
product between q at position m and k at position n then depends only on
(m - n), the relative distance. Relative position emerges from the rotation,
without needing to encode it explicitly.

This is what LLaMA, Mistral and GPT-NeoX use.
"""

import torch


def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    """
    Pre-compute cos and sin for all positions.

    These values are not recomputed on every forward: they depend only on
    position and dimension, never on the data, so we compute them once.

    head_dim must be even: coordinates are processed in pairs (x1, x2), each
    pair undergoing a 2D rotation.

    Returns cos, sin of shape (seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Decreasing frequencies: theta_i = base^(-2i/d).
    # Early dimensions rotate fast (encoding local position), later ones rotate
    # slowly (global position). Same idea as the sinusoidal encodings in the
    # original paper.
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim/2)

    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation to x.

    x   : (B, n_heads, T, head_dim)
    cos : (T, head_dim/2)
    sin : (T, head_dim/2)

    We split x into pairs of adjacent coordinates and apply the standard 2D
    rotation to each pair:
        x1' = x1*cos - x2*sin
        x2' = x1*sin + x2*cos
    """
    # Separate even and odd indices -> two halves of shape (B, nh, T, hd/2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # (T, hd/2) -> (1, 1, T, hd/2) to broadcast over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos

    # Re-interleave: stack then flatten restores the pairs in their original order
    out = torch.stack((rot1, rot2), dim=-1).flatten(start_dim=-2)
    return out
