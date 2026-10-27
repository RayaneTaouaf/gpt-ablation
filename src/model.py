"""
The model: a decoder-only GPT, with pre-norm or post-norm selectable.

THIS IS WHERE THE ABLATION HAPPENS.

Post-norm (original "Attention Is All You Need" paper, 2017):
    x = LayerNorm(x + Attention(x))
    Normalization comes AFTER the residual addition.

Pre-norm (GPT-2 and everything since):
    x = x + Attention(LayerNorm(x))
    Normalization comes BEFORE the submodule, inside the branch.

Why it matters: in pre-norm the residual path runs from input to output without
passing through any normalization, so gradients flow back through it without
being rescaled at each layer. In post-norm every layer renormalizes the main
path, and gradients degrade with depth. In practice post-norm needs careful
learning-rate warmup to converge; pre-norm is much more tolerant.

The ablation measures this gap on our data.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention import CausalSelfAttention, MLP
from .rope import build_rope_cache


class Block(nn.Module):
    """A transformer block. norm_style is "pre" or "post"."""

    def __init__(self, n_embd, n_head, dropout=0.0, norm_style="pre"):
        super().__init__()
        assert norm_style in ("pre", "post")
        self.norm_style = norm_style

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, cos, sin, kv_cache=None):
        if self.norm_style == "pre":
            # x + f(LN(x)): the residual path stays untouched
            attn_out, new_cache = self.attn(self.ln1(x), cos, sin, kv_cache)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
        else:
            # LN(x + f(x)): normalization sits on the main path
            attn_out, new_cache = self.attn(x, cos, sin, kv_cache)
            x = self.ln1(x + attn_out)
            x = self.ln2(x + self.mlp(x))

        return x, new_cache


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6,
                 block_size=256, dropout=0.0, norm_style="pre"):
        super().__init__()
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.norm_style = norm_style

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # No learned positional embedding: RoPE handles position in attention.
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout, norm_style) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: the embedding matrix and the output layer share their
        # weights. Saves parameters and usually improves quality (used by GPT-2).
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        # Scaled initialization on the residual projections.
        # With n_layer blocks adding into the residual stream, variance grows
        # like sqrt(n_layer), so we divide by sqrt(2 * n_layer) to keep variance
        # stable with depth. A detail from GPT-2, and it really matters for
        # post-norm.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        # RoPE cache, registered as a buffer (follows the model onto GPU, but is
        # not a trainable parameter).
        cos, sin = build_rope_cache(block_size, self.head_dim, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence {T} > block_size {self.block_size}"

        x = self.drop(self.token_emb(idx))

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for block in self.blocks:
            x, _ = block(x, cos, sin)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        # cross_entropy applies the log-softmax internally: we pass it raw
        # logits, never an output already run through softmax.
        loss = F.cross_entropy(
            logits.view(B * T, -1),
            targets.reshape(B * T),
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """
        Generate text token by token.

        use_cache=True enables the KV cache: at each step only the LAST token is
        passed through the model, since the previous keys/values are already
        stored. Without the cache, the whole sequence is re-run on every token.
        """
        self.eval()
        caches = [{"k": None, "v": None} for _ in self.blocks] if use_cache else None
        pos = 0

        for step in range(max_new_tokens):
            if use_cache:
                # First iteration: the whole prompt. After that: a single token.
                idx_cond = idx if step == 0 else idx[:, -1:]
            else:
                idx_cond = idx[:, -self.block_size:]

            T = idx_cond.shape[1]

            if use_cache:
                cos = self.rope_cos[pos:pos + T]
                sin = self.rope_sin[pos:pos + T]
                pos += T
            else:
                cos = self.rope_cos[:T]
                sin = self.rope_sin[:T]

            x = self.token_emb(idx_cond)
            for i, block in enumerate(self.blocks):
                cache = caches[i] if use_cache else None
                x, new_cache = block(x, cos, sin, cache)
                if use_cache:
                    caches[i] = new_cache

            logits = self.head(self.ln_f(x))[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

            if pos >= self.block_size:
                break  # context limit reached

        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
