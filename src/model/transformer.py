"""Transformer model for GRN generation using MLX."""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    """Model configuration arguments."""

    vocab_size: int = 150
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.1
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    """Multi-head attention with rotary position embeddings."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_heads
        self.head_dim = args.embed_dim // args.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.k_proj = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.v_proj = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.o_proj = nn.Linear(args.embed_dim, args.embed_dim, bias=False)

        self.rope = nn.RoPE(
            self.head_dim, traditional=True, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[tuple[mx.array, mx.array]] = None,
    ) -> tuple[mx.array, Optional[tuple[mx.array, mx.array]]]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Handle KV cache for generation
        if cache is not None:
            k_cache, v_cache = cache
            q = self.rope(q, offset=k_cache.shape[2])
            k = self.rope(k, offset=k_cache.shape[2])
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        new_cache = (k, v)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Reshape and project
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), new_cache


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.embed_dim, args.mlp_dim, bias=False)
        self.up_proj = nn.Linear(args.embed_dim, args.mlp_dim, bias=False)
        self.down_proj = nn.Linear(args.mlp_dim, args.embed_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.mlp = MLP(args)
        self.attention_norm = RMSNorm(args.embed_dim)
        self.mlp_norm = RMSNorm(args.embed_dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[tuple[mx.array, mx.array]] = None,
    ) -> tuple[mx.array, Optional[tuple[mx.array, mx.array]]]:
        # Pre-norm attention
        h, new_cache = self.attention(self.attention_norm(x), mask, cache)
        x = x + h

        # Pre-norm MLP
        x = x + self.mlp(self.mlp_norm(x))

        return x, new_cache


class GRNTransformer(nn.Module):
    """
    Decoder-only transformer for gene regulatory network generation.

    Follows modern LLM architecture patterns:
    - RoPE positional embeddings
    - Pre-normalization
    - SwiGLU activation
    - RMSNorm
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embed_tokens = nn.Embedding(args.vocab_size, args.embed_dim)
        self.layers = [TransformerBlock(args) for _ in range(args.num_layers)]
        self.norm = RMSNorm(args.embed_dim)
        self.lm_head = nn.Linear(args.embed_dim, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[list[tuple[mx.array, mx.array]]] = None,
    ) -> tuple[mx.array, Optional[list[tuple[mx.array, mx.array]]]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            cache: Optional KV cache for generation

        Returns:
            Tuple of (logits, new_cache)
        """
        B, L = input_ids.shape

        # Token embeddings
        h = self.embed_tokens(input_ids)

        # Create causal mask
        mask = None
        if L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(h.dtype)

        # If using cache, adjust mask
        if cache is not None and cache[0] is not None:
            cache_len = cache[0][0].shape[2]
            # Create mask for new tokens attending to cached + new tokens
            mask = mx.zeros((L, cache_len + L))
            # New tokens can't attend to future new tokens
            for i in range(L):
                mask[i, cache_len + i + 1 :] = float("-inf")
            mask = mask.astype(h.dtype)

        # Process through layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, new_layer_cache = layer(h, mask, layer_cache)
            new_cache.append(new_layer_cache)

        # Final norm and project to vocab
        h = self.norm(h)
        logits = self.lm_head(h)

        return logits, new_cache

    def forward(self, input_ids: mx.array) -> mx.array:
        """Simple forward pass returning only logits (no cache)."""
        logits, _ = self(input_ids, cache=None)
        return logits

    def log_likelihood(
        self,
        input_ids: mx.array,
        reduction: str = "sum",
    ) -> mx.array:
        """
        Compute log probability of token sequence.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            reduction: "sum", "mean", or "none"

        Returns:
            Log probability per sequence
        """
        logits = self.forward(input_ids)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Compute log probabilities
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

        # Gather log probs for actual tokens
        B, L, V = log_probs.shape
        token_log_probs = mx.take_along_axis(
            log_probs.reshape(B * L, V),
            shift_labels.reshape(B * L, 1),
            axis=1,
        ).reshape(B, L)

        if reduction == "sum":
            return token_log_probs.sum(axis=-1)
        elif reduction == "mean":
            return token_log_probs.mean(axis=-1)
        else:
            return token_log_probs

    def get_logprobs(
        self,
        input_ids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Get per-token log probabilities for GRPO.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)

        Returns:
            Tuple of (all_log_probs, token_log_probs)
            - all_log_probs: Shape (batch, seq_len-1, vocab)
            - token_log_probs: Shape (batch, seq_len-1)
        """
        logits = self.forward(input_ids)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Compute log probabilities
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

        # Gather log probs for actual tokens
        B, L, V = log_probs.shape
        token_log_probs = mx.take_along_axis(
            log_probs.reshape(B * L, V),
            shift_labels.reshape(B * L, 1),
            axis=1,
        ).reshape(B, L)

        return log_probs, token_log_probs

    @property
    def num_parameters(self) -> int:
        """Count total parameters."""
        def count_params(params):
            total = 0
            if isinstance(params, dict):
                for v in params.values():
                    total += count_params(v)
            elif isinstance(params, list):
                for v in params:
                    total += count_params(v)
            elif hasattr(params, 'size'):
                total += params.size
            return total
        return count_params(self.parameters())


def create_model(
    vocab_size: int = 150,
    embed_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    mlp_dim: int = 1024,
    max_seq_len: int = 128,
    dropout: float = 0.1,
) -> GRNTransformer:
    """Create a GRN transformer model with specified configuration."""
    args = ModelArgs(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    return GRNTransformer(args)
