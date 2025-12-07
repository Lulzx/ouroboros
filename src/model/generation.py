"""Generation utilities for GRN transformer."""

from typing import Optional

import mlx.core as mx

from .transformer import GRNTransformer


def sample_top_p(logits: mx.array, top_p: float, temperature: float = 1.0) -> mx.array:
    """
    Sample from logits using nucleus (top-p) sampling.

    Args:
        logits: Logits of shape (batch, vocab)
        top_p: Cumulative probability threshold
        temperature: Sampling temperature

    Returns:
        Sampled token IDs of shape (batch,)
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to probabilities
    probs = mx.softmax(logits, axis=-1)

    # Sort probabilities descending
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

    # Create mask for tokens to keep (cumsum <= top_p, but keep at least 1)
    mask = cumsum_probs <= top_p
    # Shift mask to include the first token that exceeds top_p
    mask = mx.concatenate(
        [mx.ones((mask.shape[0], 1), dtype=mx.bool_), mask[:, :-1]], axis=-1
    )

    # Zero out probabilities for tokens outside top-p
    sorted_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)

    # Sample from sorted distribution
    cumsum = mx.cumsum(sorted_probs, axis=-1)
    uniform = mx.random.uniform(shape=(logits.shape[0], 1))
    sorted_idx = mx.argmax((cumsum >= uniform).astype(mx.int32), axis=-1)

    # Map back to original indices
    sampled = mx.take_along_axis(
        sorted_indices, sorted_idx.reshape(-1, 1), axis=-1
    ).squeeze(-1)

    return sampled


def sample_top_k(logits: mx.array, top_k: int, temperature: float = 1.0) -> mx.array:
    """
    Sample from logits using top-k sampling.

    Args:
        logits: Logits of shape (batch, vocab)
        top_k: Number of top tokens to consider
        temperature: Sampling temperature

    Returns:
        Sampled token IDs of shape (batch,)
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Get top-k indices and values
    top_k = min(top_k, logits.shape[-1])

    # Sort and get top-k
    sorted_indices = mx.argsort(-logits, axis=-1)
    top_k_indices = sorted_indices[:, :top_k]
    top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)

    # Sample from top-k
    probs = mx.softmax(top_k_logits, axis=-1)
    cumsum = mx.cumsum(probs, axis=-1)
    uniform = mx.random.uniform(shape=(logits.shape[0], 1))
    idx = mx.argmax((cumsum >= uniform).astype(mx.int32), axis=-1)

    # Map back to original indices
    sampled = mx.take_along_axis(top_k_indices, idx.reshape(-1, 1), axis=-1).squeeze(-1)

    return sampled


def generate(
    model: GRNTransformer,
    prompt_ids: mx.array,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    eos_token_id: int = 2,
    pad_token_id: int = 0,
) -> mx.array:
    """
    Generate sequences autoregressively.

    Args:
        model: GRNTransformer model
        prompt_ids: Prompt token IDs of shape (batch, prompt_len)
        max_length: Maximum total sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (None to disable)
        top_p: Top-p (nucleus) sampling parameter (None to disable)
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID

    Returns:
        Generated token IDs of shape (batch, seq_len)
    """
    B, L = prompt_ids.shape
    generated = prompt_ids

    # Initialize cache
    cache = None

    # Track which sequences have finished
    finished = mx.zeros((B,), dtype=mx.bool_)

    for _ in range(max_length - L):
        # Get logits for last position
        if cache is None:
            logits, cache = model(generated)
            logits = logits[:, -1, :]
        else:
            # Only process last token with cache
            logits, cache = model(generated[:, -1:], cache)
            logits = logits[:, -1, :]

        # Sample next token
        if top_p is not None and top_p < 1.0:
            next_token = sample_top_p(logits, top_p, temperature)
        elif top_k is not None:
            next_token = sample_top_k(logits, top_k, temperature)
        else:
            # Greedy sampling
            if temperature != 1.0:
                logits = logits / temperature
            next_token = mx.argmax(logits, axis=-1)

        # Replace with pad for finished sequences
        next_token = mx.where(finished, pad_token_id, next_token)

        # Append to generated
        generated = mx.concatenate(
            [generated, next_token.reshape(-1, 1)], axis=1
        )

        # Update finished status
        finished = finished | (next_token == eos_token_id)

        # Stop if all finished
        if mx.all(finished):
            break

        mx.eval(generated, finished, cache)

    return generated


def generate_batch(
    model: GRNTransformer,
    phenotype_ids: mx.array,
    tokenizer,
    num_samples: int = 4,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
) -> list[mx.array]:
    """
    Generate multiple samples for each phenotype.

    Args:
        model: GRNTransformer model
        phenotype_ids: Phenotype token IDs to condition on, shape (num_phenotypes,)
        tokenizer: GRNTokenizer for special tokens
        num_samples: Number of samples per phenotype
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter

    Returns:
        List of generated sequences for each phenotype
    """
    results = []

    for phenotype_id in phenotype_ids:
        # Create prompts: <bos> <phenotype>
        prompts = mx.array([[tokenizer.bos_token_id, int(phenotype_id)]] * num_samples)

        # Generate
        generated = generate(
            model,
            prompts,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        results.append(generated)

    return results


def compute_sequence_log_probs(
    model: GRNTransformer,
    sequences: mx.array,
    pad_token_id: int = 0,
) -> mx.array:
    """
    Compute log probability of complete sequences.

    Args:
        model: GRNTransformer model
        sequences: Token IDs of shape (batch, seq_len)
        pad_token_id: Padding token ID to ignore

    Returns:
        Log probabilities of shape (batch,)
    """
    # Get per-token log probs
    _, token_log_probs = model.get_logprobs(sequences)

    # Create mask for non-padding tokens (shifted since log_probs is for prediction)
    mask = (sequences[:, 1:] != pad_token_id).astype(mx.float32)

    # Sum log probs for valid tokens
    return (token_log_probs * mask).sum(axis=-1)
