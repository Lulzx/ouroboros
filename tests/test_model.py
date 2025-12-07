"""Tests for GRN transformer model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from src.data.tokenizer import GRNTokenizer
from src.model.transformer import GRNTransformer, ModelArgs
from src.model.generation import generate


def test_model_creation():
    """Test model initializes correctly."""
    args = ModelArgs(
        vocab_size=150,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
    )
    model = GRNTransformer(args)

    assert model.num_parameters > 0
    print(f"Model parameters: {model.num_parameters:,}")


def test_forward_pass():
    """Test forward pass produces correct output shape."""
    args = ModelArgs(
        vocab_size=150,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
    )
    model = GRNTransformer(args)

    # Create dummy input
    batch_size = 4
    seq_len = 10
    input_ids = mx.randint(0, 150, (batch_size, seq_len))

    # Forward pass
    logits = model.forward(input_ids)
    mx.eval(logits)

    assert logits.shape == (batch_size, seq_len, 150)


def test_generation():
    """Test autoregressive generation."""
    tokenizer = GRNTokenizer()

    args = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
    )
    model = GRNTransformer(args)

    # Create prompt: <bos> <oscillator>
    phenotype_id = tokenizer.token_to_id["<oscillator>"]
    prompt = mx.array([[tokenizer.bos_token_id, phenotype_id]])

    # Generate
    generated = generate(
        model,
        prompt,
        max_length=20,
        temperature=1.0,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    mx.eval(generated)

    assert generated.shape[0] == 1
    assert generated.shape[1] >= 2  # At least prompt length


def test_log_likelihood():
    """Test log likelihood computation."""
    args = ModelArgs(
        vocab_size=150,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
    )
    model = GRNTransformer(args)

    # Create dummy sequence
    input_ids = mx.randint(0, 150, (2, 10))

    # Compute log likelihood
    log_probs = model.log_likelihood(input_ids, reduction="sum")
    mx.eval(log_probs)

    assert log_probs.shape == (2,)
    assert all(log_probs < 0)  # Log probs should be negative


def test_get_logprobs():
    """Test get_logprobs for GRPO."""
    args = ModelArgs(
        vocab_size=150,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
    )
    model = GRNTransformer(args)

    input_ids = mx.randint(0, 150, (2, 10))

    all_log_probs, token_log_probs = model.get_logprobs(input_ids)
    mx.eval(all_log_probs, token_log_probs)

    assert all_log_probs.shape == (2, 9, 150)  # seq_len - 1
    assert token_log_probs.shape == (2, 9)


if __name__ == "__main__":
    test_model_creation()
    print("✓ test_model_creation")

    test_forward_pass()
    print("✓ test_forward_pass")

    test_generation()
    print("✓ test_generation")

    test_log_likelihood()
    print("✓ test_log_likelihood")

    test_get_logprobs()
    print("✓ test_get_logprobs")

    print("\nAll model tests passed!")
