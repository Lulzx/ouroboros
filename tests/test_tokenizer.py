"""Tests for GRN tokenizer."""

import sys
from pathlib import Path

# Add src to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))

from tokenizer import GRNTokenizer


def test_tokenizer_initialization():
    """Test tokenizer initializes correctly."""
    tokenizer = GRNTokenizer()
    assert tokenizer.vocab_size > 0
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2


def test_encode_decode():
    """Test encoding and decoding a circuit."""
    tokenizer = GRNTokenizer()

    circuit = {
        "phenotype": "oscillator",
        "interactions": [
            {"source": "lacI", "target": "tetR", "type": "inhibits"},
            {"source": "tetR", "target": "cI", "type": "inhibits"},
            {"source": "cI", "target": "lacI", "type": "inhibits"},
        ],
    }

    # Encode
    tokens = tokenizer.encode_flat(circuit)
    assert tokens[0] == tokenizer.bos_token_id
    assert tokens[-1] == tokenizer.eos_token_id

    # Decode
    decoded = tokenizer.decode(tokens)
    assert decoded["phenotype"] == "oscillator"
    assert len(decoded["interactions"]) == 3


def test_phenotype_tokens():
    """Test phenotype token handling."""
    tokenizer = GRNTokenizer()

    phenotype_ids = tokenizer.get_phenotype_token_ids()
    assert len(phenotype_ids) == 6  # 6 phenotype classes

    # Check we can get phenotype names back
    for pid in phenotype_ids:
        name = tokenizer.get_phenotype_name(pid)
        assert name in [
            "oscillator",
            "toggle_switch",
            "adaptation",
            "pulse_generator",
            "amplifier",
            "stable",
        ]


def test_batch_encoding():
    """Test batch encoding with padding."""
    tokenizer = GRNTokenizer()

    circuits = [
        {
            "phenotype": "stable",
            "interactions": [
                {"source": "geneA", "target": "geneB", "type": "activates"}
            ],
        },
        {
            "phenotype": "oscillator",
            "interactions": [
                {"source": "lacI", "target": "tetR", "type": "inhibits"},
                {"source": "tetR", "target": "cI", "type": "inhibits"},
                {"source": "cI", "target": "lacI", "type": "inhibits"},
            ],
        },
    ]

    padded, lengths = tokenizer.encode_batch(circuits)
    assert len(padded) == 2
    assert len(padded[0]) == len(padded[1])  # Same length after padding
    assert lengths[0] < lengths[1]  # First circuit is shorter


if __name__ == "__main__":
    test_tokenizer_initialization()
    print("✓ test_tokenizer_initialization")

    test_encode_decode()
    print("✓ test_encode_decode")

    test_phenotype_tokens()
    print("✓ test_phenotype_tokens")

    test_batch_encoding()
    print("✓ test_batch_encoding")

    print("\nAll tokenizer tests passed!")
