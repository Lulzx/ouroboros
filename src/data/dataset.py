"""Dataset classes for gene regulatory network circuits."""

import json
import random
from pathlib import Path
from typing import Optional, Iterator

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

try:
    from .tokenizer import GRNTokenizer
except ImportError:
    from tokenizer import GRNTokenizer


class GRNDataset:
    """Dataset for gene regulatory network circuits."""

    def __init__(
        self,
        circuits: list[dict],
        tokenizer: GRNTokenizer,
        max_length: int = 128,
        augment: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            circuits: List of circuit dictionaries
            tokenizer: GRNTokenizer instance
            max_length: Maximum sequence length
            augment: Whether to apply data augmentation (shuffle interaction order)
        """
        self.circuits = circuits
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

        # Pre-encode all circuits
        self._encode_circuits()

    def _encode_circuits(self):
        """Encode all circuits to token sequences."""
        self.encoded = []
        for circuit in self.circuits:
            tokens = self.tokenizer.encode_flat(circuit)
            if len(tokens) <= self.max_length:
                self.encoded.append(tokens)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict:
        """Get a single item."""
        tokens = self.encoded[idx].copy()

        # Data augmentation: shuffle interaction order
        if self.augment:
            tokens = self._augment_sequence(tokens)

        return {
            "input_ids": tokens,
            "phenotype": self.circuits[idx].get("phenotype", "unknown"),
        }

    def _augment_sequence(self, tokens: list[int]) -> list[int]:
        """
        Augment by shuffling interaction order.
        Format: <bos> <phenotype> [src type tgt]* <eos>
        """
        # Find phenotype and interactions
        bos = tokens[0]
        phenotype = tokens[1]
        eos = tokens[-1]

        # Extract triplets (excluding bos, phenotype, eos)
        interaction_tokens = tokens[2:-1]

        # Group into triplets
        triplets = []
        for i in range(0, len(interaction_tokens), 3):
            if i + 2 < len(interaction_tokens):
                triplets.append(interaction_tokens[i : i + 3])

        # Shuffle triplets
        random.shuffle(triplets)

        # Reconstruct sequence
        result = [bos, phenotype]
        for triplet in triplets:
            result.extend(triplet)
        result.append(eos)

        return result

    def get_phenotype_distribution(self) -> dict[str, int]:
        """Get distribution of phenotypes in the dataset."""
        dist = {}
        for circuit in self.circuits:
            phenotype = circuit.get("phenotype", "unknown")
            dist[phenotype] = dist.get(phenotype, 0) + 1
        return dist


class GRNDataLoader:
    """DataLoader for batching GRN circuits."""

    def __init__(
        self,
        dataset: GRNDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Initialize dataloader.

        Args:
            dataset: GRNDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict]:
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            if end > len(indices) and self.drop_last:
                break

            batch_indices = indices[start:end]
            batch_items = [self.dataset[i] for i in batch_indices]

            # Collate batch
            yield self._collate(batch_items)

    def _collate(self, items: list[dict]) -> dict:
        """Collate items into a batch."""
        # Get sequences and pad
        sequences = [item["input_ids"] for item in items]
        max_len = max(len(s) for s in sequences)

        # Pad sequences
        padded = []
        attention_mask = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded.append(seq + [self.dataset.tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)

        if HAS_MLX:
            return {
                "input_ids": mx.array(padded),
                "attention_mask": mx.array(attention_mask),
                "phenotypes": [item["phenotype"] for item in items],
            }
        else:
            return {
                "input_ids": np.array(padded),
                "attention_mask": np.array(attention_mask),
                "phenotypes": [item["phenotype"] for item in items],
            }


def load_circuits(path: str | Path) -> list[dict]:
    """Load circuits from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("circuits", data)


def create_data_splits(
    circuits: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Create stratified train/val/test splits.

    Args:
        circuits: List of circuit dictionaries
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_circuits, val_circuits, test_circuits)
    """
    random.seed(seed)

    # Group by phenotype
    by_phenotype: dict[str, list[dict]] = {}
    for circuit in circuits:
        phenotype = circuit.get("phenotype", "unknown")
        if phenotype not in by_phenotype:
            by_phenotype[phenotype] = []
        by_phenotype[phenotype].append(circuit)

    train, val, test = [], [], []

    # Stratified split for each phenotype
    for phenotype, phenotype_circuits in by_phenotype.items():
        random.shuffle(phenotype_circuits)
        n = len(phenotype_circuits)

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(phenotype_circuits[:n_train])
        val.extend(phenotype_circuits[n_train : n_train + n_val])
        test.extend(phenotype_circuits[n_train + n_val :])

    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_splits(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    output_dir: str | Path,
):
    """Save data splits to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump({"circuits": data}, f, indent=2)


def expand_dataset_with_variants(
    circuits: list[dict],
    tokenizer: GRNTokenizer,
    num_variants: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Expand dataset by creating gene-swapped variants of each circuit.
    This creates more training data while preserving topology.

    Args:
        circuits: Original circuits
        tokenizer: Tokenizer with gene vocabulary
        num_variants: Number of variants per circuit
        seed: Random seed

    Returns:
        Expanded list of circuits
    """
    random.seed(seed)
    expanded = list(circuits)  # Include originals

    # Get genes for swapping (exclude generic geneA, geneB, etc.)
    swap_genes = [g for g in tokenizer.genes if not g.startswith("gene")]

    for circuit in circuits:
        for _ in range(num_variants):
            # Create gene mapping
            original_genes = set()
            for interaction in circuit["interactions"]:
                original_genes.add(interaction["source"].lower())
                original_genes.add(interaction["target"].lower())

            # Map to random genes
            gene_map = {}
            available = swap_genes.copy()
            random.shuffle(available)

            for i, gene in enumerate(original_genes):
                if i < len(available):
                    gene_map[gene] = available[i]
                else:
                    gene_map[gene] = gene  # Keep original if we run out

            # Create variant
            variant = {
                "id": f"{circuit['id']}_var{_}",
                "phenotype": circuit["phenotype"],
                "interactions": [
                    {
                        "source": gene_map.get(
                            inter["source"].lower(), inter["source"]
                        ),
                        "target": gene_map.get(
                            inter["target"].lower(), inter["target"]
                        ),
                        "type": inter["type"],
                    }
                    for inter in circuit["interactions"]
                ],
                "source": "augmented",
                "reference": f"Variant of {circuit.get('reference', circuit['id'])}",
            }
            expanded.append(variant)

    return expanded
