#!/usr/bin/env python3
"""Preprocess data: load circuits, create tokenizer, and split data."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import GRNTokenizer
from src.data.dataset import (
    load_circuits,
    create_data_splits,
    save_splits,
    expand_dataset_with_variants,
)


def main():
    parser = argparse.ArgumentParser(description="Preprocess GRN data")
    parser.add_argument(
        "--circuits",
        type=str,
        default="data/synthetic/classic_circuits.json",
        help="Path to circuits JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=10,
        help="Number of variants per circuit (0 to disable)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training data ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation data ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load circuits
    print(f"Loading circuits from {args.circuits}...")
    circuits = load_circuits(args.circuits)
    print(f"Loaded {len(circuits)} circuits")

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = GRNTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Expand dataset with variants
    if args.expand > 0:
        print(f"Expanding dataset with {args.expand} variants per circuit...")
        circuits = expand_dataset_with_variants(
            circuits,
            tokenizer,
            num_variants=args.expand,
            seed=args.seed,
        )
        print(f"Expanded to {len(circuits)} circuits")

    # Create splits
    print("Creating train/val/test splits...")
    train, val, test = create_data_splits(
        circuits,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        seed=args.seed,
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Save splits
    splits_dir = output_dir / "splits"
    save_splits(train, val, test, splits_dir)
    print(f"Saved splits to {splits_dir}")

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Save all circuits
    all_circuits_path = output_dir / "circuits.json"
    with open(all_circuits_path, "w") as f:
        json.dump({"circuits": circuits}, f, indent=2)
    print(f"Saved all circuits to {all_circuits_path}")

    # Print phenotype distribution
    print("\nPhenotype distribution:")
    from collections import Counter
    dist = Counter(c["phenotype"] for c in circuits)
    for phenotype, count in sorted(dist.items()):
        print(f"  {phenotype}: {count}")

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
