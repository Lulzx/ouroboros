#!/usr/bin/env python3
"""Evaluate trained GRN generation model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from src.data.tokenizer import GRNTokenizer
from src.data.dataset import load_circuits
from src.model.transformer import GRNTransformer, ModelArgs
from src.evaluation.metrics import evaluate_model
from src.evaluation.analyze import (
    analyze_generated_circuits,
    create_confusion_matrix,
    sample_circuits,
)
from src.utils.logging import setup_logger


def load_model(checkpoint_path: Path, tokenizer: GRNTokenizer) -> GRNTransformer:
    """Load model from checkpoint."""
    state_path = checkpoint_path.with_suffix(".json")
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        model_args = ModelArgs(**state["model_args"])
    else:
        model_args = ModelArgs(vocab_size=tokenizer.vocab_size)

    model = GRNTransformer(model_args)
    weights = mx.load(str(checkpoint_path))
    model.load_weights(list(weights.items()))

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRN generation model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Processed data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples per phenotype",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip oracle (simulator) evaluation",
    )
    parser.add_argument(
        "--sample-circuits",
        type=int,
        default=5,
        help="Number of sample circuits to display per phenotype",
    )
    args = parser.parse_args()

    logger = setup_logger("grn.evaluate")

    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer_path = data_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = GRNTokenizer.load(tokenizer_path)
    else:
        tokenizer = GRNTokenizer()

    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, tokenizer)
    logger.info(f"Model parameters: {model.num_parameters:,}")

    # Load training data for novelty computation
    train_path = data_dir / "splits" / "train.json"
    training_circuits = None
    if train_path.exists():
        training_circuits = load_circuits(train_path)
        logger.info(f"Loaded {len(training_circuits)} training circuits for novelty check")

    # Run evaluation
    logger.info(f"Running evaluation with {args.num_samples} samples per phenotype...")

    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        training_circuits=training_circuits,
        num_samples=args.num_samples,
        max_length=64,
        temperature=args.temperature,
        top_p=args.top_p,
        run_oracle=not args.skip_oracle,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 Self-Consistency:")
    print(f"  Overall Accuracy: {results['self_consistency']['accuracy']:.3f}")
    for name, acc in results['self_consistency']['per_phenotype'].items():
        print(f"    {name}: {acc:.3f}")

    if 'oracle_consistency' in results:
        print("\n🔬 Oracle Consistency:")
        print(f"  Overall Accuracy: {results['oracle_consistency']['accuracy']:.3f}")
        for name, acc in results['oracle_consistency']['per_phenotype'].items():
            print(f"    {name}: {acc:.3f}")

    print("\n🎯 Diversity:")
    print(f"  Mean Edit Distance: {results['diversity']['mean_edit_distance']:.2f}")
    print(f"  Mean Unique Ratio: {results['diversity']['mean_unique_ratio']:.3f}")

    if 'novelty' in results:
        print("\n🆕 Novelty:")
        print(f"  Exact Novelty: {results['novelty']['exact_novelty']:.3f}")
        print(f"  Approx Novelty: {results['novelty']['approx_novelty']:.3f}")

    print("\n✅ Validity:")
    print(f"  Valid Circuits: {results['validity']['validity']:.3f}")
    print(f"  Has Interactions: {results['validity']['has_interactions_ratio']:.3f}")

    # Analyze generated circuits
    print("\n" + "=" * 60)
    print("CIRCUIT ANALYSIS")
    print("=" * 60)

    analysis = analyze_generated_circuits(
        model, tokenizer, num_samples=50,
        temperature=args.temperature, top_p=args.top_p
    )

    for phenotype, stats in analysis.items():
        print(f"\n{phenotype}:")
        print(f"  Avg genes: {stats['avg_num_genes']:.1f} ± {stats['std_num_genes']:.1f}")
        print(f"  Avg interactions: {stats['avg_num_interactions']:.1f} ± {stats['std_num_interactions']:.1f}")
        if stats['motif_patterns']:
            print(f"  Top motifs: {list(stats['motif_patterns'].keys())[:3]}")

    # Create confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (Self-Classification)")
    print("=" * 60)

    conf_matrix = create_confusion_matrix(
        model, tokenizer, use_oracle=False, num_samples=30,
        temperature=args.temperature, top_p=args.top_p
    )

    phenotypes = conf_matrix['phenotypes']
    print("\n" + " " * 15 + "  ".join(f"{p[:8]:>8}" for p in phenotypes))
    for true_name in phenotypes:
        row = [conf_matrix['normalized'][true_name].get(pred, 0) for pred in phenotypes]
        row_str = "  ".join(f"{v:8.2f}" for v in row)
        print(f"{true_name[:14]:>14} {row_str}")

    # Sample circuits
    if args.sample_circuits > 0:
        print("\n" + "=" * 60)
        print("SAMPLE GENERATED CIRCUITS")
        print("=" * 60)

        for phenotype in ["oscillator", "toggle_switch", "stable"]:
            print(f"\n{phenotype.upper()}:")
            try:
                samples = sample_circuits(
                    model, tokenizer, phenotype,
                    num_samples=args.sample_circuits,
                    temperature=args.temperature, top_p=args.top_p
                )
                for i, circuit in enumerate(samples):
                    interactions = circuit.get('interactions', [])
                    if interactions:
                        edges = []
                        for inter in interactions[:5]:  # Limit display
                            src, tgt, typ = inter['source'], inter['target'], inter['type']
                            symbol = '→' if typ == 'activates' else '⊣'
                            edges.append(f"{src}{symbol}{tgt}")
                        print(f"  {i+1}. {', '.join(edges)}")
            except Exception as e:
                print(f"  Error: {e}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "checkpoint": str(args.checkpoint),
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "results": results,
            "analysis": analysis,
            "confusion_matrix": conf_matrix,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
