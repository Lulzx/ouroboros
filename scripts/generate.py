#!/usr/bin/env python3
"""Generate circuits from trained model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from src.data.tokenizer import GRNTokenizer
from src.model.transformer import GRNTransformer, ModelArgs
from src.model.generation import generate
from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier


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


def format_circuit(circuit: dict, show_simulation: bool = False) -> str:
    """Format circuit for display."""
    lines = []
    lines.append(f"Phenotype: {circuit.get('phenotype', 'unknown')}")
    lines.append("Interactions:")

    for inter in circuit.get('interactions', []):
        src = inter['source']
        tgt = inter['target']
        typ = inter['type']
        symbol = '→' if typ == 'activates' else '⊣'
        lines.append(f"  {src} {symbol} {tgt} ({typ})")

    if show_simulation and circuit.get('interactions'):
        try:
            network = BooleanNetwork.from_circuit(circuit)
            classifier = BehaviorClassifier()
            predicted, details = classifier.classify(network)
            lines.append(f"Simulated behavior: {predicted}")
            if 'reason' in details:
                lines.append(f"  Reason: {details['reason']}")
        except Exception as e:
            lines.append(f"Simulation error: {e}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate GRN circuits")
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
        help="Processed data directory (for tokenizer)",
    )
    parser.add_argument(
        "--phenotype",
        type=str,
        default="oscillator",
        choices=["oscillator", "toggle_switch", "adaptation",
                 "pulse_generator", "amplifier", "stable"],
        help="Phenotype to generate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of circuits to generate",
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
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run boolean simulation on generated circuits",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (JSON)",
    )
    parser.add_argument(
        "--all-phenotypes",
        action="store_true",
        help="Generate for all phenotypes",
    )
    args = parser.parse_args()

    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer_path = data_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = GRNTokenizer.load(tokenizer_path)
    else:
        tokenizer = GRNTokenizer()

    # Load model
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, tokenizer)
    print(f"Model parameters: {model.num_parameters:,}")

    # Determine phenotypes to generate
    if args.all_phenotypes:
        phenotypes = ["oscillator", "toggle_switch", "adaptation",
                     "pulse_generator", "amplifier", "stable"]
    else:
        phenotypes = [args.phenotype]

    all_circuits = []

    for phenotype in phenotypes:
        print(f"\n{'='*60}")
        print(f"Generating {args.num_samples} {phenotype} circuits")
        print('='*60)

        phenotype_token = tokenizer.phenotype_to_token(phenotype)
        phenotype_id = tokenizer.token_to_id[phenotype_token]

        # Create prompts
        prompts = mx.array(
            [[tokenizer.bos_token_id, phenotype_id]] * args.num_samples
        )

        # Generate
        generated = generate(
            model,
            prompts,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        mx.eval(generated)

        # Decode and display
        for i in range(args.num_samples):
            tokens = generated[i].tolist()
            circuit = tokenizer.decode(tokens)
            circuit['generated_phenotype'] = phenotype

            print(f"\n--- Circuit {i+1} ---")
            print(format_circuit(circuit, show_simulation=args.simulate))

            all_circuits.append(circuit)

    # Save to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'circuits': all_circuits,
                'settings': {
                    'checkpoint': str(args.checkpoint),
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'top_k': args.top_k,
                }
            }, f, indent=2)

        print(f"\nSaved {len(all_circuits)} circuits to {output_path}")


if __name__ == "__main__":
    main()
