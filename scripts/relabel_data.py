#!/usr/bin/env python3
"""Re-label training data using the fixed oracle (BehaviorClassifier with constitutive rule)."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier


def main():
    parser = argparse.ArgumentParser(description="Re-label circuits using fixed oracle")
    parser.add_argument(
        "--input",
        type=str,
        default="data/synthetic/classic_circuits.json",
        help="Input circuits JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/classic_circuits_relabeled.json",
        help="Output circuits JSON file with corrected labels",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="constitutive",
        help="Boolean update rule (default: constitutive)",
    )
    parser.add_argument(
        "--num-initial-conditions",
        type=int,
        default=20,
        help="Number of initial conditions for simulation",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum simulation steps",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    args = parser.parse_args()

    # Load circuits
    print(f"Loading circuits from {args.input}...")
    with open(args.input, "r") as f:
        data = json.load(f)
    circuits = data["circuits"]
    print(f"Loaded {len(circuits)} circuits")

    # Create classifier with fixed oracle settings
    classifier = BehaviorClassifier(
        num_initial_conditions=args.num_initial_conditions,
        max_steps=args.max_steps,
        rule=args.rule,
    )

    # Re-label each circuit
    relabeled_circuits = []
    changes = []
    matches = 0

    print(f"\nRe-labeling using {args.rule} rule...")
    for circuit in circuits:
        circuit_id = circuit["id"]
        original_phenotype = circuit["phenotype"]

        # Create boolean network from circuit
        network = BooleanNetwork.from_circuit(circuit)

        # Classify using fixed oracle
        predicted_phenotype, details = classifier.classify(network, seed=42)

        # Create relabeled circuit
        relabeled = circuit.copy()
        relabeled["phenotype"] = predicted_phenotype
        relabeled["original_phenotype"] = original_phenotype
        relabeled_circuits.append(relabeled)

        if original_phenotype != predicted_phenotype:
            changes.append({
                "id": circuit_id,
                "original": original_phenotype,
                "predicted": predicted_phenotype,
                "reason": details.get("reason", ""),
            })
            if args.verbose:
                print(f"  {circuit_id}: {original_phenotype} -> {predicted_phenotype} ({details.get('reason', '')})")
        else:
            matches += 1

    # Save relabeled circuits
    output_data = {"circuits": relabeled_circuits}
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved relabeled circuits to {args.output}")

    # Print summary
    total = len(circuits)
    changed = len(changes)
    print(f"\n{'='*60}")
    print(f"Re-labeling Summary")
    print(f"{'='*60}")
    print(f"Total circuits: {total}")
    print(f"Matching labels: {matches} ({100*matches/total:.1f}%)")
    print(f"Changed labels: {changed} ({100*changed/total:.1f}%)")

    # Group changes by type
    print(f"\nLabel changes by transition:")
    transitions = {}
    for c in changes:
        key = f"{c['original']} -> {c['predicted']}"
        if key not in transitions:
            transitions[key] = []
        transitions[key].append(c['id'])

    for transition, ids in sorted(transitions.items(), key=lambda x: -len(x[1])):
        print(f"  {transition}: {len(ids)}")
        if args.verbose:
            for cid in ids[:3]:
                print(f"    - {cid}")
            if len(ids) > 3:
                print(f"    ... and {len(ids)-3} more")

    # Print new phenotype distribution
    print(f"\nNew phenotype distribution:")
    from collections import Counter
    dist = Counter(c["phenotype"] for c in relabeled_circuits)
    for phenotype, count in sorted(dist.items()):
        print(f"  {phenotype}: {count}")


if __name__ == "__main__":
    main()
