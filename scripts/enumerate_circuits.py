#!/usr/bin/env python3
"""
Exhaustive enumeration of all small circuits to find verified working patterns.

This implements the first step of a principled approach:
1. Enumerate ALL possible 2, 3, 4-gene circuit topologies
2. Classify each with the fixed oracle
3. Discover the exact patterns that GUARANTEE each phenotype
4. Build a verified circuit database for constrained generation

Mathematical foundation:
- For n genes, each directed edge can be: absent (0), activates (1), inhibits (2)
- Total possible edges: n^2 (including self-loops)
- Total topologies: 3^(n^2)
- n=2: 3^4 = 81 circuits
- n=3: 3^9 = 19,683 circuits
- n=4: 3^16 = 43,046,721 circuits (we'll sample this)
"""

import argparse
import json
import itertools
from pathlib import Path
from collections import defaultdict
from typing import Iterator
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier


def enumerate_topologies(n_genes: int) -> Iterator[list[tuple[int, int, int]]]:
    """
    Enumerate all possible circuit topologies with n genes.

    Yields lists of (source, target, edge_type) where:
    - source, target in [0, n_genes)
    - edge_type: 0=none, 1=activates, 2=inhibits
    """
    gene_names = [f"g{i}" for i in range(n_genes)]
    n_edges = n_genes * n_genes

    # Generate all combinations of edge types
    for edge_types in itertools.product([0, 1, 2], repeat=n_edges):
        edges = []
        edge_idx = 0
        for src in range(n_genes):
            for tgt in range(n_genes):
                if edge_types[edge_idx] != 0:
                    edges.append((src, tgt, edge_types[edge_idx]))
                edge_idx += 1

        # Skip empty circuits
        if edges:
            yield edges


def edges_to_circuit(edges: list[tuple[int, int, int]], n_genes: int) -> dict:
    """Convert edge list to circuit dictionary."""
    gene_names = [f"g{i}" for i in range(n_genes)]
    interactions = []

    for src, tgt, etype in edges:
        interactions.append({
            "source": gene_names[src],
            "target": gene_names[tgt],
            "type": "activates" if etype == 1 else "inhibits"
        })

    return {"interactions": interactions}


def edges_to_canonical(edges: list[tuple[int, int, int]], n_genes: int) -> str:
    """
    Convert edges to a canonical string representation.
    This allows us to identify unique topological patterns.
    """
    # Sort edges for canonical form
    sorted_edges = sorted(edges)
    return str(sorted_edges)


def analyze_topology(edges: list[tuple[int, int, int]], n_genes: int) -> dict:
    """
    Analyze topological properties of a circuit.

    Returns features that determine dynamical behavior:
    - Has cycle (and cycle properties)
    - Has mutual inhibition
    - Has IFFL pattern
    - Has cascade
    - Has autoregulation
    """
    features = {
        "n_genes": n_genes,
        "n_edges": len(edges),
        "has_self_activation": False,
        "has_self_inhibition": False,
        "has_mutual_inhibition": False,
        "has_activation_cascade": False,
        "has_inhibition_cycle": False,
        "has_iffl": False,
        "cycle_lengths": [],
        "inhibition_cycle_odd": False,
    }

    # Build adjacency
    activators = defaultdict(set)
    inhibitors = defaultdict(set)
    all_edges = defaultdict(set)  # For cycle detection

    for src, tgt, etype in edges:
        if etype == 1:
            activators[tgt].add(src)
        else:
            inhibitors[tgt].add(src)
        all_edges[src].add((tgt, etype))

    # Self-regulation
    for src, tgt, etype in edges:
        if src == tgt:
            if etype == 1:
                features["has_self_activation"] = True
            else:
                features["has_self_inhibition"] = True

    # Mutual inhibition: A⊣B and B⊣A
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            if i in inhibitors[j] and j in inhibitors[i]:
                features["has_mutual_inhibition"] = True

    # Find cycles using DFS
    def find_cycles(start: int, current: int, path: list, visited: set, inhibitions: int):
        for next_node, etype in all_edges[current]:
            new_inhib = inhibitions + (1 if etype == 2 else 0)
            if next_node == start and len(path) > 0:
                # Found cycle
                cycle_len = len(path) + 1
                features["cycle_lengths"].append(cycle_len)
                if new_inhib > 0:
                    features["has_inhibition_cycle"] = True
                    if new_inhib % 2 == 1:
                        features["inhibition_cycle_odd"] = True
            elif next_node not in visited:
                find_cycles(start, next_node, path + [next_node], visited | {next_node}, new_inhib)

    for start in range(n_genes):
        find_cycles(start, start, [], {start}, 0)

    # IFFL pattern: A→B, A→C, B⊣C
    for a in range(n_genes):
        targets_of_a = set()
        for tgt, etype in all_edges[a]:
            if etype == 1:  # A activates
                targets_of_a.add(tgt)
        for b in targets_of_a:
            for c in targets_of_a:
                if b != c and b in inhibitors[c]:
                    features["has_iffl"] = True

    # Activation cascade: A→B→C (chain of activations, length >= 2)
    def find_activation_chain(start: int, length: int, visited: set):
        if length >= 2:
            return True
        for tgt, etype in all_edges[start]:
            if etype == 1 and tgt not in visited:
                if find_activation_chain(tgt, length + 1, visited | {tgt}):
                    return True
        return False

    for start in range(n_genes):
        if find_activation_chain(start, 0, {start}):
            features["has_activation_cascade"] = True
            break

    return features


def main():
    parser = argparse.ArgumentParser(description="Enumerate and classify all small circuits")
    parser.add_argument("--max-genes", type=int, default=4, help="Maximum number of genes")
    parser.add_argument("--output", type=str, default="data/verified_circuits.json")
    parser.add_argument("--sample-4gene", type=int, default=100000,
                       help="Number of 4-gene circuits to sample (full enumeration too large)")
    args = parser.parse_args()

    classifier = BehaviorClassifier(
        num_initial_conditions=20,
        max_steps=200,
        rule="constitutive"
    )

    results = {
        "phenotype_patterns": defaultdict(list),
        "verified_circuits": defaultdict(list),
        "statistics": {},
        "feature_analysis": defaultdict(lambda: defaultdict(int))
    }

    total_tested = 0

    for n_genes in range(2, args.max_genes + 1):
        print(f"\n{'='*60}")
        print(f"Enumerating {n_genes}-gene circuits...")
        print(f"{'='*60}")

        if n_genes <= 3:
            # Full enumeration
            topologies = list(enumerate_topologies(n_genes))
            print(f"Total topologies: {len(topologies)}")
        else:
            # Random sampling for larger circuits
            import random
            random.seed(42)
            all_topos = list(enumerate_topologies(n_genes))
            topologies = random.sample(all_topos, min(args.sample_4gene, len(all_topos)))
            print(f"Sampled {len(topologies)} of {len(all_topos)} topologies")

        phenotype_counts = defaultdict(int)

        for i, edges in enumerate(topologies):
            if i % 5000 == 0 and i > 0:
                print(f"  Processed {i}/{len(topologies)}...")

            circuit = edges_to_circuit(edges, n_genes)
            network = BooleanNetwork.from_circuit(circuit)
            phenotype, details = classifier.classify(network, seed=42)

            phenotype_counts[phenotype] += 1
            total_tested += 1

            # Analyze topology features
            features = analyze_topology(edges, n_genes)

            # Store verified circuit
            canonical = edges_to_canonical(edges, n_genes)
            results["verified_circuits"][phenotype].append({
                "edges": edges,
                "n_genes": n_genes,
                "canonical": canonical,
                "features": features,
                "details": details.get("reason", "")
            })

            # Track feature correlations with phenotype
            for feat_name, feat_val in features.items():
                if isinstance(feat_val, bool) and feat_val:
                    results["feature_analysis"][phenotype][feat_name] += 1

        print(f"\nResults for {n_genes}-gene circuits:")
        for pheno, count in sorted(phenotype_counts.items()):
            pct = 100 * count / len(topologies)
            print(f"  {pheno}: {count} ({pct:.1f}%)")

        results["statistics"][f"{n_genes}_genes"] = dict(phenotype_counts)

    # Analyze what features predict each phenotype
    print(f"\n{'='*60}")
    print("FEATURE ANALYSIS BY PHENOTYPE")
    print(f"{'='*60}")

    for phenotype in classifier.PHENOTYPES:
        n_circuits = len(results["verified_circuits"][phenotype])
        if n_circuits == 0:
            continue

        print(f"\n{phenotype.upper()} ({n_circuits} circuits):")

        # Calculate feature frequencies
        feat_freqs = {}
        for feat_name in ["has_mutual_inhibition", "has_inhibition_cycle",
                          "inhibition_cycle_odd", "has_iffl", "has_activation_cascade",
                          "has_self_activation", "has_self_inhibition"]:
            count = results["feature_analysis"][phenotype].get(feat_name, 0)
            freq = count / n_circuits
            feat_freqs[feat_name] = freq
            if freq > 0.3:  # Only show significant features
                print(f"  {feat_name}: {freq:.1%}")

    # Find MINIMAL circuits for each phenotype (most informative)
    print(f"\n{'='*60}")
    print("MINIMAL WORKING CIRCUITS PER PHENOTYPE")
    print(f"{'='*60}")

    minimal_circuits = {}
    for phenotype in classifier.PHENOTYPES:
        circuits = results["verified_circuits"][phenotype]
        if not circuits:
            continue

        # Sort by number of edges, then genes
        circuits_sorted = sorted(circuits, key=lambda x: (len(x["edges"]), x["n_genes"]))
        minimal = circuits_sorted[:5]  # Top 5 minimal
        minimal_circuits[phenotype] = minimal

        print(f"\n{phenotype.upper()}:")
        for c in minimal:
            edges_str = []
            for src, tgt, etype in c["edges"]:
                sym = "→" if etype == 1 else "⊣"
                edges_str.append(f"g{src}{sym}g{tgt}")
            print(f"  {', '.join(edges_str)}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert defaultdicts for JSON serialization
    results["phenotype_patterns"] = dict(results["phenotype_patterns"])
    results["verified_circuits"] = {k: v for k, v in results["verified_circuits"].items()}
    results["feature_analysis"] = {k: dict(v) for k, v in results["feature_analysis"].items()}
    results["minimal_circuits"] = minimal_circuits

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Total circuits tested: {total_tested}")
    print(f"Results saved to {output_path}")

    # Summary statistics
    print(f"\nCIRCUITS PER PHENOTYPE (all sizes):")
    for phenotype in classifier.PHENOTYPES:
        n = len(results["verified_circuits"][phenotype])
        print(f"  {phenotype}: {n}")


if __name__ == "__main__":
    main()
