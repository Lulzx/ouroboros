#!/usr/bin/env python3
"""
Benchmark: Classical Optimization vs Ouroboros Amortized Generation.

This script directly compares:
1. Classical methods (evolutionary, hill climbing, random search)
2. Ouroboros methods (template-based, neural)

Metrics:
- Time to generate N valid circuits
- Success rate per target phenotype
- Diversity of solutions
- Cost per circuit

Usage:
    python scripts/benchmark_optimization.py --num-targets 10 --circuits-per-target 10
"""

import argparse
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier
from src.optimization.evolutionary import (
    EvolutionaryOptimizer,
    RandomSearchOptimizer,
    HillClimbingOptimizer,
)

# For topology classifier benchmark
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    method: str
    phenotype: str
    num_circuits_requested: int
    num_circuits_produced: int
    success_rate: float
    total_time: float
    time_per_circuit: float
    evaluations: int
    evaluations_per_circuit: float
    unique_topologies: int
    diversity_score: float


def load_gene_pool(tokenizer_path: str = "data/processed_relabeled/tokenizer.json") -> list[str]:
    """Load gene vocabulary from tokenizer."""
    with open(tokenizer_path) as f:
        tokenizer_data = json.load(f)

    # Handle both formats: {"genes": [...]} or {"token_to_id": {...}}
    if "genes" in tokenizer_data:
        return tokenizer_data["genes"]

    genes = []
    for token in tokenizer_data.get("token_to_id", {}).keys():
        if not token.startswith("<") and token not in ["activates", "inhibits"]:
            genes.append(token)

    return genes


def load_verified_db(path: str = "data/verified_circuits.json") -> dict:
    """Load verified circuit database."""
    with open(path) as f:
        return json.load(f)


def compute_topology_hash(circuit: dict) -> str:
    """Compute a hash of circuit topology (ignoring gene names)."""
    edges = []
    genes = {}
    gene_counter = 0

    for interaction in circuit.get("interactions", []):
        src = interaction["source"].lower()
        tgt = interaction["target"].lower()
        itype = interaction["type"]

        if src not in genes:
            genes[src] = gene_counter
            gene_counter += 1
        if tgt not in genes:
            genes[tgt] = gene_counter
            gene_counter += 1

        edges.append((genes[src], genes[tgt], itype))

    edges.sort()
    return str(edges)


def compute_diversity(circuits: list[dict]) -> tuple[int, float]:
    """
    Compute diversity metrics.

    Returns:
        (num_unique_topologies, diversity_score)
    """
    if not circuits:
        return 0, 0.0

    topologies = set()
    for circuit in circuits:
        h = compute_topology_hash(circuit)
        topologies.add(h)

    unique = len(topologies)
    diversity = unique / len(circuits) if circuits else 0.0

    return unique, diversity


class TemplateBenchmark:
    """Benchmark for template-based generation (Ouroboros)."""

    def __init__(self, verified_db: dict, gene_pool: list[str], seed: int = 42):
        self.db = verified_db
        self.gene_pool = gene_pool
        self.rng = random.Random(seed)
        self.classifier = BehaviorClassifier(
            num_initial_conditions=20,
            max_steps=200,
            rule="constitutive"
        )
        self.circuits_by_phenotype = verified_db.get("verified_circuits", {})

    def generate(self, phenotype: str, num_circuits: int) -> tuple[list[dict], float, int]:
        """
        Generate circuits via template sampling.

        Returns:
            (circuits, time, evaluations)
        """
        start = time.time()
        circuits = []
        evaluations = 0

        templates = self.circuits_by_phenotype.get(phenotype, [])
        if not templates:
            return [], time.time() - start, 0

        for _ in range(num_circuits):
            template = self.rng.choice(templates)
            edges = template["edges"]
            n_genes = template["n_genes"]

            gene_names = self.rng.sample(self.gene_pool, min(n_genes, len(self.gene_pool)))

            interactions = []
            for src, tgt, etype in edges:
                if src < len(gene_names) and tgt < len(gene_names):
                    interactions.append({
                        "source": gene_names[src],
                        "target": gene_names[tgt],
                        "type": "activates" if etype == 1 else "inhibits"
                    })

            circuit = {"interactions": interactions, "phenotype": phenotype}

            # Verify (optional, for fair comparison)
            network = BooleanNetwork.from_circuit(circuit)
            predicted, _ = self.classifier.classify(network)
            evaluations += 1

            if predicted == phenotype:
                circuits.append(circuit)

        elapsed = time.time() - start
        return circuits, elapsed, evaluations


class ClassicalBenchmark:
    """Benchmark wrapper for classical optimizers."""

    def __init__(self, gene_pool: list[str], method: str = "evolutionary", seed: int = 42):
        self.gene_pool = gene_pool
        self.method = method
        self.seed = seed

    def generate(
        self,
        phenotype: str,
        num_circuits: int,
        max_evals_per_circuit: int = 1000,
    ) -> tuple[list[dict], float, int]:
        """
        Generate circuits via classical optimization.

        Each circuit requires a fresh optimization run.

        Returns:
            (circuits, time, evaluations)
        """
        start = time.time()
        circuits = []
        total_evaluations = 0

        for i in range(num_circuits):
            if self.method == "evolutionary":
                optimizer = EvolutionaryOptimizer(
                    gene_pool=self.gene_pool,
                    num_genes=3,
                    population_size=30,
                    offspring_size=30,
                    seed=self.seed + i,
                )
            elif self.method == "hill_climbing":
                optimizer = HillClimbingOptimizer(
                    gene_pool=self.gene_pool,
                    num_genes=3,
                    restarts=10,
                    steps_per_restart=50,
                    seed=self.seed + i,
                )
            else:  # random
                optimizer = RandomSearchOptimizer(
                    gene_pool=self.gene_pool,
                    num_genes=3,
                    seed=self.seed + i,
                )

            result = optimizer.optimize(
                target_phenotype=phenotype,
                max_evaluations=max_evals_per_circuit,
                verbose=False,
            )

            total_evaluations += result.evaluations

            if result.success and result.best_circuit:
                circuits.append(result.best_circuit.to_dict())

        elapsed = time.time() - start
        return circuits, elapsed, total_evaluations


# Import feature extraction from train_topology_final for 92%+ accuracy
from scripts.train_topology_final import (
    extract_combined_features as extract_topology_features,
    PHENOTYPE_TO_ID,
    ID_TO_PHENOTYPE,
)


class TopologyClassifier(nn.Module):
    """Residual MLP classifier for topology features (matches train_topology_final.py)."""

    def __init__(self, input_dim: int = 48, hidden_dim: int = 192, n_layers: int = 4):
        super().__init__()
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.blocks = []
        for _ in range(n_layers):
            self.blocks.append({
                'linear1': nn.Linear(hidden_dim, hidden_dim),
                'linear2': nn.Linear(hidden_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
            })

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )

    def __call__(self, x):
        h = nn.gelu(self.input_norm(self.input_embed(x)))
        for block in self.blocks:
            residual = h
            h = nn.gelu(block['linear1'](h))
            h = block['norm'](block['linear2'](h) + residual)
        return self.head(h)


class TopologyBenchmark:
    """Benchmark for topology-based generation with neural classifier."""

    def __init__(self, verified_db: dict, gene_pool: list[str], seed: int = 42):
        self.db = verified_db
        self.gene_pool = gene_pool
        self.rng = random.Random(seed)
        self.circuits_by_phenotype = verified_db.get("verified_circuits", {})

        # Train a quick classifier on the verified circuits
        if HAS_MLX:
            self.model = self._train_classifier()
        else:
            self.model = None

    def _train_classifier(self) -> TopologyClassifier:
        """Train classifier on verified circuits (full training for 92%+ accuracy)."""
        features_list = []
        labels = []

        for phenotype, circuits in self.circuits_by_phenotype.items():
            if phenotype not in PHENOTYPE_TO_ID:
                continue
            for circuit in circuits:  # Use all data
                feat = extract_topology_features(circuit)
                features_list.append(feat)
                labels.append(PHENOTYPE_TO_ID[phenotype])

        print(f"    Training topology classifier on {len(features_list)} circuits...")

        model = TopologyClassifier(input_dim=48, hidden_dim=192, n_layers=4)
        optimizer = optim.Adam(learning_rate=5e-4)

        def focal_loss(logits, labels, gamma=2.0):
            ce = nn.losses.cross_entropy(logits, labels, reduction='none')
            pt = mx.exp(-ce)
            return mx.mean(((1 - pt) ** gamma) * ce)

        def loss_fn(model, x, y):
            logits = model(x)
            return focal_loss(logits, y)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        # Full training (5000 steps for 90%+ accuracy)
        batch_size = 96
        for step in range(5000):
            idx = np.random.choice(len(features_list), batch_size)
            batch_x = mx.array(np.stack([features_list[i] for i in idx]))
            batch_y = mx.array([labels[i] for i in idx])
            loss, grads = loss_and_grad(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        return model

    def generate(self, phenotype: str, num_circuits: int) -> tuple[list[dict], float, int]:
        """Generate circuits using neural classifier for verification."""
        if not HAS_MLX or self.model is None:
            return [], 0.0, 0

        start = time.time()
        circuits = []
        evaluations = 0

        templates = self.circuits_by_phenotype.get(phenotype, [])
        if not templates:
            return [], time.time() - start, 0

        target_id = PHENOTYPE_TO_ID.get(phenotype, 0)

        for _ in range(num_circuits):
            template = self.rng.choice(templates)
            edges = template["edges"]
            n_genes = template["n_genes"]

            gene_names = self.rng.sample(self.gene_pool, min(n_genes, len(self.gene_pool)))

            interactions = []
            for src, tgt, etype in edges:
                if src < len(gene_names) and tgt < len(gene_names):
                    interactions.append({
                        "source": gene_names[src],
                        "target": gene_names[tgt],
                        "type": "activates" if etype == 1 else "inhibits"
                    })

            circuit = {
                "interactions": interactions,
                "phenotype": phenotype,
                "edges": edges,
                "n_genes": n_genes,
                "features": template.get("features", {}),
            }

            # Verify with neural classifier
            feat = extract_topology_features(circuit)
            logits = self.model(mx.array(feat[None]))
            pred_id = int(mx.argmax(logits[0]))
            evaluations += 1

            if pred_id == target_id:
                circuits.append(circuit)

        elapsed = time.time() - start
        return circuits, elapsed, evaluations


def run_benchmark(
    phenotypes: list[str],
    num_circuits: int,
    gene_pool: list[str],
    verified_db: dict,
    max_evals_per_circuit: int = 1000,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Run full benchmark across all methods and phenotypes."""

    results = []

    # Methods to benchmark
    methods = {
        "template": TemplateBenchmark(verified_db, gene_pool, seed),
        "topology_nn": TopologyBenchmark(verified_db, gene_pool, seed) if HAS_MLX else None,
        "evolutionary": ClassicalBenchmark(gene_pool, "evolutionary", seed),
        "hill_climbing": ClassicalBenchmark(gene_pool, "hill_climbing", seed),
        "random_search": ClassicalBenchmark(gene_pool, "random", seed),
    }
    # Filter out None methods
    methods = {k: v for k, v in methods.items() if v is not None}

    for phenotype in phenotypes:
        print(f"\n{'='*60}")
        print(f"Phenotype: {phenotype}")
        print(f"{'='*60}")

        for method_name, benchmark in methods.items():
            print(f"\n  {method_name}...")

            if method_name in ("template", "topology_nn"):
                circuits, elapsed, evals = benchmark.generate(phenotype, num_circuits)
            else:
                circuits, elapsed, evals = benchmark.generate(
                    phenotype, num_circuits, max_evals_per_circuit
                )

            unique_topos, diversity = compute_diversity(circuits)

            result = BenchmarkResult(
                method=method_name,
                phenotype=phenotype,
                num_circuits_requested=num_circuits,
                num_circuits_produced=len(circuits),
                success_rate=len(circuits) / num_circuits if num_circuits > 0 else 0,
                total_time=elapsed,
                time_per_circuit=elapsed / num_circuits if num_circuits > 0 else 0,
                evaluations=evals,
                evaluations_per_circuit=evals / num_circuits if num_circuits > 0 else 0,
                unique_topologies=unique_topos,
                diversity_score=diversity,
            )

            results.append(result)

            print(f"    Success rate: {result.success_rate:.1%}")
            print(f"    Time: {elapsed:.3f}s ({result.time_per_circuit*1000:.2f}ms/circuit)")
            print(f"    Evaluations: {evals} ({result.evaluations_per_circuit:.1f}/circuit)")
            print(f"    Diversity: {unique_topos} unique topologies ({diversity:.1%})")

    return results


def summarize_results(results: list[BenchmarkResult]) -> dict:
    """Aggregate results by method."""
    by_method = defaultdict(list)
    for r in results:
        by_method[r.method].append(r)

    summary = {}
    for method, method_results in by_method.items():
        summary[method] = {
            "avg_success_rate": np.mean([r.success_rate for r in method_results]),
            "avg_time_per_circuit_ms": np.mean([r.time_per_circuit * 1000 for r in method_results]),
            "avg_evaluations_per_circuit": np.mean([r.evaluations_per_circuit for r in method_results]),
            "avg_diversity": np.mean([r.diversity_score for r in method_results]),
            "total_time": sum(r.total_time for r in method_results),
            "total_circuits": sum(r.num_circuits_produced for r in method_results),
        }

    return summary


def print_summary_table(summary: dict):
    """Print summary as formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Method':<20} {'Success':<10} {'Time/circuit':<15} {'Evals/circuit':<15} {'Diversity':<10}")
    print("-" * 80)

    for method, stats in summary.items():
        print(f"{method:<20} "
              f"{stats['avg_success_rate']:.1%}       "
              f"{stats['avg_time_per_circuit_ms']:.2f}ms         "
              f"{stats['avg_evaluations_per_circuit']:.1f}            "
              f"{stats['avg_diversity']:.1%}")

    print("-" * 80)

    # Cost comparison
    template_stats = summary.get("template", {})
    evo_stats = summary.get("evolutionary", {})

    if template_stats and evo_stats:
        speedup = evo_stats["avg_time_per_circuit_ms"] / template_stats["avg_time_per_circuit_ms"]
        eval_ratio = evo_stats["avg_evaluations_per_circuit"] / max(template_stats["avg_evaluations_per_circuit"], 1)

        print(f"\nTemplate vs Evolutionary:")
        print(f"  Speedup: {speedup:.0f}x faster")
        print(f"  Evaluation efficiency: {eval_ratio:.0f}x fewer evaluations")


def compute_breakeven(summary: dict, training_cost_seconds: float = 7200) -> dict:
    """
    Compute break-even point for amortization.

    Args:
        training_cost_seconds: One-time training cost in seconds

    Returns:
        Break-even analysis
    """
    template_stats = summary.get("template", {})
    evo_stats = summary.get("evolutionary", {})

    if not template_stats or not evo_stats:
        return {}

    c_opt = evo_stats["avg_time_per_circuit_ms"] / 1000  # seconds
    c_sample = template_stats["avg_time_per_circuit_ms"] / 1000  # seconds
    c_train = training_cost_seconds

    if c_opt <= c_sample:
        return {"breakeven_n": float("inf"), "message": "Template not faster than optimization"}

    breakeven_n = c_train / (c_opt - c_sample)

    return {
        "training_cost_seconds": c_train,
        "cost_per_optimization_seconds": c_opt,
        "cost_per_template_seconds": c_sample,
        "breakeven_n": breakeven_n,
        "message": f"Amortization pays off after {breakeven_n:.0f} circuits"
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimization methods")
    parser.add_argument("--num-circuits", type=int, default=10,
                        help="Circuits to generate per phenotype")
    parser.add_argument("--max-evals", type=int, default=1000,
                        help="Max evaluations per optimization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file")
    parser.add_argument("--phenotypes", type=str, nargs="+",
                        default=["oscillator", "toggle_switch", "adaptation",
                                 "pulse_generator", "amplifier", "stable"],
                        help="Phenotypes to benchmark")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    gene_pool = load_gene_pool()
    verified_db = load_verified_db()

    print(f"Gene pool: {len(gene_pool)} genes")
    print(f"Phenotypes: {args.phenotypes}")
    print(f"Circuits per phenotype: {args.num_circuits}")

    # Run benchmark
    results = run_benchmark(
        phenotypes=args.phenotypes,
        num_circuits=args.num_circuits,
        gene_pool=gene_pool,
        verified_db=verified_db,
        max_evals_per_circuit=args.max_evals,
        seed=args.seed,
    )

    # Summarize
    summary = summarize_results(results)
    print_summary_table(summary)

    # Break-even analysis
    breakeven = compute_breakeven(summary)
    if breakeven:
        print(f"\n{'='*80}")
        print("AMORTIZATION ANALYSIS")
        print("="*80)
        print(f"Training cost (one-time): {breakeven.get('training_cost_seconds', 0):.0f}s")
        print(f"Cost per classical optimization: {breakeven.get('cost_per_optimization_seconds', 0)*1000:.2f}ms")
        print(f"Cost per template generation: {breakeven.get('cost_per_template_seconds', 0)*1000:.2f}ms")
        print(f"\n{breakeven.get('message', '')}")

    # Save results
    output_data = {
        "config": {
            "num_circuits": args.num_circuits,
            "max_evals": args.max_evals,
            "phenotypes": args.phenotypes,
            "seed": args.seed,
        },
        "results": [asdict(r) for r in results],
        "summary": summary,
        "breakeven_analysis": breakeven,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
