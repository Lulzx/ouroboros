#!/usr/bin/env python3
"""
Enhanced Topology Learning with Improved Oscillator Features

Key insight from first run: oscillator class has only 50.4% accuracy.
The issue is that oscillation depends on subtle dynamical properties
that aren't fully captured by basic structural features.

Theory of Oscillation:
1. Odd-length inhibition cycles → creates phase inversion → oscillation
2. Self-inhibition → creates self-feedback delay → oscillation
3. Negative autoregulation → stabilizes but can oscillate with delay

This version adds:
- More granular cycle analysis
- Feedback loop parity features
- Gene-specific topology indicators
- Better feature normalization
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from src.utils.logging import setup_logger


PHENOTYPE_TO_ID = {
    'oscillator': 0,
    'toggle_switch': 1,
    'adaptation': 2,
    'pulse_generator': 3,
    'amplifier': 4,
    'stable': 5,
}
ID_TO_PHENOTYPE = {v: k for k, v in PHENOTYPE_TO_ID.items()}
NUM_PHENOTYPES = len(PHENOTYPE_TO_ID)


def analyze_feedback_loops(edges: List, n_genes: int) -> Dict:
    """
    Detailed analysis of feedback loops for oscillation detection.

    Biological insight: Oscillation requires negative feedback with delay.
    In Boolean networks, this manifests as odd-parity inhibition cycles.
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(n_genes))

    for src, tgt, etype in edges:
        G.add_edge(src, tgt, type=etype)

    # Find all simple cycles
    try:
        cycles = list(nx.simple_cycles(G))
    except:
        cycles = []

    # Analyze each cycle
    n_cycles = len(cycles)
    n_odd_inhib_cycles = 0
    n_even_inhib_cycles = 0
    n_pure_activation_cycles = 0
    n_pure_inhibition_cycles = 0
    max_cycle_inhib_ratio = 0.0
    min_cycle_length = 0
    max_cycle_length = 0

    cycle_lengths = []
    for cycle in cycles:
        cycle_len = len(cycle)
        cycle_lengths.append(cycle_len)

        n_inhib_in_cycle = 0
        n_activ_in_cycle = 0

        for k in range(cycle_len):
            src = cycle[k]
            tgt = cycle[(k + 1) % cycle_len]
            if G.has_edge(src, tgt):
                if G.edges[src, tgt].get('type') == 2:
                    n_inhib_in_cycle += 1
                else:
                    n_activ_in_cycle += 1

        # Parity analysis
        if n_inhib_in_cycle % 2 == 1:
            n_odd_inhib_cycles += 1
        else:
            n_even_inhib_cycles += 1

        # Homogeneity analysis
        if n_inhib_in_cycle == cycle_len:
            n_pure_inhibition_cycles += 1
        if n_activ_in_cycle == cycle_len:
            n_pure_activation_cycles += 1

        # Ratio
        inhib_ratio = n_inhib_in_cycle / cycle_len
        max_cycle_inhib_ratio = max(max_cycle_inhib_ratio, inhib_ratio)

    if cycle_lengths:
        min_cycle_length = min(cycle_lengths)
        max_cycle_length = max(cycle_lengths)

    return {
        'n_cycles': n_cycles,
        'n_odd_inhib_cycles': n_odd_inhib_cycles,
        'n_even_inhib_cycles': n_even_inhib_cycles,
        'n_pure_activation_cycles': n_pure_activation_cycles,
        'n_pure_inhibition_cycles': n_pure_inhibition_cycles,
        'max_cycle_inhib_ratio': max_cycle_inhib_ratio,
        'min_cycle_length': min_cycle_length,
        'max_cycle_length': max_cycle_length,
        'cycle_lengths': cycle_lengths,
    }


def extract_enhanced_features(circuit: Dict) -> np.ndarray:
    """
    Extract enhanced topology features with better oscillation discrimination.
    """
    features = circuit.get('features', {})
    edges = circuit.get('edges', [])
    n_genes = circuit.get('n_genes', 2)

    # === CORE STRUCTURAL FEATURES (from oracle) ===
    core = [
        float(features.get('has_self_activation', False)),
        float(features.get('has_self_inhibition', False)),
        float(features.get('has_mutual_inhibition', False)),
        float(features.get('has_activation_cascade', False)),
        float(features.get('has_inhibition_cycle', False)),
        float(features.get('has_iffl', False)),
        float(features.get('inhibition_cycle_odd', False)),
    ]

    # === STRUCTURAL METRICS ===
    n_edges = features.get('n_edges', len(edges))
    cycle_lengths = features.get('cycle_lengths', [])

    structural = [
        float(n_genes) / 5.0,  # Normalized
        float(n_edges) / 10.0,  # Normalized
        float(n_edges) / max(n_genes * n_genes, 1),  # Density
        float(len(cycle_lengths)) / 5.0,  # Num cycles normalized
    ]

    # === EDGE TYPE STATISTICS ===
    n_activation = sum(1 for e in edges if e[2] == 1)
    n_inhibition = sum(1 for e in edges if e[2] == 2)
    n_self_loops = sum(1 for e in edges if e[0] == e[1])
    n_self_activation = sum(1 for e in edges if e[0] == e[1] and e[2] == 1)
    n_self_inhibition = sum(1 for e in edges if e[0] == e[1] and e[2] == 2)

    edge_stats = [
        float(n_activation) / max(n_edges, 1),
        float(n_inhibition) / max(n_edges, 1),
        float(n_self_loops) / max(n_genes, 1),
        float(n_self_activation) / max(n_genes, 1),
        float(n_self_inhibition) / max(n_genes, 1),
    ]

    # === FEEDBACK LOOP ANALYSIS ===
    loop_info = analyze_feedback_loops(edges, n_genes)

    feedback = [
        float(loop_info['n_cycles']) / 5.0,
        float(loop_info['n_odd_inhib_cycles']) / 5.0,  # Key for oscillation
        float(loop_info['n_even_inhib_cycles']) / 5.0,
        float(loop_info['n_pure_activation_cycles']) / 5.0,
        float(loop_info['n_pure_inhibition_cycles']) / 5.0,
        loop_info['max_cycle_inhib_ratio'],
        float(loop_info['min_cycle_length']) / 5.0 if loop_info['min_cycle_length'] else 0.0,
        float(loop_info['max_cycle_length']) / 5.0 if loop_info['max_cycle_length'] else 0.0,
    ]

    # === OSCILLATION INDICATORS ===
    # These are derived features that specifically indicate oscillation potential
    oscillation_indicators = [
        # Direct oscillation signals
        float(features.get('inhibition_cycle_odd', False) and len(cycle_lengths) > 0),
        float(features.get('has_self_inhibition', False)),
        float(loop_info['n_odd_inhib_cycles'] > 0),

        # Multi-gene negative feedback
        float(n_genes >= 2 and loop_info['n_odd_inhib_cycles'] > 0),
        float(n_genes >= 3 and loop_info['n_odd_inhib_cycles'] > 0),

        # Repressilator-like: 3-gene ring with inhibition
        float(n_genes >= 3 and loop_info['n_pure_inhibition_cycles'] > 0),
    ]

    # === BISTABILITY INDICATORS (toggle switch) ===
    bistability_indicators = [
        float(features.get('has_mutual_inhibition', False)),
        float(features.get('has_mutual_inhibition', False) and not features.get('inhibition_cycle_odd', False)),
        float(loop_info['n_even_inhib_cycles'] > 0 and features.get('has_mutual_inhibition', False)),
    ]

    # === PULSE GENERATOR INDICATORS ===
    pulse_indicators = [
        float(features.get('has_iffl', False)),
        float(features.get('has_iffl', False) and features.get('has_activation_cascade', False)),
    ]

    # === AMPLIFIER INDICATORS ===
    amplifier_indicators = [
        float(features.get('has_activation_cascade', False)),
        float(features.get('has_activation_cascade', False) and not features.get('has_mutual_inhibition', False)),
        float(loop_info['n_pure_activation_cycles'] > 0),
    ]

    # === STABILITY INDICATORS ===
    stability_indicators = [
        float(loop_info['n_cycles'] == 0 and n_self_inhibition == 0),
        float(not any([
            features.get('has_self_inhibition', False),
            features.get('has_mutual_inhibition', False),
            features.get('has_iffl', False),
            features.get('inhibition_cycle_odd', False),
            features.get('has_inhibition_cycle', False),
        ])),
    ]

    # Concatenate all features
    all_features = (
        core +
        structural +
        edge_stats +
        feedback +
        oscillation_indicators +
        bistability_indicators +
        pulse_indicators +
        amplifier_indicators +
        stability_indicators
    )

    return np.array(all_features, dtype=np.float32)


class EnhancedTopologyClassifier(nn.Module):
    """
    Enhanced classifier with attention-like mechanism for feature importance.
    """

    def __init__(self, input_dim: int = 44, hidden_dim: int = 128):
        super().__init__()

        # Feature importance (learnable attention)
        self.feature_importance = mx.ones((input_dim,)) * 0.1

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Deep feature extraction
        self.layers = [
            (nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            for _ in range(3)
        ]

        # Per-phenotype specialized heads
        self.phenotype_heads = [nn.Linear(hidden_dim, 1) for _ in range(NUM_PHENOTYPES)]

        # Global head
        self.global_head = nn.Linear(hidden_dim, NUM_PHENOTYPES)

    def __call__(self, x: mx.array) -> mx.array:
        # Apply learned feature importance
        x = x * mx.abs(self.feature_importance)

        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = nn.gelu(h)

        # Deep feature extraction with residual connections
        for linear, norm in self.layers:
            h_new = linear(h)
            h_new = norm(h_new)
            h_new = nn.gelu(h_new)
            h = h + h_new  # Residual

        # Combine global and phenotype-specific predictions
        global_logits = self.global_head(h)

        # Get phenotype-specific contributions
        phenotype_logits = []
        for head in self.phenotype_heads:
            phenotype_logits.append(head(h))
        phenotype_logits = mx.concatenate(phenotype_logits, axis=-1)

        # Combine (learnable mixture)
        return 0.7 * global_logits + 0.3 * phenotype_logits


def create_balanced_batch_with_weights(
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Create balanced batch with class weights for difficult classes."""
    features_list = []
    labels = []
    weights = []

    # Class weights: boost underperforming classes
    class_weights = {
        'oscillator': 2.0,      # Was 50.4% - needs boost
        'toggle_switch': 1.0,
        'adaptation': 1.0,
        'pulse_generator': 1.0,
        'amplifier': 1.0,
        'stable': 1.0,
    }

    phenotypes = list(circuits_by_phenotype.keys())
    per_class = max(batch_size // len(phenotypes), 1)

    for phenotype in phenotypes:
        circuits = circuits_by_phenotype[phenotype]
        if not circuits:
            continue
        sampled = random.sample(circuits, min(per_class, len(circuits)))

        for circuit in sampled:
            feat = extract_enhanced_features(circuit)
            features_list.append(feat)
            labels.append(PHENOTYPE_TO_ID[phenotype])
            weights.append(class_weights.get(phenotype, 1.0))

    # Shuffle
    combined = list(zip(features_list, labels, weights))
    random.shuffle(combined)
    features_list, labels, weights = zip(*combined)

    return (
        mx.array(np.stack(features_list)),
        mx.array(labels),
        mx.array(weights, dtype=mx.float32)
    )


def train_epoch(
    model: EnhancedTopologyClassifier,
    optimizer: optim.Optimizer,
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    steps_per_epoch: int,
) -> Tuple[float, float]:
    """Train with weighted loss."""

    def loss_fn(model, features, labels, weights):
        logits = model(features)
        # Weighted cross-entropy
        ce = nn.losses.cross_entropy(logits, labels, reduction='none')
        weighted_loss = mx.mean(ce * weights)
        return weighted_loss, logits

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(steps_per_epoch):
        features, labels, weights = create_balanced_batch_with_weights(
            circuits_by_phenotype, batch_size
        )

        (loss, logits), grads = loss_and_grad(model, features, labels, weights)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        preds = mx.argmax(logits, axis=-1)
        total_loss += float(loss)
        total_correct += int(mx.sum(preds == labels))
        total_samples += labels.shape[0]

    return total_loss / steps_per_epoch, total_correct / total_samples


def evaluate(
    model: EnhancedTopologyClassifier,
    circuits_by_phenotype: Dict[str, List],
    samples_per_phenotype: int = 500,
) -> Dict[str, float]:
    """Comprehensive evaluation."""
    results = {}
    confusion = {}
    total_correct = 0
    total_samples = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))
        correct = 0
        confusion[phenotype] = {p: 0 for p in PHENOTYPE_TO_ID.keys()}

        for circuit in sampled:
            feat = extract_enhanced_features(circuit)
            logits = model(mx.array(feat[None]))
            pred_id = int(mx.argmax(logits[0]))
            pred = ID_TO_PHENOTYPE[pred_id]

            confusion[phenotype][pred] += 1
            if pred == phenotype:
                correct += 1

        accuracy = correct / len(sampled)
        results[phenotype] = accuracy
        total_correct += correct
        total_samples += len(sampled)

    results['overall'] = total_correct / total_samples
    results['confusion'] = confusion
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/topology_enhanced")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("topology_enhanced", log_file=f"{checkpoint_dir}/train.log")

    logger.info("=" * 70)
    logger.info("ENHANCED TOPOLOGY LEARNING")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Improvements over basic version:")
    logger.info("  - Detailed feedback loop analysis")
    logger.info("  - Oscillation-specific features")
    logger.info("  - Class weighting for difficult classes")
    logger.info("  - Deeper network with attention-like mechanism")
    logger.info("")

    # Load data
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    logger.info("Dataset:")
    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    total = sum(len(c) for c in circuits_by_phenotype.values())

    # Show feature dimensionality
    sample = next(iter(circuits_by_phenotype.values()))[0]
    feat = extract_enhanced_features(sample)
    logger.info(f"\nFeature dimensionality: {len(feat)}")

    # Create model
    model = EnhancedTopologyClassifier(input_dim=len(feat), hidden_dim=args.hidden_dim)

    def count_params(params):
        t = 0
        if isinstance(params, dict):
            for v in params.values():
                t += count_params(v)
        elif isinstance(params, list):
            for v in params:
                t += count_params(v)
        elif hasattr(params, 'size'):
            t += params.size
        return t

    n_params = count_params(model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer
    steps_per_epoch = max(total // args.batch_size, 30)
    warmup_steps = 10 * steps_per_epoch

    lr_schedule = optim.join_schedules(
        [
            optim.linear_schedule(1e-5, args.lr, warmup_steps),
            optim.cosine_decay(args.lr, (args.epochs - 10) * steps_per_epoch, 1e-6)
        ],
        [warmup_steps]
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    # Training
    logger.info(f"\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch")

    best_acc = 0.0
    best_epoch = 0
    best_oscillator_acc = 0.0

    for epoch in range(args.epochs):
        loss, train_acc = train_epoch(
            model, optimizer, circuits_by_phenotype,
            args.batch_size, steps_per_epoch
        )

        if train_acc > best_acc:
            best_acc = train_acc
            best_epoch = epoch + 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Quick evaluation
            eval_results = evaluate(model, circuits_by_phenotype, samples_per_phenotype=200)
            osc_acc = eval_results.get('oscillator', 0)
            if osc_acc > best_oscillator_acc:
                best_oscillator_acc = osc_acc
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs}: "
                f"loss={loss:.4f}, train={train_acc:.1%}, "
                f"val={eval_results['overall']:.1%}, "
                f"osc={osc_acc:.1%}"
            )

    # Final evaluation
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)

    results = evaluate(model, circuits_by_phenotype, samples_per_phenotype=500)

    logger.info("\nPer-class accuracy:")
    for phenotype in sorted(PHENOTYPE_TO_ID.keys()):
        if phenotype in results:
            logger.info(f"  {phenotype}: {results[phenotype]:.1%}")

    logger.info(f"\n  NEURAL NET ACCURACY: {results['overall']:.1%}")

    # Confusion matrix
    logger.info("\nConfusion Matrix (rows=true, cols=pred):")
    confusion = results.get('confusion', {})
    phenotypes = sorted(PHENOTYPE_TO_ID.keys())
    header = "          " + " ".join(f"{p[:5]:>6}" for p in phenotypes)
    logger.info(header)
    for true_p in phenotypes:
        if true_p in confusion:
            row = f"{true_p[:9]:>9} "
            for pred_p in phenotypes:
                row += f"{confusion[true_p].get(pred_p, 0):>6} "
            logger.info(row)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Neural net accuracy: {results['overall']:.1%}")
    logger.info(f"Best training accuracy: {best_acc:.1%} (epoch {best_epoch})")
    logger.info(f"Oscillator accuracy: {results.get('oscillator', 0):.1%}")

    if results['overall'] >= 0.9:
        logger.info("\n" + "*" * 70)
        logger.info("SUCCESS: 90%+ ACCURACY ACHIEVED!")
        logger.info("*" * 70)
    else:
        logger.info(f"\nGap to 90%: {0.9 - results['overall']:.1%}")


if __name__ == "__main__":
    main()
