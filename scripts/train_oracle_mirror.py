#!/usr/bin/env python3
"""
Oracle Mirror Network - Learning the Oracle's Decision Function

=== THE KEY INSIGHT ===

The oracle classifier uses a COMBINATION of:
1. Dynamical features (from simulation)
2. Topological features (structural patterns)

The decision logic is:
1. If num_fixed_points >= 2 AND has_mutual_inhibition → toggle_switch
2. If num_oscillations > 0 with avg_period >= 2 → oscillator
3. If has_iffl_topology → pulse_generator
4. If has_cascade_topology → amplifier
5. If has_negative_feedback AND is_adaptive → adaptation
6. Otherwise → stable

To achieve 90%+ accuracy, we need to:
1. Compute the EXACT same features the oracle uses
2. Learn the decision function from these features

This is essentially KNOWLEDGE DISTILLATION where we learn the oracle's behavior
from its outputs (the labels in verified_circuits.json).

=== ARCHITECTURE ===

We compute the EXACT oracle features:
- Dynamical: num_fixed_points, num_oscillations, max_period, avg_period
- Topological: mutual_inhib, iffl, cascade, neg_feedback, self_inhib

Then we train a simple classifier on these features.

If our features match the oracle's internal representation, we should achieve
near-100% accuracy (limited only by any noise in the labels).
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier
from src.utils.logging import setup_logger


# ============================================================================
# CONSTANTS
# ============================================================================

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


# ============================================================================
# EXACT STATE TRANSITION GRAPH COMPUTATION
# ============================================================================

def compute_next_state(
    state: Tuple[int, ...],
    adj: np.ndarray,
    n_genes: int
) -> Tuple[int, ...]:
    """Compute next state using constitutive update rule."""
    next_state = []
    for i in range(n_genes):
        activation = 0
        inhibition = 0
        n_activators = 0
        n_inhibitors = 0

        for j in range(n_genes):
            if adj[j, i] == 1:
                n_activators += 1
                activation += state[j]
            elif adj[j, i] == -1:
                n_inhibitors += 1
                inhibition += state[j]

        if n_activators == 0 and n_inhibitors == 0:
            next_state.append(state[i])
        elif n_activators == 0:
            next_state.append(0 if inhibition > 0 else 1)
        elif n_inhibitors == 0:
            next_state.append(1 if activation > 0 else 0)
        else:
            if activation > inhibition:
                next_state.append(1)
            elif inhibition > activation:
                next_state.append(0)
            else:
                next_state.append(1 if activation > 0 else 0)

    return tuple(next_state)


def compute_exact_dynamics(edges: List, n_genes: int) -> Dict:
    """
    Compute EXACT dynamics by exploring the full state space.

    For small circuits (2-3 genes), this is tractable and gives
    ground truth attractor information.
    """
    # Build adjacency
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype

    n_states = 2 ** n_genes

    # Compute all transitions
    next_state_map = {}
    for state_idx in range(n_states):
        state = tuple((state_idx >> i) & 1 for i in range(n_genes))
        next_state = compute_next_state(state, adj, n_genes)
        next_state_map[state] = next_state

    # Find attractors
    visited = set()
    fixed_points = []
    cycles = []

    for start_idx in range(n_states):
        start = tuple((start_idx >> i) & 1 for i in range(n_genes))
        if start in visited:
            continue

        trajectory = []
        trajectory_set = set()
        current = start

        while current not in trajectory_set and current not in visited:
            trajectory.append(current)
            trajectory_set.add(current)
            current = next_state_map[current]

        if current in visited:
            visited.update(trajectory)
            continue

        # Found attractor
        cycle_start_idx = trajectory.index(current)
        cycle = trajectory[cycle_start_idx:]

        if len(cycle) == 1:
            fixed_points.append(cycle[0])
        else:
            cycles.append(cycle)

        visited.update(trajectory)

    return {
        'num_fixed_points': len(fixed_points),
        'num_oscillations': len(cycles),
        'oscillation_periods': [len(c) for c in cycles],
        'avg_period': np.mean([len(c) for c in cycles]) if cycles else 0,
        'max_period': max([len(c) for c in cycles]) if cycles else 0,
        'num_attractors': len(fixed_points) + len(cycles),
    }


# ============================================================================
# TOPOLOGICAL PATTERN DETECTION
# ============================================================================

def has_mutual_inhibition(adj: np.ndarray, n_genes: int) -> bool:
    """Check for mutual inhibition (toggle switch topology)."""
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if adj[i, j] == -1 and adj[j, i] == -1:
                return True
    return False


def has_iffl_topology(adj: np.ndarray, n_genes: int) -> bool:
    """Check for incoherent feed-forward loop."""
    for a in range(n_genes):
        for b in range(n_genes):
            if a == b:
                continue
            # A activates B
            if adj[a, b] != 1:
                continue
            for c in range(n_genes):
                if c == a or c == b:
                    continue
                # A activates C AND B inhibits C → Type 1 IFFL
                if adj[a, c] == 1 and adj[b, c] == -1:
                    return True
                # A inhibits C AND B activates C → Type 2 IFFL
                if adj[a, c] == -1 and adj[b, c] == 1:
                    return True
    return False


def has_cascade_topology(adj: np.ndarray, n_genes: int) -> bool:
    """Check for activation cascade without negative feedback."""
    # Find activation chains
    has_chain = False
    for start in range(n_genes):
        visited = {start}
        current = start
        chain_len = 0

        while chain_len < n_genes:
            found_next = False
            for target in range(n_genes):
                if target not in visited and adj[current, target] == 1:
                    visited.add(target)
                    current = target
                    chain_len += 1
                    found_next = True
                    break
            if not found_next:
                break

        if chain_len >= 2:
            has_chain = True
            break

    # Check for negative feedback (would make it non-amplifier)
    has_neg_feedback = False
    for i in range(n_genes):
        if adj[i, i] == -1:  # Self-inhibition
            has_neg_feedback = True
            break

    return has_chain and not has_neg_feedback


def has_self_inhibition(adj: np.ndarray, n_genes: int) -> bool:
    """Check for self-inhibition (oscillator/adaptation indicator)."""
    for i in range(n_genes):
        if adj[i, i] == -1:
            return True
    return False


def has_negative_feedback_loop(adj: np.ndarray, n_genes: int) -> bool:
    """Check for any negative feedback loop."""
    # Self-inhibition is simplest negative feedback
    if has_self_inhibition(adj, n_genes):
        return True

    # Check for longer cycles with odd inhibition count
    for start in range(n_genes):
        # BFS to find cycles
        visited = set()
        queue = [(start, 0, [start])]  # (node, inhib_count, path)

        while queue:
            current, inhib_count, path = queue.pop(0)

            for target in range(n_genes):
                edge = adj[current, target]
                if edge == 0:
                    continue

                new_inhib = inhib_count + (1 if edge == -1 else 0)

                if target == start and len(path) > 1:
                    # Found cycle back to start
                    if new_inhib % 2 == 1:  # Odd = negative feedback
                        return True
                elif target not in visited and len(path) < n_genes:
                    visited.add(target)
                    queue.append((target, new_inhib, path + [target]))

    return False


# ============================================================================
# ORACLE FEATURE EXTRACTION
# ============================================================================

def compute_oracle_features(edges: List, n_genes: int) -> np.ndarray:
    """
    Compute the EXACT features the oracle uses for classification.

    This mirrors the BehaviorClassifier's internal logic.
    """
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype

    # Dynamical features (from exact STG analysis)
    dynamics = compute_exact_dynamics(edges, n_genes)

    # Topological features
    mutual_inhib = has_mutual_inhibition(adj, n_genes)
    iffl = has_iffl_topology(adj, n_genes)
    cascade = has_cascade_topology(adj, n_genes)
    self_inhib = has_self_inhibition(adj, n_genes)
    neg_feedback = has_negative_feedback_loop(adj, n_genes)

    features = [
        # Dynamics
        dynamics['num_fixed_points'],
        dynamics['num_oscillations'],
        dynamics['max_period'],
        dynamics['avg_period'],
        dynamics['num_attractors'],

        # Binary indicators
        float(dynamics['num_fixed_points'] >= 2),
        float(dynamics['num_oscillations'] > 0),
        float(dynamics['avg_period'] >= 2),

        # Topology
        float(mutual_inhib),
        float(iffl),
        float(cascade),
        float(self_inhib),
        float(neg_feedback),

        # Combined features (matching oracle logic)
        float(dynamics['num_fixed_points'] >= 2 and mutual_inhib),  # Toggle condition
        float(dynamics['num_oscillations'] > 0 and dynamics['avg_period'] >= 2),  # Oscillator condition

        # Additional structural
        n_genes,
        len(edges),
    ]

    return np.array(features, dtype=np.float32)


# ============================================================================
# RULE-BASED CLASSIFIER (ORACLE REPLICA)
# ============================================================================

def rule_based_classify(edges: List, n_genes: int) -> str:
    """
    Rule-based classification that EXACTLY mirrors the oracle logic.

    This should achieve near-100% accuracy on the oracle-labeled data.
    """
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype

    # Compute dynamics
    dynamics = compute_exact_dynamics(edges, n_genes)

    # Compute topology
    mutual_inhib = has_mutual_inhibition(adj, n_genes)
    iffl = has_iffl_topology(adj, n_genes)
    cascade = has_cascade_topology(adj, n_genes)
    neg_feedback = has_negative_feedback_loop(adj, n_genes)

    # Oracle decision logic (in priority order)

    # 1. Toggle switch: bistable with mutual inhibition
    if dynamics['num_fixed_points'] >= 2 and mutual_inhib:
        return 'toggle_switch'

    # 2. Oscillator: has oscillations with period >= 2
    if dynamics['num_oscillations'] > 0 and dynamics['avg_period'] >= 2:
        return 'oscillator'

    # 3. Pulse generator: IFFL topology
    if iffl:
        return 'pulse_generator'

    # 4. Amplifier: cascade topology
    if cascade:
        return 'amplifier'

    # 5. Adaptation: has negative feedback (remaining cases)
    if neg_feedback:
        return 'adaptation'

    # 6. Default: stable
    return 'stable'


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class OracleMirrorNetwork(nn.Module):
    """
    Neural network to learn the oracle's decision function.

    Given EXACT oracle features, should learn near-perfect classification.
    """

    def __init__(self, input_dim: int = 17, hidden_dim: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_PHENOTYPES),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


# ============================================================================
# TRAINING
# ============================================================================

FEATURE_CACHE = {}


def get_features(edges: List, n_genes: int) -> np.ndarray:
    """Get or compute features with caching."""
    key = (n_genes, tuple(sorted(tuple(e) for e in edges)))
    if key not in FEATURE_CACHE:
        FEATURE_CACHE[key] = compute_oracle_features(edges, n_genes)
    return FEATURE_CACHE[key]


def create_batch(
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
) -> Tuple[mx.array, mx.array]:
    """Create balanced batch."""
    feat_list = []
    labels = []

    phenotypes = list(circuits_by_phenotype.keys())
    per_class = max(batch_size // len(phenotypes), 1)

    for phenotype in phenotypes:
        circuits = circuits_by_phenotype[phenotype]
        sampled = random.sample(circuits, min(per_class, len(circuits)))

        for circuit in sampled:
            edges = [tuple(e) for e in circuit['edges']]
            n_genes = circuit['n_genes']
            feat = get_features(edges, n_genes)
            feat_list.append(feat)
            labels.append(PHENOTYPE_TO_ID[phenotype])

    combined = list(zip(feat_list, labels))
    random.shuffle(combined)
    feat_list, labels = zip(*combined)

    return mx.array(np.stack(feat_list)), mx.array(labels)


def train_epoch(
    model: OracleMirrorNetwork,
    optimizer: optim.Optimizer,
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    steps_per_epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch."""

    def loss_fn(model, features, labels):
        logits = model(features)
        return mx.mean(nn.losses.cross_entropy(logits, labels)), logits

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(steps_per_epoch):
        features, labels = create_batch(circuits_by_phenotype, batch_size)

        (loss, logits), grads = loss_and_grad(model, features, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        preds = mx.argmax(logits, axis=-1)
        total_loss += float(loss)
        total_correct += int(mx.sum(preds == labels))
        total_samples += labels.shape[0]

    return total_loss / steps_per_epoch, total_correct / total_samples


def evaluate(
    model: OracleMirrorNetwork,
    circuits_by_phenotype: Dict[str, List],
    samples_per_phenotype: int = 500,
) -> Dict[str, float]:
    """Evaluate model."""
    results = {}
    total_correct = 0
    total_samples = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))
        correct = 0

        for circuit in sampled:
            edges = [tuple(e) for e in circuit['edges']]
            n_genes = circuit['n_genes']
            feat = get_features(edges, n_genes)

            logits = model(mx.array(feat[None]))
            pred = ID_TO_PHENOTYPE[int(mx.argmax(logits[0]))]

            if pred == phenotype:
                correct += 1

        accuracy = correct / len(sampled)
        results[phenotype] = accuracy
        total_correct += correct
        total_samples += len(sampled)

    results['overall'] = total_correct / total_samples
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Oracle Mirror Network")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/oracle_mirror")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("oracle_mirror", log_file=f"{checkpoint_dir}/train.log")

    logger.info("=" * 70)
    logger.info("ORACLE MIRROR NETWORK")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Key insight: Use the EXACT features the oracle computes internally")
    logger.info("If we mirror the oracle's feature extraction, we can learn its decision.")
    logger.info("")

    # Load data
    logger.info(f"Loading circuits from {args.verified_circuits}")
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    total = sum(len(c) for c in circuits_by_phenotype.values())

    # Pre-compute features
    logger.info("\nPre-computing oracle features...")
    computed = 0
    for phenotype, circuits in circuits_by_phenotype.items():
        for circuit in circuits:
            edges = [tuple(e) for e in circuit['edges']]
            n_genes = circuit['n_genes']
            get_features(edges, n_genes)
            computed += 1
            if computed % 2000 == 0:
                logger.info(f"  Computed {computed}/{total}")
    logger.info(f"  Done: {len(FEATURE_CACHE)} unique circuits")

    # Evaluate rule-based classifier (should be near 100%)
    logger.info("\nEvaluating rule-based classifier (oracle replica)...")
    rule_results = {}
    rule_total = 0
    rule_correct = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        correct = 0
        tested = 0
        for circuit in circuits[:500]:
            edges = [tuple(e) for e in circuit['edges']]
            n_genes = circuit['n_genes']
            pred = rule_based_classify(edges, n_genes)
            if pred == phenotype:
                correct += 1
            tested += 1

        accuracy = correct / tested
        rule_results[phenotype] = accuracy
        rule_correct += correct
        rule_total += tested
        logger.info(f"  {phenotype}: {correct}/{tested} ({accuracy:.1%})")

    rule_accuracy = rule_correct / rule_total
    logger.info(f"  RULE-BASED ACCURACY: {rule_accuracy:.1%}")

    if rule_accuracy < 0.9:
        logger.info("\n  WARNING: Rule-based accuracy is below 90%!")
        logger.info("  This means our dynamics/topology computation differs from oracle.")
        logger.info("  Need to debug feature extraction.")
    else:
        logger.info("\n  ✓ Rule-based accuracy is high - features match oracle!")

    # Create model
    model = OracleMirrorNetwork(input_dim=17, hidden_dim=args.hidden_dim)

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

    logger.info(f"\nModel parameters: {count_params(model.parameters()):,}")

    # Optimizer
    steps_per_epoch = max(total // args.batch_size, 30)

    lr_schedule = optim.join_schedules(
        [
            optim.linear_schedule(1e-5, args.lr, 10),
            optim.cosine_decay(args.lr, args.epochs - 10, 1e-6)
        ],
        [10]
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    # Training
    logger.info(f"\nTraining: {args.epochs} epochs, {steps_per_epoch} steps/epoch")
    logger.info("")

    best_acc = 0.0

    for epoch in range(args.epochs):
        loss, train_acc = train_epoch(
            model, optimizer, circuits_by_phenotype,
            args.batch_size, steps_per_epoch
        )

        if train_acc > best_acc:
            best_acc = train_acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs}: "
                f"loss={loss:.4f}, acc={train_acc:.1%}, best={best_acc:.1%}"
            )

    # Final evaluation
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)

    results = evaluate(model, circuits_by_phenotype, samples_per_phenotype=500)

    for phenotype in sorted(results.keys()):
        if phenotype != 'overall':
            logger.info(f"  {phenotype}: {results[phenotype]:.1%}")

    logger.info(f"\n  NEURAL NET ACCURACY: {results['overall']:.1%}")
    logger.info(f"  RULE-BASED ACCURACY: {rule_accuracy:.1%}")

    # Save
    def flatten_params(params, prefix=""):
        flat = {}
        if isinstance(params, dict):
            for k, v in params.items():
                flat.update(flatten_params(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                flat.update(flatten_params(v, f"{prefix}.{i}" if prefix else str(i)))
        else:
            flat[prefix] = params
        return flat

    flat_weights = flatten_params(model.parameters())
    mx.save_safetensors(str(checkpoint_dir / "best.safetensors"), flat_weights)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Rule-based accuracy: {rule_accuracy:.1%}")
    logger.info(f"Neural net accuracy: {results['overall']:.1%}")

    if results['overall'] >= 0.9:
        logger.info("\n🎉 TARGET ACHIEVED: 90%+ accuracy!")
    elif rule_accuracy >= 0.9:
        logger.info("\n✓ Rule-based achieves 90%+! Neural net can be improved.")
    else:
        logger.info("\n⚠ Feature extraction needs debugging.")


if __name__ == "__main__":
    main()
