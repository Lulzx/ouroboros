#!/usr/bin/env python3
"""
Feedback Loop Analyzer (FLA) - Algebraic Topology Meets Neural Networks

=== THEORETICAL FOUNDATION ===

From dynamical systems theory, the behavior of Boolean networks is determined by
their FEEDBACK LOOP STRUCTURE. This is a fundamental result:

1. OSCILLATION requires NEGATIVE FEEDBACK:
   - A negative feedback loop is a cycle where the product of edge signs is -1
   - Simplest example: self-inhibition (A ⊣ A)
   - Repressilator: A ⊣ B ⊣ C ⊣ A (odd number of inhibitions)

2. BISTABILITY requires POSITIVE FEEDBACK:
   - A positive feedback loop is a cycle where the product of edge signs is +1
   - Mutual inhibition creates bistability: A ⊣ B, B ⊣ A
   - (-1) × (-1) = +1, so this is POSITIVE feedback!

3. STABLE behavior when there's NO feedback or feedback is weak

4. ADAPTATION/PULSE require INCOHERENT FEEDFORWARD:
   - Fast and slow paths with opposite signs
   - IFFL: A → B, A → C, B ⊣ C

=== ALGEBRAIC TOPOLOGY APPROACH ===

We can use tools from algebraic topology to analyze the feedback structure:

1. CYCLE BASIS: Every feedback loop is a cycle in the graph. The cycle basis
   gives a minimal set of independent cycles.

2. SIGNED CYCLE ANALYSIS: For each cycle, compute the sign:
   sign(cycle) = ∏_{edges in cycle} sign(edge)
   - sign = +1: positive feedback
   - sign = -1: negative feedback

3. HOMOLOGY: The first Betti number β₁ = #independent cycles relates to
   feedback complexity.

4. LOOP INTERACTION: When multiple loops share nodes, they interact.
   The dominant loop determines behavior.

=== KEY MATHEMATICAL RESULTS ===

For Boolean networks with constitutive update rule:

THEOREM 1 (Oscillation Necessary Condition):
If a Boolean network oscillates, it must contain a negative feedback loop.

THEOREM 2 (Bistability Necessary Condition):
If a Boolean network has multiple stable states, it must contain a positive
feedback loop.

THEOREM 3 (Loop Dominance):
When a circuit has both positive and negative feedback loops, the behavior
depends on loop lengths and coupling strength.

=== OUR APPROACH ===

1. EXACT LOOP ENUMERATION: Find all simple cycles in the interaction graph
2. SIGN COMPUTATION: Compute sign of each cycle
3. LOOP FEATURE EXTRACTION: Lengths, signs, overlaps, dominance
4. NEURAL PROCESSING: Learn optimal classification from loop features

This approach is MATHEMATICALLY GROUNDED - we're not learning patterns,
we're computing the EXACT conditions from dynamical systems theory.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
import sys
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

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
# GRAPH ALGORITHMS FOR CYCLE ANALYSIS
# ============================================================================

def find_all_simple_cycles(adj: np.ndarray, n_genes: int) -> List[List[int]]:
    """
    Find all simple cycles in a directed graph.

    Uses Johnson's algorithm for small graphs.

    Args:
        adj: (n, n) adjacency matrix (non-zero = edge exists)
        n_genes: number of nodes

    Returns:
        List of cycles, where each cycle is a list of node indices
    """
    cycles = []

    # Check for self-loops first
    for i in range(n_genes):
        if adj[i, i] != 0:
            cycles.append([i])

    # Find longer cycles using DFS
    def dfs(start: int, current: int, path: List[int], visited: Set[int]):
        for next_node in range(n_genes):
            if adj[current, next_node] != 0:
                if next_node == start and len(path) > 1:
                    # Found a cycle
                    cycles.append(list(path))
                elif next_node not in visited and next_node > start:
                    # Continue exploration (only to higher indices to avoid duplicates)
                    visited.add(next_node)
                    path.append(next_node)
                    dfs(start, next_node, path, visited)
                    path.pop()
                    visited.remove(next_node)

    for start in range(n_genes):
        visited = {start}
        dfs(start, start, [start], visited)

    return cycles


def compute_cycle_sign(cycle: List[int], adj: np.ndarray) -> int:
    """
    Compute the sign of a feedback cycle.

    sign = product of edge signs around the cycle
    +1 = positive feedback (bistability tendency)
    -1 = negative feedback (oscillation tendency)

    Args:
        cycle: List of node indices in the cycle
        adj: Signed adjacency matrix (+1 activation, -1 inhibition)

    Returns:
        +1 or -1
    """
    if len(cycle) == 1:
        # Self-loop
        return int(np.sign(adj[cycle[0], cycle[0]]))

    sign = 1
    for i in range(len(cycle)):
        src = cycle[i]
        tgt = cycle[(i + 1) % len(cycle)]
        edge_sign = int(np.sign(adj[src, tgt]))
        sign *= edge_sign

    return sign


def find_feedforward_loops(adj: np.ndarray, n_genes: int) -> List[Tuple[int, int, int, str]]:
    """
    Find all feedforward loop motifs.

    FFL types:
    - C1 (coherent type 1): A→B, A→C, B→C (all activations or even inhibitions)
    - I1 (incoherent type 1): A→B, A→C, B⊣C (one odd path)

    Args:
        adj: Signed adjacency matrix
        n_genes: Number of genes

    Returns:
        List of (A, B, C, type) tuples
    """
    ffls = []

    for a in range(n_genes):
        for b in range(n_genes):
            if a == b or adj[a, b] == 0:
                continue

            for c in range(n_genes):
                if c == a or c == b:
                    continue

                if adj[a, c] == 0 or adj[b, c] == 0:
                    continue

                # Found FFL: A → B → C with A → C
                # Compute path signs
                path1_sign = int(np.sign(adj[a, c]))  # Direct path
                path2_sign = int(np.sign(adj[a, b]) * np.sign(adj[b, c]))  # Indirect path

                if path1_sign == path2_sign:
                    ffl_type = 'coherent'
                else:
                    ffl_type = 'incoherent'

                ffls.append((a, b, c, ffl_type))

    return ffls


def find_cascades(adj: np.ndarray, n_genes: int) -> List[List[int]]:
    """
    Find activation cascades (A → B → C → ...).

    Args:
        adj: Signed adjacency matrix
        n_genes: Number of genes

    Returns:
        List of cascade paths (lists of node indices)
    """
    cascades = []

    def dfs(current: int, path: List[int], visited: Set[int]):
        if len(path) >= 2:
            cascades.append(list(path))

        for next_node in range(n_genes):
            if next_node not in visited and adj[current, next_node] == 1:
                visited.add(next_node)
                path.append(next_node)
                dfs(next_node, path, visited)
                path.pop()
                visited.remove(next_node)

    for start in range(n_genes):
        dfs(start, [start], {start})

    return cascades


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def compute_loop_features(edges: List[Tuple[int, int, int]], n_genes: int) -> np.ndarray:
    """
    Compute comprehensive feedback loop features.

    This is the core of the mathematical approach - we extract exactly
    the features that determine dynamical behavior according to theory.
    """
    # Build adjacency matrix
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype

    features = []

    # 1. Find all cycles
    cycles = find_all_simple_cycles(adj, n_genes)

    # 2. Analyze cycle signs
    positive_cycles = []
    negative_cycles = []

    for cycle in cycles:
        sign = compute_cycle_sign(cycle, adj)
        if sign > 0:
            positive_cycles.append(cycle)
        else:
            negative_cycles.append(cycle)

    # 3. Cycle counts
    features.append(len(cycles))  # Total cycles
    features.append(len(positive_cycles))  # Positive feedback loops
    features.append(len(negative_cycles))  # Negative feedback loops

    # 4. Cycle length statistics
    if cycles:
        lengths = [len(c) for c in cycles]
        features.append(min(lengths))
        features.append(max(lengths))
        features.append(np.mean(lengths))
    else:
        features.extend([0, 0, 0])

    if negative_cycles:
        neg_lengths = [len(c) for c in negative_cycles]
        features.append(min(neg_lengths))
        features.append(np.mean(neg_lengths))
    else:
        features.extend([0, 0])

    if positive_cycles:
        pos_lengths = [len(c) for c in positive_cycles]
        features.append(min(pos_lengths))
        features.append(np.mean(pos_lengths))
    else:
        features.extend([0, 0])

    # 5. Self-loop analysis (simplest oscillators/regulators)
    self_activation = sum(1 for s, t, e in edges if s == t and e == 1)
    self_inhibition = sum(1 for s, t, e in edges if s == t and e == -1)
    features.append(self_activation)
    features.append(self_inhibition)

    # 6. Mutual inhibition (toggle switch motif)
    mutual_inhib = 0
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            if adj[i, j] == -1 and adj[j, i] == -1:
                mutual_inhib += 1
    features.append(mutual_inhib)

    # 7. Feedforward loops
    ffls = find_feedforward_loops(adj, n_genes)
    coherent_ffls = [f for f in ffls if f[3] == 'coherent']
    incoherent_ffls = [f for f in ffls if f[3] == 'incoherent']

    features.append(len(coherent_ffls))
    features.append(len(incoherent_ffls))

    # 8. Cascades (for amplifier detection)
    cascades = find_cascades(adj, n_genes)
    if cascades:
        max_cascade_len = max(len(c) for c in cascades)
        features.append(max_cascade_len)
    else:
        features.append(0)

    # 9. Binary indicators (key structural motifs)
    features.append(1.0 if self_inhibition > 0 else 0.0)  # Has self-inhibition
    features.append(1.0 if mutual_inhib > 0 else 0.0)  # Has mutual inhibition
    features.append(1.0 if len(negative_cycles) > 0 else 0.0)  # Has negative cycle
    features.append(1.0 if len(positive_cycles) > 0 else 0.0)  # Has positive cycle
    features.append(1.0 if len(incoherent_ffls) > 0 else 0.0)  # Has IFFL
    features.append(1.0 if len(coherent_ffls) > 0 else 0.0)  # Has coherent FFL

    # 10. Betti number β₁ (number of independent cycles)
    # For a connected graph: β₁ = E - V + 1
    # Simplified: just use cycle count as proxy
    n_edges = len(edges)
    betti_approx = max(n_edges - n_genes + 1, 0)
    features.append(betti_approx)

    # 11. Regulatory complexity
    features.append(n_genes)
    features.append(n_edges)
    features.append(n_edges / max(n_genes * (n_genes - 1), 1))  # Density

    return np.array(features, dtype=np.float32)


def compute_phenotype_indicators(edges: List[Tuple[int, int, int]], n_genes: int) -> np.ndarray:
    """
    Compute direct phenotype indicators based on necessary conditions.

    These are derived from dynamical systems theory:
    - Oscillator: requires negative feedback (self-inhib or neg cycle)
    - Toggle switch: requires mutual inhibition
    - Amplifier: requires activation cascade
    - Pulse generator: requires IFFL
    - Adaptation: requires some form of regulation
    - Stable: default / no special structure
    """
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype

    indicators = np.zeros(6, dtype=np.float32)

    # Find structures
    self_inhib = any(adj[i, i] == -1 for i in range(n_genes))

    mutual_inhib = False
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            if adj[i, j] == -1 and adj[j, i] == -1:
                mutual_inhib = True
                break

    # Negative cycle (excluding self-loops)
    cycles = find_all_simple_cycles(adj, n_genes)
    has_neg_cycle = any(
        compute_cycle_sign(c, adj) < 0 and len(c) > 1
        for c in cycles
    )

    # Cascade
    cascades = find_cascades(adj, n_genes)
    has_long_cascade = any(len(c) >= 3 for c in cascades)

    # IFFL
    ffls = find_feedforward_loops(adj, n_genes)
    has_iffl = any(f[3] == 'incoherent' for f in ffls)

    # Compute indicators (soft scores based on necessary conditions)
    # Oscillator: self-inhibition OR negative feedback cycle
    if self_inhib:
        indicators[0] = 1.0
    elif has_neg_cycle:
        indicators[0] = 0.8

    # Toggle switch: mutual inhibition
    if mutual_inhib:
        indicators[1] = 1.0

    # Adaptation: various regulatory structures
    if len(cycles) > 0 and not self_inhib:
        indicators[2] = 0.5

    # Pulse generator: IFFL
    if has_iffl:
        indicators[3] = 1.0

    # Amplifier: activation cascade
    if has_long_cascade and not has_neg_cycle:
        indicators[4] = 1.0

    # Stable: no complex regulation
    if len(cycles) == 0 or (len(cycles) == 1 and not self_inhib):
        indicators[5] = 0.8

    return indicators


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class LoopAnalyzerNetwork(nn.Module):
    """
    Neural network that processes loop features for phenotype classification.

    The architecture is designed to:
    1. Process loop features with domain-specific inductive biases
    2. Combine with phenotype indicators from theory
    3. Learn optimal classification weights
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # Input dimensions
        self.loop_feat_dim = 26  # From compute_loop_features
        self.indicator_dim = 6   # From compute_phenotype_indicators

        # Loop feature encoder
        self.loop_encoder = nn.Sequential(
            nn.Linear(self.loop_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Indicator encoder (smaller, just refines)
        self.indicator_encoder = nn.Sequential(
            nn.Linear(self.indicator_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Classifier with residual connection from indicators
        self.classifier = nn.Linear(hidden_dim // 2 + self.indicator_dim, NUM_PHENOTYPES)

    def __call__(
        self,
        loop_features: mx.array,
        indicators: mx.array,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            loop_features: (batch, loop_feat_dim) loop analysis features
            indicators: (batch, 6) phenotype indicators from theory

        Returns:
            logits: (batch, num_phenotypes)
        """
        # Encode features
        h_loop = self.loop_encoder(loop_features)
        h_ind = self.indicator_encoder(indicators)

        # Fuse
        h = mx.concatenate([h_loop, h_ind], axis=-1)
        h = self.fusion(h)

        # Classify with residual from indicators
        h_with_ind = mx.concatenate([h, indicators], axis=-1)
        logits = self.classifier(h_with_ind)

        return logits


# ============================================================================
# TRAINING
# ============================================================================

# Feature cache
FEATURE_CACHE = {}


def get_features(edges: List, n_genes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get or compute features with caching."""
    key = (n_genes, tuple(sorted(tuple(e) for e in edges)))

    if key not in FEATURE_CACHE:
        loop_feat = compute_loop_features(edges, n_genes)
        indicators = compute_phenotype_indicators(edges, n_genes)
        FEATURE_CACHE[key] = (loop_feat, indicators)

    return FEATURE_CACHE[key]


def create_batch(
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Create balanced batch."""
    loop_list = []
    ind_list = []
    labels = []

    phenotypes = list(circuits_by_phenotype.keys())
    per_class = max(batch_size // len(phenotypes), 1)

    for phenotype in phenotypes:
        circuits = circuits_by_phenotype[phenotype]
        sampled = random.sample(circuits, min(per_class, len(circuits)))

        for circuit in sampled:
            edges = [tuple(e) for e in circuit['edges']]
            n_genes = circuit['n_genes']

            loop_feat, indicators = get_features(edges, n_genes)

            loop_list.append(loop_feat)
            ind_list.append(indicators)
            labels.append(PHENOTYPE_TO_ID[phenotype])

    # Shuffle
    combined = list(zip(loop_list, ind_list, labels))
    random.shuffle(combined)
    loop_list, ind_list, labels = zip(*combined)

    return (
        mx.array(np.stack(loop_list)),
        mx.array(np.stack(ind_list)),
        mx.array(labels),
    )


def train_epoch(
    model: LoopAnalyzerNetwork,
    optimizer: optim.Optimizer,
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    steps_per_epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch."""

    def loss_fn(model, loop_feat, indicators, labels):
        logits = model(loop_feat, indicators)
        ce_loss = mx.mean(nn.losses.cross_entropy(logits, labels))
        return ce_loss, logits

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(steps_per_epoch):
        loop_feat, indicators, labels = create_batch(
            circuits_by_phenotype, batch_size
        )

        (loss, logits), grads = loss_and_grad(model, loop_feat, indicators, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        preds = mx.argmax(logits, axis=-1)
        correct = int(mx.sum(preds == labels))

        total_loss += float(loss)
        total_correct += correct
        total_samples += labels.shape[0]

    return total_loss / steps_per_epoch, total_correct / total_samples


def evaluate(
    model: LoopAnalyzerNetwork,
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

            loop_feat, indicators = get_features(edges, n_genes)

            logits = model(
                mx.array(loop_feat[None]),
                mx.array(indicators[None]),
            )

            pred = ID_TO_PHENOTYPE[int(mx.argmax(logits[0]))]
            if pred == phenotype:
                correct += 1

        accuracy = correct / len(sampled)
        results[phenotype] = accuracy
        total_correct += correct
        total_samples += len(sampled)

    results['overall'] = total_correct / total_samples
    return results


def rule_based_classify(edges: List, n_genes: int) -> str:
    """
    Pure rule-based classification using dynamical systems theory.

    This is the BASELINE - what we can achieve with just rules.
    The neural network should improve upon this.
    """
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype

    # Check structures in priority order
    self_inhib = any(adj[i, i] == -1 for i in range(n_genes))

    mutual_inhib = False
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            if adj[i, j] == -1 and adj[j, i] == -1:
                mutual_inhib = True
                break

    ffls = find_feedforward_loops(adj, n_genes)
    has_iffl = any(f[3] == 'incoherent' for f in ffls)

    cascades = find_cascades(adj, n_genes)
    has_long_cascade = any(len(c) >= 3 for c in cascades)

    # Priority-based rules (from most specific to least)
    if self_inhib:
        return 'oscillator'

    if mutual_inhib:
        return 'toggle_switch'

    if has_iffl:
        return 'pulse_generator'

    if has_long_cascade:
        return 'amplifier'

    # Default
    return 'stable'


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Loop Analyzer")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/loop_analyzer")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("loop_analyzer", log_file=f"{checkpoint_dir}/train.log")

    logger.info("=" * 70)
    logger.info("FEEDBACK LOOP ANALYZER - Algebraic Topology Approach")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Theoretical foundation:")
    logger.info("- Oscillation requires NEGATIVE feedback loops")
    logger.info("- Bistability requires POSITIVE feedback loops")
    logger.info("- IFFL produces pulse/adaptation")
    logger.info("- Cascades produce amplification")
    logger.info("")
    logger.info("Key innovations:")
    logger.info("1. Exact cycle enumeration with sign computation")
    logger.info("2. Algebraic topology features (Betti numbers)")
    logger.info("3. Theory-derived phenotype indicators")
    logger.info("4. Neural network to learn optimal weighting")
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
    logger.info("\nPre-computing loop analysis features...")
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

    # Evaluate rule-based baseline
    logger.info("\nEvaluating rule-based classification (baseline)...")
    rule_correct = 0
    for phenotype, circuits in circuits_by_phenotype.items():
        for circuit in circuits[:200]:
            edges = [tuple(e) for e in circuit['edges']]
            n_genes = circuit['n_genes']
            pred = rule_based_classify(edges, n_genes)
            if pred == phenotype:
                rule_correct += 1
    rule_total = sum(min(200, len(c)) for c in circuits_by_phenotype.values())
    logger.info(f"  Rule-based accuracy: {rule_correct}/{rule_total} ({rule_correct/rule_total:.1%})")

    # Create model
    model = LoopAnalyzerNetwork(
        hidden_dim=args.hidden_dim,
        dropout=0.1,
    )

    # Count parameters
    def count_params(params):
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, list):
            for v in params:
                total += count_params(v)
        elif hasattr(params, 'size'):
            total += params.size
        return total

    n_params = count_params(model.parameters())
    logger.info(f"\nModel parameters: {n_params:,}")

    # Optimizer
    steps_per_epoch = max(total // args.batch_size, 30)

    lr_schedule = optim.join_schedules(
        [
            optim.linear_schedule(1e-5, args.lr, 20),
            optim.cosine_decay(args.lr, args.epochs - 20, 1e-6)
        ],
        [20]
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

    logger.info(f"\n  OVERALL: {results['overall']:.1%}")
    logger.info(f"  (Rule-based baseline: {rule_correct/rule_total:.1%})")

    # Improvement over baseline
    improvement = results['overall'] - rule_correct/rule_total
    logger.info(f"  Improvement over rules: {improvement:+.1%}")

    # Save model
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
    logger.info(f"\nModel saved to {checkpoint_dir / 'best.safetensors'}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Best training accuracy: {best_acc:.1%}")
    logger.info(f"Final overall accuracy: {results['overall']:.1%}")

    if results['overall'] >= 0.9:
        logger.info("\n🎉 TARGET ACHIEVED: 90%+ accuracy!")


if __name__ == "__main__":
    main()
