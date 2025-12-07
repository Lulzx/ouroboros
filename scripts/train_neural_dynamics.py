#!/usr/bin/env python3
"""
Neural Dynamics Classifier (NDC) - Learning Topology to Dynamics Mapping

=== THE CORE INSIGHT ===

Previous approaches fail because they either:
1. Use topology features that don't capture dynamical invariants (GNN v1-v4)
2. Use exact dynamics features which defeats the purpose of learning (trivial)

The true challenge is: can we learn a neural network that PREDICTS dynamical
behavior from topology ALONE, without computing the full state transition graph?

=== KEY MATHEMATICAL INSIGHT ===

For Boolean networks, the dynamics are determined by:
1. The adjacency matrix A (topology)
2. The update rule f (fixed: constitutive rule)

The phenotype is a function of the attractor structure, which is DETERMINISTIC
given A and f. Therefore, there exists a function:

    g: A → phenotype

The question is: what is the best neural architecture to approximate g?

=== CRITICAL OBSERVATION ===

The dynamics can be computed by iterating the update rule:
    x_{t+1} = f(x_t, A)

If we make f differentiable (soft Boolean), we can:
1. Compute trajectories in a differentiable manner
2. Backpropagate through the dynamics
3. Learn features that predict phenotype from trajectory patterns

=== NOVEL APPROACH: NEURAL DIFFERENTIAL DYNAMICS ===

Instead of hand-crafting trajectory features, we:
1. Learn a DIFFERENTIABLE approximation of Boolean dynamics
2. Simulate for T steps with learnable initial conditions
3. Extract trajectory features with learned temporal encoder
4. Classify phenotype from trajectory representation

The key innovations:
1. LEARNABLE INITIALIZATION: Instead of random, learn optimal starting states
2. GRADIENT-BASED TRAJECTORY ENCODING: Learn to extract relevant patterns
3. MULTIPLE SIMULATION RUNS: Capture attractor diversity
4. ATTENTION OVER TRAJECTORIES: Weight by importance

=== ARCHITECTURE ===

1. Initial State Generator: A → {x_0^1, ..., x_0^K}
2. Neural Dynamics Simulator: (x_t, A) → x_{t+1}
3. Trajectory Encoder: [x_0, ..., x_T] → h_traj
4. Trajectory Aggregator: {h_1, ..., h_K} → h_circuit
5. Phenotype Classifier: h_circuit → logits
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

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
# NEURAL DYNAMICS COMPONENTS
# ============================================================================

class NeuralDynamics(nn.Module):
    """
    Differentiable Boolean network dynamics.

    The key is to make the discrete Boolean update DIFFERENTIABLE:
    - Use sigmoid with high temperature for thresholding
    - Use soft max/min for AND/OR operations

    The constitutive rule becomes:
    - x'[i] = sigmoid(temp * (activation[i] - inhibition[i] + bias[i]))

    Where:
    - activation[i] = sum_j A_act[j,i] * x[j]
    - inhibition[i] = sum_j A_inh[j,i] * x[j]
    - bias[i] depends on regulator types (constitutive expression)
    """

    def __init__(self, max_genes: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.max_genes = max_genes

        # Temperature for soft Boolean operations (learnable)
        self.temperature = 10.0  # High = more discrete

        # Learnable bias for constitutive expression
        # Genes with only inhibitors should have positive bias (default ON)
        self.bias_net = nn.Sequential(
            nn.Linear(max_genes * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_genes),
        )

    def compute_bias(self, adj: mx.array, node_mask: mx.array) -> mx.array:
        """
        Compute per-gene bias based on regulator structure.

        Args:
            adj: (batch, n, n) signed adjacency
            node_mask: (batch, n) mask

        Returns:
            bias: (batch, n) per-gene bias
        """
        B, n = node_mask.shape

        # Compute regulator statistics
        adj_act = mx.maximum(adj, 0)  # Activation edges
        adj_inh = mx.maximum(-adj, 0)  # Inhibition edges

        n_activators = mx.sum(adj_act, axis=1)  # (B, n)
        n_inhibitors = mx.sum(adj_inh, axis=1)  # (B, n)

        # Features for bias network
        features = mx.concatenate([
            n_activators,
            n_inhibitors,
            node_mask[:, :, None].squeeze(-1) if len(node_mask.shape) == 2 else node_mask,
        ], axis=1).reshape(B, -1)

        # Pad if needed
        if features.shape[1] < self.max_genes * 3:
            padding = mx.zeros((B, self.max_genes * 3 - features.shape[1]))
            features = mx.concatenate([features, padding], axis=1)

        bias = self.bias_net(features[:, :self.max_genes * 3])

        return bias

    def step(
        self,
        state: mx.array,
        adj: mx.array,
        node_mask: mx.array,
        bias: mx.array,
    ) -> mx.array:
        """
        One step of differentiable dynamics.

        Args:
            state: (batch, n) current state in [0, 1]
            adj: (batch, n, n) signed adjacency
            node_mask: (batch, n) mask
            bias: (batch, n) per-gene bias

        Returns:
            next_state: (batch, n) next state in [0, 1]
        """
        B, n = state.shape

        # Separate activation and inhibition
        adj_act = mx.maximum(adj, 0)
        adj_inh = mx.maximum(-adj, 0)

        # Compute incoming signals
        state_col = state[:, :, None]  # (B, n, 1)

        # activation[i] = sum_j adj_act[j,i] * state[j]
        activation = mx.matmul(adj_act.transpose(0, 2, 1), state_col).squeeze(-1)
        inhibition = mx.matmul(adj_inh.transpose(0, 2, 1), state_col).squeeze(-1)

        # Soft update: next_state = sigmoid(temp * (activation - inhibition + bias))
        logit = self.temperature * (activation - inhibition + bias)
        next_state = mx.sigmoid(logit)

        # Apply mask
        next_state = next_state * node_mask

        return next_state

    def simulate(
        self,
        adj: mx.array,
        node_mask: mx.array,
        initial_state: mx.array,
        n_steps: int = 30,
    ) -> mx.array:
        """
        Simulate dynamics for n_steps.

        Returns:
            trajectory: (batch, n_steps+1, n_genes) trajectory
        """
        bias = self.compute_bias(adj, node_mask)

        trajectory = [initial_state]
        state = initial_state

        for _ in range(n_steps):
            state = self.step(state, adj, node_mask, bias)
            trajectory.append(state)

        return mx.stack(trajectory, axis=1)


class InitialStateGenerator(nn.Module):
    """
    Generate K diverse initial states from adjacency matrix.

    The key insight: different initial states reveal different attractors.
    We learn to generate initial states that efficiently explore the state space.
    """

    def __init__(self, max_genes: int = 10, n_states: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.max_genes = max_genes
        self.n_states = n_states

        # Generate initial states from adjacency structure
        self.encoder = nn.Sequential(
            nn.Linear(max_genes * max_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.state_generators = [
            nn.Linear(hidden_dim, max_genes) for _ in range(n_states)
        ]

    def __call__(self, adj: mx.array, node_mask: mx.array) -> mx.array:
        """
        Generate diverse initial states.

        Args:
            adj: (batch, n, n) adjacency
            node_mask: (batch, n) mask

        Returns:
            initial_states: (batch, n_states, n) initial states
        """
        B = adj.shape[0]
        n = self.max_genes

        # Flatten adjacency
        adj_flat = adj.reshape(B, -1)

        # Pad if needed
        if adj_flat.shape[1] < n * n:
            padding = mx.zeros((B, n * n - adj_flat.shape[1]))
            adj_flat = mx.concatenate([adj_flat, padding], axis=1)

        # Encode
        h = self.encoder(adj_flat[:, :n * n])

        # Generate states
        states = []
        for gen in self.state_generators:
            state = mx.sigmoid(gen(h))  # In [0, 1]
            state = state * node_mask  # Mask inactive genes
            states.append(state)

        return mx.stack(states, axis=1)


class TrajectoryEncoder(nn.Module):
    """
    Encode a trajectory into a fixed-size representation.

    Uses a combination of:
    1. Temporal convolutions (local patterns)
    2. Attention (global patterns)
    3. Statistics (summary features)
    """

    def __init__(self, max_genes: int = 10, hidden_dim: int = 128):
        super().__init__()
        self.max_genes = max_genes
        self.hidden_dim = hidden_dim

        # Temporal convolutions for local patterns
        self.conv1 = nn.Conv1d(max_genes, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1)

        # Attention for global patterns
        self.attention = nn.MultiHeadAttention(hidden_dim // 2, num_heads=4)
        self.pos_embed = nn.Linear(1, hidden_dim // 2)

        # Final projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def __call__(self, trajectory: mx.array) -> mx.array:
        """
        Encode trajectory.

        Args:
            trajectory: (batch, T, n_genes) trajectory

        Returns:
            h: (batch, hidden_dim) trajectory embedding
        """
        B, T, n = trajectory.shape

        # Pad to max_genes if needed
        if n < self.max_genes:
            padding = mx.zeros((B, T, self.max_genes - n))
            trajectory = mx.concatenate([trajectory, padding], axis=-1)

        # Temporal convolutions: (B, T, n) -> (B, n, T) for conv1d
        x = trajectory.transpose(0, 2, 1)  # (B, n, T)
        x = nn.relu(self.conv1(x))  # (B, hidden//2, T)
        x = nn.relu(self.conv2(x))  # (B, hidden//2, T)
        x = x.transpose(0, 2, 1)  # (B, T, hidden//2)

        # Add positional encoding
        positions = mx.arange(T).reshape(1, T, 1).astype(mx.float32) / T
        positions = mx.broadcast_to(positions, (B, T, 1))
        pos_enc = self.pos_embed(positions)
        x = x + pos_enc

        # Self-attention
        x_attn = self.attention(x, x, x)
        x = x + x_attn

        # Pool: concatenate mean and max
        h_mean = mx.mean(x, axis=1)  # (B, hidden//2)
        h_max = mx.max(x, axis=1)  # (B, hidden//2)
        h = mx.concatenate([h_mean, h_max], axis=-1)  # (B, hidden)

        return self.output(h)


class TrajectorySetAggregator(nn.Module):
    """
    Aggregate multiple trajectory embeddings into a single circuit embedding.

    Uses attention to weight trajectories by their importance.
    Different attractors should contribute differently to classification.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention over trajectories
        self.attention = nn.MultiHeadAttention(hidden_dim, num_heads=4)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def __call__(self, trajectory_embeddings: mx.array) -> mx.array:
        """
        Aggregate trajectory embeddings.

        Args:
            trajectory_embeddings: (batch, K, hidden) K trajectory embeddings

        Returns:
            h: (batch, hidden) circuit embedding
        """
        B, K, D = trajectory_embeddings.shape

        # Self-attention over trajectories
        x = self.attention(trajectory_embeddings, trajectory_embeddings, trajectory_embeddings)

        # Pool: mean and max
        h_mean = mx.mean(x, axis=1)
        h_max = mx.max(x, axis=1)
        h = mx.concatenate([h_mean, h_max], axis=-1)

        return self.output(h)


class NeuralDynamicsClassifier(nn.Module):
    """
    Full model: Neural Dynamics Classifier.

    Pipeline:
    1. Generate K initial states from adjacency
    2. Simulate dynamics for T steps from each initial state
    3. Encode each trajectory
    4. Aggregate trajectory embeddings
    5. Classify phenotype
    """

    def __init__(
        self,
        max_genes: int = 10,
        n_initial_states: int = 8,
        n_sim_steps: int = 30,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.max_genes = max_genes
        self.n_initial_states = n_initial_states
        self.n_sim_steps = n_sim_steps

        # Components
        self.init_generator = InitialStateGenerator(max_genes, n_initial_states, hidden_dim // 2)
        self.dynamics = NeuralDynamics(max_genes, hidden_dim // 2)
        self.traj_encoder = TrajectoryEncoder(max_genes, hidden_dim)
        self.aggregator = TrajectorySetAggregator(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, NUM_PHENOTYPES),
        )

        # Auxiliary heads
        self.aux_oscillation = nn.Linear(hidden_dim, 1)
        self.aux_bistable = nn.Linear(hidden_dim, 1)

    def __call__(
        self,
        adj: mx.array,
        node_mask: mx.array,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Forward pass.

        Args:
            adj: (batch, max_genes, max_genes) signed adjacency
            node_mask: (batch, max_genes) mask

        Returns:
            logits: (batch, num_phenotypes)
            aux: dict of auxiliary predictions
        """
        B = adj.shape[0]
        n = adj.shape[1]

        # Pad to max_genes if needed
        if n < self.max_genes:
            adj_padded = mx.zeros((B, self.max_genes, self.max_genes))
            adj_padded = adj_padded.at[:, :n, :n].add(adj)
            adj = adj_padded

            mask_padded = mx.zeros((B, self.max_genes))
            mask_padded = mask_padded.at[:, :n].add(node_mask)
            node_mask = mask_padded

        # Generate initial states: (B, K, n)
        initial_states = self.init_generator(adj, node_mask)

        # Simulate each initial state
        trajectory_embeddings = []
        for k in range(self.n_initial_states):
            # Simulate: (B, T+1, n)
            traj = self.dynamics.simulate(
                adj, node_mask,
                initial_states[:, k, :],
                self.n_sim_steps
            )

            # Encode: (B, hidden)
            h_traj = self.traj_encoder(traj)
            trajectory_embeddings.append(h_traj)

        # Stack: (B, K, hidden)
        trajectory_embeddings = mx.stack(trajectory_embeddings, axis=1)

        # Aggregate: (B, hidden)
        h_circuit = self.aggregator(trajectory_embeddings)

        # Classify
        logits = self.classifier(h_circuit)

        aux = {
            'oscillation': mx.sigmoid(self.aux_oscillation(h_circuit)),
            'bistable': mx.sigmoid(self.aux_bistable(h_circuit)),
        }

        return logits, aux


# ============================================================================
# STRUCTURAL FEATURES (for comparison)
# ============================================================================

def compute_structural_features(edges: List, n_genes: int) -> np.ndarray:
    """Compute structural features from topology."""
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype

    features = []

    # Basic counts
    n_edges = len(edges)
    n_act = sum(1 for _, _, e in edges if e == 1)
    n_inh = sum(1 for _, _, e in edges if e == -1)

    features.extend([n_genes, n_edges, n_act, n_inh])

    # Self-loops
    self_act = sum(1 for s, t, e in edges if s == t and e == 1)
    self_inh = sum(1 for s, t, e in edges if s == t and e == -1)
    features.extend([self_act, self_inh])

    # Mutual inhibition
    mutual = 0
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            if adj[i,j] == -1 and adj[j,i] == -1:
                mutual += 1
    features.append(mutual)

    # Cascade
    cascade = 0
    for i in range(n_genes):
        for j in range(n_genes):
            if i != j and adj[i,j] == 1:
                for k in range(n_genes):
                    if k != i and k != j and adj[j,k] == 1:
                        cascade += 1
    features.append(min(cascade, 5))

    return np.array(features, dtype=np.float32)


# ============================================================================
# TRAINING
# ============================================================================

def create_batch(
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Create balanced batch."""
    adj_list = []
    mask_list = []
    labels = []

    phenotypes = list(circuits_by_phenotype.keys())
    per_class = max(batch_size // len(phenotypes), 1)

    for phenotype in phenotypes:
        circuits = circuits_by_phenotype[phenotype]
        sampled = random.sample(circuits, min(per_class, len(circuits)))

        for circuit in sampled:
            edges = circuit['edges']
            n_genes = circuit['n_genes']

            adj = np.zeros((max_genes, max_genes), dtype=np.float32)
            for src, tgt, etype in edges:
                adj[src, tgt] = float(etype)

            mask = np.zeros(max_genes, dtype=np.float32)
            mask[:n_genes] = 1.0

            adj_list.append(adj)
            mask_list.append(mask)
            labels.append(PHENOTYPE_TO_ID[phenotype])

    # Shuffle
    combined = list(zip(adj_list, mask_list, labels))
    random.shuffle(combined)
    adj_list, mask_list, labels = zip(*combined)

    return (
        mx.array(np.stack(adj_list)),
        mx.array(np.stack(mask_list)),
        mx.array(labels),
    )


def train_epoch(
    model: NeuralDynamicsClassifier,
    optimizer: optim.Optimizer,
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    max_genes: int,
    steps_per_epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch."""

    def loss_fn(model, adj, mask, labels):
        logits, aux = model(adj, mask)
        ce_loss = mx.mean(nn.losses.cross_entropy(logits, labels))
        return ce_loss, logits

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for step in range(steps_per_epoch):
        adj, mask, labels = create_batch(
            circuits_by_phenotype, batch_size, max_genes
        )

        (loss, logits), grads = loss_and_grad(model, adj, mask, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        preds = mx.argmax(logits, axis=-1)
        correct = int(mx.sum(preds == labels))

        total_loss += float(loss)
        total_correct += correct
        total_samples += labels.shape[0]

    return total_loss / steps_per_epoch, total_correct / total_samples


def evaluate(
    model: NeuralDynamicsClassifier,
    circuits_by_phenotype: Dict[str, List],
    max_genes: int,
    samples_per_phenotype: int = 200,
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
            edges = circuit['edges']
            n_genes = circuit['n_genes']

            adj = np.zeros((max_genes, max_genes), dtype=np.float32)
            for src, tgt, etype in edges:
                adj[src, tgt] = float(etype)

            mask = np.zeros(max_genes, dtype=np.float32)
            mask[:n_genes] = 1.0

            logits, _ = model(
                mx.array(adj[None]),
                mx.array(mask[None]),
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Neural Dynamics Classifier")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/neural_dynamics")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-initial-states", type=int, default=8)
    parser.add_argument("--n-sim-steps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("neural_dynamics", log_file=f"{checkpoint_dir}/train.log")

    logger.info("=" * 70)
    logger.info("NEURAL DYNAMICS CLASSIFIER")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Key insight: Learn DIFFERENTIABLE dynamics simulation")
    logger.info("Then classify phenotype from learned trajectory patterns")
    logger.info("")
    logger.info("Architecture:")
    logger.info("1. Initial State Generator: adjacency → diverse starting states")
    logger.info("2. Neural Dynamics: differentiable Boolean network simulation")
    logger.info("3. Trajectory Encoder: temporal patterns → embedding")
    logger.info("4. Trajectory Aggregator: multiple trajectories → circuit embedding")
    logger.info("5. Phenotype Classifier: embedding → class logits")
    logger.info("")

    # Load data
    logger.info(f"Loading circuits from {args.verified_circuits}")
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    total = sum(len(c) for c in circuits_by_phenotype.values())

    # Create model
    model = NeuralDynamicsClassifier(
        max_genes=args.max_genes,
        n_initial_states=args.n_initial_states,
        n_sim_steps=args.n_sim_steps,
        hidden_dim=args.hidden_dim,
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
            args.batch_size, args.max_genes, steps_per_epoch
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

    results = evaluate(model, circuits_by_phenotype, args.max_genes, samples_per_phenotype=500)

    for phenotype in sorted(results.keys()):
        if phenotype != 'overall':
            logger.info(f"  {phenotype}: {results[phenotype]:.1%}")

    logger.info(f"\n  OVERALL: {results['overall']:.1%}")

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
