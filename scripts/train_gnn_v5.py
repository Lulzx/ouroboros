#!/usr/bin/env python3
"""
GNN v5: Simulation-Informed Features

Key insight: Use SIMULATION OUTPUT as INPUT FEATURES (not as auxiliary loss).

The previous approaches failed because:
- v1-v2: Auxiliary losses (dynamics, contrastive) interfered with classification
- v3-v4: Topology alone has limited predictive power for some phenotypes

This version:
1. Run a quick Boolean network simulation for each circuit
2. Extract dynamical features from the simulation trajectory
3. Use these as input features alongside structural features
4. Pure classification objective (no auxiliary losses)

The simulation features directly encode:
- Whether states repeat (cycles)
- Whether the system settles (fixed point)
- Trajectory variance (oscillation amplitude)
- Return behavior after perturbation (adaptation)
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
import mlx.utils
import numpy as np

from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier
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


def simulate_circuit(edges: List, n_genes: int, max_steps: int = 50) -> Dict[str, float]:
    """
    Run Boolean network simulation and extract dynamical features.

    This is the key innovation: we use simulation to generate INPUT features
    rather than using it as a loss function.
    """
    # Build adjacency
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype  # +1 or -1

    features = {
        'has_cycle': 0.0,
        'has_fixed_point': 0.0,
        'cycle_length': 0.0,
        'avg_variance': 0.0,
        'returns_to_start': 0.0,
        'n_unique_states': 0.0,
        'final_activity': 0.0,
        'perturbation_response': 0.0,
    }

    # Run from multiple initial conditions
    n_trials = 5
    all_cycles = []
    all_fixed = []
    all_unique = []
    all_final_act = []

    for trial in range(n_trials):
        # Random initial state
        state = np.random.randint(0, 2, n_genes).astype(np.float32)
        trajectory = [state.copy()]

        # Simulate
        for step in range(max_steps):
            next_state = np.zeros(n_genes, dtype=np.float32)
            for i in range(n_genes):
                # Compute input to gene i
                activation = 0.0
                inhibition = 0.0
                for j in range(n_genes):
                    if adj[j, i] == 1:
                        activation += state[j]
                    elif adj[j, i] == -1:
                        inhibition += state[j]

                # Constitutive rule
                if activation > 0 and inhibition == 0:
                    next_state[i] = 1.0
                elif inhibition > 0:
                    next_state[i] = 0.0
                else:
                    next_state[i] = state[i]  # Maintain

            state = next_state
            trajectory.append(state.copy())

            # Check for cycle
            for prev_idx, prev_state in enumerate(trajectory[:-1]):
                if np.array_equal(state, prev_state):
                    cycle_len = len(trajectory) - 1 - prev_idx
                    all_cycles.append(cycle_len)
                    break
            else:
                continue
            break

        # Analyze trajectory
        trajectory = np.array(trajectory)
        unique_states = len(set(tuple(s) for s in trajectory))
        all_unique.append(unique_states / max(len(trajectory), 1))

        if len(trajectory) > 1:
            # Check if fixed point (last states are same)
            if np.array_equal(trajectory[-1], trajectory[-2]):
                all_fixed.append(1.0)
            else:
                all_fixed.append(0.0)

        all_final_act.append(np.mean(trajectory[-1]))

    # Aggregate features
    if all_cycles:
        features['has_cycle'] = 1.0
        features['cycle_length'] = np.mean(all_cycles) / max_steps

    if all_fixed:
        features['has_fixed_point'] = np.mean(all_fixed)

    features['n_unique_states'] = np.mean(all_unique)
    features['final_activity'] = np.mean(all_final_act)

    # Test perturbation response
    state = np.zeros(n_genes, dtype=np.float32)
    baseline_trajectory = []
    for step in range(20):
        next_state = np.zeros(n_genes, dtype=np.float32)
        for i in range(n_genes):
            activation = sum(state[j] for j in range(n_genes) if adj[j, i] == 1)
            inhibition = sum(state[j] for j in range(n_genes) if adj[j, i] == -1)
            if activation > 0 and inhibition == 0:
                next_state[i] = 1.0
            elif inhibition > 0:
                next_state[i] = 0.0
            else:
                next_state[i] = state[i]
        state = next_state
        baseline_trajectory.append(np.mean(state))

    # Apply perturbation
    state[0] = 1.0 - state[0]  # Flip first gene
    perturbed_trajectory = []
    for step in range(30):
        next_state = np.zeros(n_genes, dtype=np.float32)
        for i in range(n_genes):
            activation = sum(state[j] for j in range(n_genes) if adj[j, i] == 1)
            inhibition = sum(state[j] for j in range(n_genes) if adj[j, i] == -1)
            if activation > 0 and inhibition == 0:
                next_state[i] = 1.0
            elif inhibition > 0:
                next_state[i] = 0.0
            else:
                next_state[i] = state[i]
        state = next_state
        perturbed_trajectory.append(np.mean(state))

    # Check if returns to baseline
    if baseline_trajectory and perturbed_trajectory:
        baseline_final = baseline_trajectory[-1] if baseline_trajectory else 0.0
        perturbed_final = perturbed_trajectory[-1] if perturbed_trajectory else 0.0
        features['returns_to_start'] = 1.0 if abs(perturbed_final - baseline_final) < 0.1 else 0.0
        features['perturbation_response'] = max(perturbed_trajectory) - baseline_final

    return features


def compute_structural_features(adj: np.ndarray, n_genes: int) -> np.ndarray:
    """Compute structural features."""
    features = []

    adj_real = adj[:n_genes, :n_genes]
    has_edge = np.abs(adj_real) > 0.5
    has_activation = adj_real > 0.5
    has_inhibition = adj_real < -0.5

    # Mutual inhibition
    mutual_inhib = has_inhibition & has_inhibition.T
    has_mutual_inhib = float(np.any(mutual_inhib))
    features.append(has_mutual_inhib)

    # Self-loops
    self_inh = float(np.sum(np.diag(has_inhibition)))
    features.append(self_inh)

    # Cycle detection
    adj_binary = has_edge.astype(np.float32)
    has_cycle = 0.0
    power = adj_binary.copy()
    for k in range(1, n_genes + 1):
        if np.any(np.diag(power) > 0):
            has_cycle = 1.0
            break
        power = power @ adj_binary
    features.append(has_cycle)

    # IFFL detection
    has_iffl = 0.0
    for a in range(n_genes):
        for b in range(n_genes):
            if a == b:
                continue
            for c in range(n_genes):
                if c == a or c == b:
                    continue
                if has_activation[a, b] and has_edge[a, c] and has_inhibition[b, c]:
                    has_iffl = 1.0
                    break
            if has_iffl:
                break
        if has_iffl:
            break
    features.append(has_iffl)

    # Cascade
    has_cascade = 0.0
    for a in range(n_genes):
        for b in range(n_genes):
            if a == b or not has_activation[a, b]:
                continue
            for c in range(n_genes):
                if c == a or c == b:
                    continue
                if has_activation[b, c] and not has_edge[c, a] and not has_edge[c, b]:
                    has_cascade = 1.0
                    break
            if has_cascade:
                break
        if has_cascade:
            break
    features.append(has_cascade)

    # Edge statistics
    n_act = float(np.sum(has_activation))
    n_inh = float(np.sum(has_inhibition))
    total = n_act + n_inh + 1e-8
    features.extend([n_act / total, n_inh / total, float(n_genes)])

    return np.array(features, dtype=np.float32)


class SimInformedGNN(nn.Module):
    """GNN with simulation-informed features."""

    def __init__(
        self,
        max_genes: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_classes: int = 6,
        struct_feat_dim: int = 8,
        sim_feat_dim: int = 8,
    ):
        super().__init__()
        self.max_genes = max_genes
        self.hidden_dim = hidden_dim

        # Node embedding
        self.node_embed = nn.Linear(max_genes, hidden_dim)

        # Message passing layers
        self.mp_layers = []
        for _ in range(num_layers):
            self.mp_layers.append(MPLayer(hidden_dim))

        # Feature processing
        total_feat_dim = hidden_dim * 2 + struct_feat_dim + sim_feat_dim
        self.pre_classifier = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def __call__(
        self,
        adj_matrix: mx.array,
        node_mask: mx.array,
        features: mx.array,  # Combined struct + sim features
    ) -> mx.array:
        B, N, _ = adj_matrix.shape

        # Node features
        pos_enc = mx.eye(N)
        pos_enc = mx.broadcast_to(pos_enc[None], (B, N, N))
        h = self.node_embed(pos_enc)

        # Separate adjacencies
        adj_act = mx.maximum(adj_matrix, 0.0)
        adj_inh = mx.maximum(-adj_matrix, 0.0)

        # Message passing
        for mp_layer in self.mp_layers:
            h = mp_layer(h, adj_act, adj_inh, node_mask)

        # Global pooling
        mask_expanded = node_mask[:, :, None]
        h_masked = h * mask_expanded

        h_sum = mx.sum(h_masked, axis=1)
        n_nodes = mx.sum(node_mask, axis=1, keepdims=True) + 1e-8
        h_mean = h_sum / n_nodes

        h_for_max = h_masked + (1 - mask_expanded) * (-1e9)
        h_max = mx.max(h_for_max, axis=1)

        # Combine with features
        h_graph = mx.concatenate([h_mean, h_max, features], axis=-1)

        # Classify
        h_pre = self.pre_classifier(h_graph)
        logits = self.classifier(h_pre)

        return logits


class MPLayer(nn.Module):
    """Message Passing Layer."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg_act = nn.Linear(hidden_dim, hidden_dim)
        self.msg_inh = nn.Linear(hidden_dim, hidden_dim)
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def __call__(self, h, adj_act, adj_inh, node_mask):
        msg_act = self.msg_act(h)
        agg_act = mx.matmul(adj_act, msg_act)
        msg_inh = self.msg_inh(h)
        agg_inh = mx.matmul(adj_inh, msg_inh)
        combined = mx.concatenate([h, agg_act, agg_inh], axis=-1)
        h_new = self.node_update(combined)
        h_new = self.norm(h + h_new)
        return h_new * node_mask[:, :, None]


# Cache for simulation features
SIM_CACHE = {}


def get_circuit_features(circuit: Dict, max_genes: int) -> Tuple[mx.array, mx.array, mx.array]:
    """Get adjacency, mask, and combined features for a circuit."""
    edges = circuit['edges']
    n_genes = circuit['n_genes']

    # Adjacency
    adj = np.zeros((max_genes, max_genes), dtype=np.float32)
    for src, tgt, etype in edges:
        adj[src, tgt] = float(etype)

    mask = np.zeros(max_genes, dtype=np.float32)
    mask[:n_genes] = 1.0

    # Structural features
    struct_feat = compute_structural_features(adj, n_genes)

    # Simulation features (cached)
    edge_key = tuple(tuple(e) for e in edges)
    if edge_key not in SIM_CACHE:
        sim_features = simulate_circuit(edges, n_genes)
        SIM_CACHE[edge_key] = np.array([
            sim_features['has_cycle'],
            sim_features['has_fixed_point'],
            sim_features['cycle_length'],
            sim_features['n_unique_states'],
            sim_features['final_activity'],
            sim_features['returns_to_start'],
            sim_features['perturbation_response'],
            1.0 if sim_features['has_cycle'] and not sim_features['has_fixed_point'] else 0.0,  # oscillator indicator
        ], dtype=np.float32)

    sim_feat = SIM_CACHE[edge_key]

    # Combine features
    combined = np.concatenate([struct_feat, sim_feat])

    return mx.array(adj), mx.array(mask), mx.array(combined)


def create_balanced_batch(
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Create a balanced training batch."""
    adj_list = []
    mask_list = []
    feat_list = []
    label_list = []

    phenotypes = [p for p in circuits_by_phenotype.keys() if circuits_by_phenotype[p]]
    samples_per_class = max(batch_size // len(phenotypes), 1)

    for phenotype in phenotypes:
        circuits = circuits_by_phenotype[phenotype]
        n_samples = min(samples_per_class, len(circuits))
        sampled = random.sample(circuits, n_samples)

        for circuit in sampled:
            adj, mask, feat = get_circuit_features(circuit, max_genes)
            adj_list.append(adj)
            mask_list.append(mask)
            feat_list.append(feat)
            label_list.append(PHENOTYPE_TO_ID[phenotype])

    combined = list(zip(adj_list, mask_list, feat_list, label_list))
    random.shuffle(combined)
    adj_list, mask_list, feat_list, label_list = zip(*combined)

    return (
        mx.stack(adj_list),
        mx.stack(mask_list),
        mx.stack(feat_list),
        mx.array(label_list),
    )


def train(
    model: SimInformedGNN,
    optimizer: optim.Optimizer,
    circuits_by_phenotype: Dict[str, List],
    epochs: int,
    batch_size: int,
    max_genes: int,
    logger,
) -> float:
    """Train the model."""
    total_circuits = sum(len(v) for v in circuits_by_phenotype.values())
    steps_per_epoch = max(total_circuits // batch_size, 20)

    logger.info(f"\nTraining: {epochs} epochs, {steps_per_epoch} steps/epoch")

    def loss_fn(model, adj, mask, feat, labels):
        logits = model(adj, mask, feat)
        return mx.mean(nn.losses.cross_entropy(logits, labels)), logits

    best_acc = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step in range(steps_per_epoch):
            adj, mask, feat, labels = create_balanced_batch(
                circuits_by_phenotype, batch_size, max_genes
            )

            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            (loss, logits), grads = loss_and_grad_fn(model, adj, mask, feat, labels)

            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)
            preds = mx.argmax(logits, axis=-1)
            epoch_correct += int(mx.sum(preds == labels))
            epoch_total += labels.shape[0]

        avg_loss = epoch_loss / steps_per_epoch
        accuracy = epoch_correct / epoch_total

        if accuracy > best_acc:
            best_acc = accuracy

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                f"acc={accuracy:.1%}, best={best_acc:.1%}"
            )

    return best_acc


def evaluate(
    model: SimInformedGNN,
    circuits_by_phenotype: Dict[str, List],
    max_genes: int,
    samples_per_phenotype: int = 200,
    logger=None,
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
            adj, mask, feat = get_circuit_features(circuit, max_genes)
            logits = model(adj[None], mask[None], feat[None])
            pred = int(mx.argmax(logits[0]))

            if ID_TO_PHENOTYPE[pred] == phenotype:
                correct += 1

        accuracy = correct / len(sampled)
        results[phenotype] = accuracy
        total_correct += correct
        total_samples += len(sampled)

        if logger:
            logger.info(f"  {phenotype}: {correct}/{len(sampled)} ({accuracy:.1%})")

    results['overall'] = total_correct / total_samples
    if logger:
        logger.info(f"  Overall: {total_correct}/{total_samples} ({results['overall']:.1%})")

    return results


def oracle_evaluate(
    model: SimInformedGNN,
    circuits_by_phenotype: Dict[str, List],
    max_genes: int,
    samples_per_phenotype: int = 100,
    logger=None,
) -> Dict[str, float]:
    """Evaluate using Boolean network oracle."""
    classifier = BehaviorClassifier(
        num_initial_conditions=20,
        max_steps=200,
        rule="constitutive",
    )

    model_matches_oracle = 0
    model_matches_label = 0
    total = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))

        for circuit in sampled:
            adj, mask, feat = get_circuit_features(circuit, max_genes)

            # Model prediction
            logits = model(adj[None], mask[None], feat[None])
            model_pred = ID_TO_PHENOTYPE[int(mx.argmax(logits[0]))]

            # Oracle prediction
            gene_names = [f"gene_{i}" for i in range(circuit['n_genes'])]
            interactions = []
            for src, tgt, etype in circuit['edges']:
                interactions.append({
                    'source': gene_names[src],
                    'target': gene_names[tgt],
                    'type': 'activates' if etype == 1 else 'inhibits',
                })

            network = BooleanNetwork.from_circuit({'interactions': interactions})
            oracle_pred, _ = classifier.classify(network)

            if model_pred == oracle_pred:
                model_matches_oracle += 1
            if model_pred == phenotype:
                model_matches_label += 1
            total += 1

    results = {
        'model_oracle_agreement': model_matches_oracle / total,
        'model_label_agreement': model_matches_label / total,
    }

    if logger:
        logger.info(f"  Model-Oracle: {results['model_oracle_agreement']:.1%}")
        logger.info(f"  Model-Label: {results['model_label_agreement']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Simulation-Informed GNN v5")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gnn_v5")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("gnn_v5", log_file=f"{checkpoint_dir}/train.log")

    # Load data
    logger.info(f"Loading circuits from {args.verified_circuits}")
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    # Pre-compute simulation features for all circuits
    logger.info("\nPre-computing simulation features...")
    total = sum(len(c) for c in circuits_by_phenotype.values())
    computed = 0
    for phenotype, circuits in circuits_by_phenotype.items():
        for circuit in circuits:
            _ = get_circuit_features(circuit, args.max_genes)
            computed += 1
            if computed % 2000 == 0:
                logger.info(f"  Computed {computed}/{total} features")
    logger.info(f"  Done: {len(SIM_CACHE)} unique circuits cached")

    # Create model
    model = SimInformedGNN(
        max_genes=args.max_genes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    logger.info("Model created")

    # Optimizer
    lr_schedule = optim.join_schedules(
        [optim.linear_schedule(1e-5, args.lr, 20),
         optim.cosine_decay(args.lr, args.epochs - 20)],
        [20]
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    # Train
    best_acc = train(
        model, optimizer, circuits_by_phenotype,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_genes=args.max_genes,
        logger=logger,
    )

    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("Classification Evaluation")
    logger.info("="*60)
    results = evaluate(model, circuits_by_phenotype, args.max_genes, logger=logger)

    # Oracle evaluation
    logger.info("\n" + "="*60)
    logger.info("Oracle Evaluation")
    logger.info("="*60)
    oracle_results = oracle_evaluate(model, circuits_by_phenotype, args.max_genes, logger=logger)

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
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Best training accuracy: {best_acc:.1%}")
    logger.info(f"Final overall accuracy: {results['overall']:.1%}")
    logger.info(f"Model-Oracle agreement: {oracle_results['model_oracle_agreement']:.1%}")


if __name__ == "__main__":
    main()
