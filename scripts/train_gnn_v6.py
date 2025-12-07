#!/usr/bin/env python3
"""
GNN v6: Multi-Task Learning with Dynamics Prediction

The insight: Models plateau at ~68% because phenotype depends on DYNAMICS,
not just topology. Two topologically similar circuits can have different
phenotypes if they differ in subtle ways.

This version uses multi-task learning:
1. PRIMARY TASK: Predict phenotype (classification)
2. AUXILIARY TASK: Predict simulation outcomes (regression)

The auxiliary task forces the model to learn dynamics-relevant features,
which should transfer to the primary classification task.

Additionally, we use a more sophisticated approach:
- Focus training on hard examples (curriculum mining)
- Use focal loss to handle class imbalance dynamically
- Ensemble multiple heads for robustness
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


def simulate_and_get_labels(edges: List, n_genes: int, max_steps: int = 50) -> Dict[str, float]:
    """
    Run simulation and extract labels for auxiliary prediction.
    """
    adj = np.zeros((n_genes, n_genes), dtype=np.int32)
    for src, tgt, etype in edges:
        adj[src, tgt] = etype

    # Multiple random starts
    n_trials = 5
    cycle_detected = []
    settled = []
    cycle_lengths = []
    final_activities = []

    for _ in range(n_trials):
        state = np.random.randint(0, 2, n_genes).astype(np.float32)
        trajectory = [tuple(state)]

        for step in range(max_steps):
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

            state_tuple = tuple(state)
            if state_tuple in trajectory:
                idx = trajectory.index(state_tuple)
                cycle_len = len(trajectory) - idx
                cycle_detected.append(1.0)
                cycle_lengths.append(cycle_len)
                break
            trajectory.append(state_tuple)
        else:
            cycle_detected.append(0.0)
            cycle_lengths.append(0)

        if len(trajectory) > 1 and trajectory[-1] == trajectory[-2]:
            settled.append(1.0)
        else:
            settled.append(0.0)

        final_activities.append(np.mean(state))

    return {
        'has_cycle': np.mean(cycle_detected),
        'has_fixed_point': np.mean(settled),
        'avg_cycle_length': np.mean(cycle_lengths) / max_steps if cycle_lengths else 0.0,
        'avg_final_activity': np.mean(final_activities),
        'is_oscillator': 1.0 if np.mean(cycle_detected) > 0.5 and np.mean(settled) < 0.5 else 0.0,
    }


def compute_structural_features(adj: np.ndarray, n_genes: int) -> np.ndarray:
    """Compute structural features."""
    features = []
    adj_real = adj[:n_genes, :n_genes]

    has_edge = np.abs(adj_real) > 0.5
    has_activation = adj_real > 0.5
    has_inhibition = adj_real < -0.5

    # Mutual inhibition
    mutual_inhib = has_inhibition & has_inhibition.T
    features.append(float(np.any(mutual_inhib)))

    # Self-loops
    features.append(float(np.sum(np.diag(has_inhibition))))

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

    # Edge statistics
    n_act = float(np.sum(has_activation))
    n_inh = float(np.sum(has_inhibition))
    total = n_act + n_inh + 1e-8
    features.extend([n_act / total, n_inh / total, float(n_genes), float(np.sum(has_edge))])

    return np.array(features, dtype=np.float32)


class MultiTaskGNN(nn.Module):
    """
    Multi-task GNN that jointly predicts phenotype and simulation outcomes.
    """

    def __init__(
        self,
        max_genes: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_classes: int = 6,
        num_sim_targets: int = 5,
        struct_feat_dim: int = 7,
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

        # Shared backbone after pooling
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim * 2 + struct_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification head with multiple independent classifiers (ensemble)
        self.class_head1 = nn.Linear(hidden_dim, num_classes)
        self.class_head2 = nn.Linear(hidden_dim, num_classes)
        self.class_head3 = nn.Linear(hidden_dim, num_classes)

        # Simulation prediction head
        self.sim_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_sim_targets),
            nn.Sigmoid(),  # All targets are in [0, 1]
        )

    def __call__(
        self,
        adj_matrix: mx.array,
        node_mask: mx.array,
        struct_features: mx.array,
    ) -> Dict[str, mx.array]:
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

        # Combine with structural features
        h_graph = mx.concatenate([h_mean, h_max, struct_features], axis=-1)

        # Shared backbone
        h_shared = self.backbone(h_graph)

        # Multiple classification heads (ensemble)
        logits1 = self.class_head1(h_shared)
        logits2 = self.class_head2(h_shared)
        logits3 = self.class_head3(h_shared)

        # Average logits for final prediction
        logits_avg = (logits1 + logits2 + logits3) / 3.0

        # Simulation predictions
        sim_pred = self.sim_head(h_shared)

        return {
            'logits': logits_avg,
            'logits1': logits1,
            'logits2': logits2,
            'logits3': logits3,
            'sim_pred': sim_pred,
        }


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


# Cache for simulation labels
SIM_CACHE = {}


def get_circuit_data(
    circuit: Dict,
    max_genes: int,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Get adjacency, mask, structural features, and simulation labels."""
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

    # Simulation labels (cached)
    edge_key = tuple(tuple(e) for e in edges)
    if edge_key not in SIM_CACHE:
        sim_labels = simulate_and_get_labels(edges, n_genes)
        SIM_CACHE[edge_key] = np.array([
            sim_labels['has_cycle'],
            sim_labels['has_fixed_point'],
            sim_labels['avg_cycle_length'],
            sim_labels['avg_final_activity'],
            sim_labels['is_oscillator'],
        ], dtype=np.float32)

    sim_labels = SIM_CACHE[edge_key]

    return mx.array(adj), mx.array(mask), mx.array(struct_feat), mx.array(sim_labels)


def focal_loss(logits, labels, gamma=2.0, alpha=None):
    """
    Focal loss for handling class imbalance.
    Reduces loss contribution from easy examples.
    """
    ce = nn.losses.cross_entropy(logits, labels, reduction='none')
    pt = mx.exp(-ce)  # Probability of correct class
    focal_weight = (1 - pt) ** gamma

    if alpha is not None:
        alpha_weight = alpha[labels]
        focal_weight = focal_weight * alpha_weight

    return mx.mean(focal_weight * ce)


def create_balanced_batch(
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Create a balanced training batch."""
    adj_list = []
    mask_list = []
    struct_list = []
    sim_list = []
    label_list = []

    phenotypes = [p for p in circuits_by_phenotype.keys() if circuits_by_phenotype[p]]
    samples_per_class = max(batch_size // len(phenotypes), 1)

    for phenotype in phenotypes:
        circuits = circuits_by_phenotype[phenotype]
        n_samples = min(samples_per_class, len(circuits))
        sampled = random.sample(circuits, n_samples)

        for circuit in sampled:
            adj, mask, struct_feat, sim_labels = get_circuit_data(circuit, max_genes)
            adj_list.append(adj)
            mask_list.append(mask)
            struct_list.append(struct_feat)
            sim_list.append(sim_labels)
            label_list.append(PHENOTYPE_TO_ID[phenotype])

    combined = list(zip(adj_list, mask_list, struct_list, sim_list, label_list))
    random.shuffle(combined)
    adj_list, mask_list, struct_list, sim_list, label_list = zip(*combined)

    return (
        mx.stack(adj_list),
        mx.stack(mask_list),
        mx.stack(struct_list),
        mx.stack(sim_list),
        mx.array(label_list),
    )


def train(
    model: MultiTaskGNN,
    optimizer: optim.Optimizer,
    circuits_by_phenotype: Dict[str, List],
    epochs: int,
    batch_size: int,
    max_genes: int,
    logger,
) -> float:
    """Train the model with multi-task learning."""
    total_circuits = sum(len(v) for v in circuits_by_phenotype.values())
    steps_per_epoch = max(total_circuits // batch_size, 20)

    logger.info(f"\nTraining: {epochs} epochs, {steps_per_epoch} steps/epoch")

    # Class weights for focal loss (inverse frequency)
    counts = [len(circuits_by_phenotype.get(ID_TO_PHENOTYPE[i], [])) + 1 for i in range(6)]
    total = sum(counts)
    alpha = mx.array([total / (6 * c) for c in counts])
    alpha = alpha / mx.sum(alpha) * 6  # Normalize

    def loss_fn(model, adj, mask, struct_feat, sim_labels, labels):
        outputs = model(adj, mask, struct_feat)

        # Classification loss (focal loss for each head)
        class_loss1 = focal_loss(outputs['logits1'], labels, gamma=2.0, alpha=alpha)
        class_loss2 = focal_loss(outputs['logits2'], labels, gamma=2.0, alpha=alpha)
        class_loss3 = focal_loss(outputs['logits3'], labels, gamma=2.0, alpha=alpha)
        class_loss = (class_loss1 + class_loss2 + class_loss3) / 3.0

        # Simulation prediction loss (MSE)
        sim_loss = mx.mean((outputs['sim_pred'] - sim_labels) ** 2)

        # Total loss
        total_loss = class_loss + 0.3 * sim_loss

        return total_loss, outputs

    best_acc = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step in range(steps_per_epoch):
            adj, mask, struct_feat, sim_labels, labels = create_balanced_batch(
                circuits_by_phenotype, batch_size, max_genes
            )

            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            (loss, outputs), grads = loss_and_grad_fn(
                model, adj, mask, struct_feat, sim_labels, labels
            )

            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)
            preds = mx.argmax(outputs['logits'], axis=-1)
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
    model: MultiTaskGNN,
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
            adj, mask, struct_feat, _ = get_circuit_data(circuit, max_genes)
            outputs = model(adj[None], mask[None], struct_feat[None])
            pred = int(mx.argmax(outputs['logits'][0]))

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
    model: MultiTaskGNN,
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
    total = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))

        for circuit in sampled:
            adj, mask, struct_feat, _ = get_circuit_data(circuit, max_genes)
            outputs = model(adj[None], mask[None], struct_feat[None])
            model_pred = ID_TO_PHENOTYPE[int(mx.argmax(outputs['logits'][0]))]

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
            total += 1

    agreement = model_matches_oracle / total
    if logger:
        logger.info(f"  Model-Oracle: {agreement:.1%}")

    return {'model_oracle_agreement': agreement}


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Task GNN v6")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gnn_v6")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("gnn_v6", log_file=f"{checkpoint_dir}/train.log")

    # Load data
    logger.info(f"Loading circuits from {args.verified_circuits}")
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    # Pre-compute simulation labels
    logger.info("\nPre-computing simulation labels...")
    total = sum(len(c) for c in circuits_by_phenotype.values())
    computed = 0
    for phenotype, circuits in circuits_by_phenotype.items():
        for circuit in circuits:
            _ = get_circuit_data(circuit, args.max_genes)
            computed += 1
            if computed % 2000 == 0:
                logger.info(f"  Computed {computed}/{total}")
    logger.info(f"  Done: {len(SIM_CACHE)} unique circuits")

    # Create model
    model = MultiTaskGNN(
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

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Best training accuracy: {best_acc:.1%}")
    logger.info(f"Final overall accuracy: {results['overall']:.1%}")
    logger.info(f"Model-Oracle agreement: {oracle_results['model_oracle_agreement']:.1%}")


if __name__ == "__main__":
    main()
