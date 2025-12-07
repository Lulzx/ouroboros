#!/usr/bin/env python3
"""
Enhanced GNN Training Script v4

Building on v3's success (66.7% accuracy), this version adds:
1. Structural motif features as INPUTS (not auxiliary losses)
2. Graph-level properties (cycles, feedback loops)
3. Larger capacity model
4. Better initialization
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


def compute_structural_features(adj: np.ndarray, n_genes: int) -> np.ndarray:
    """
    Compute structural features that are predictive of phenotype.

    These encode domain knowledge about what structures produce what behaviors:
    - Mutual inhibition → toggle switch
    - Negative feedback loops → oscillator
    - IFFL motifs → pulse generator
    - Cascades → amplifier
    """
    features = []

    # Clip to actual genes
    adj_real = adj[:n_genes, :n_genes]

    # Binary versions
    has_edge = np.abs(adj_real) > 0.5
    has_activation = adj_real > 0.5
    has_inhibition = adj_real < -0.5

    # 1. Mutual inhibition: A inhibits B AND B inhibits A
    mutual_inhib = has_inhibition & has_inhibition.T
    has_mutual_inhib = float(np.any(mutual_inhib))
    n_mutual_inhib = float(np.sum(mutual_inhib) / 2)  # Each pair counted twice
    features.extend([has_mutual_inhib, n_mutual_inhib])

    # 2. Self-loops
    self_act = float(np.sum(np.diag(has_activation)))
    self_inh = float(np.sum(np.diag(has_inhibition)))
    features.extend([self_act, self_inh])

    # 3. Cycle detection via matrix powers
    adj_binary = has_edge.astype(np.float32)
    has_cycle = 0.0
    power = adj_binary.copy()
    for k in range(1, n_genes + 1):
        if np.any(np.diag(power) > 0):
            has_cycle = 1.0
            break
        power = power @ adj_binary

    features.append(has_cycle)

    # 4. Negative feedback loop (odd inhibitions in cycle)
    # Simplified: check for cycles with at least one inhibition
    neg_edges = has_inhibition.astype(np.float32)
    adj_with_neg = adj_binary.copy()
    has_neg_feedback = 0.0
    power = adj_with_neg.copy()
    neg_power = neg_edges.copy()
    for k in range(1, n_genes + 1):
        # Check if there's a cycle that includes inhibition
        if np.any((np.diag(power) > 0) & (np.diag(neg_power) > 0)):
            has_neg_feedback = 1.0
            break
        power = power @ adj_with_neg
        neg_power = neg_power @ adj_with_neg + power @ neg_edges

    features.append(has_neg_feedback)

    # 5. IFFL detection: A→B, A→C, B⊣C (incoherent feed-forward loop)
    has_iffl = 0.0
    for a in range(n_genes):
        for b in range(n_genes):
            if a == b:
                continue
            for c in range(n_genes):
                if c == a or c == b:
                    continue
                # Check A→B, A→C (or A⊣C), B⊣C
                if has_activation[a, b] and has_edge[a, c] and has_inhibition[b, c]:
                    has_iffl = 1.0
                    break
            if has_iffl:
                break
        if has_iffl:
            break

    features.append(has_iffl)

    # 6. Cascade detection: A→B→C with no feedback
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

    # 7. Degree statistics
    in_deg = np.sum(has_edge, axis=0)
    out_deg = np.sum(has_edge, axis=1)
    features.extend([
        float(np.max(in_deg)) if n_genes > 0 else 0.0,
        float(np.max(out_deg)) if n_genes > 0 else 0.0,
        float(np.sum(has_edge)),  # Total edges
    ])

    # 8. Edge type balance
    n_act = float(np.sum(has_activation))
    n_inh = float(np.sum(has_inhibition))
    total = n_act + n_inh + 1e-8
    features.extend([n_act / total, n_inh / total])

    return np.array(features, dtype=np.float32)


class EnhancedGNN(nn.Module):
    """
    Enhanced GNN with structural features as inputs.
    """

    def __init__(
        self,
        max_genes: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_classes: int = 6,
        struct_feat_dim: int = 13,  # Number of structural features
    ):
        super().__init__()
        self.max_genes = max_genes
        self.hidden_dim = hidden_dim
        self.struct_feat_dim = struct_feat_dim

        # Node embedding
        self.node_embed = nn.Linear(max_genes, hidden_dim)

        # Message passing layers
        self.mp_layers = []
        for _ in range(num_layers):
            self.mp_layers.append(MPLayer(hidden_dim))

        # Combine graph embedding with structural features
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + struct_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def __call__(
        self,
        adj_matrix: mx.array,
        node_mask: mx.array,
        struct_features: mx.array,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            adj_matrix: (B, max_genes, max_genes) signed adjacency
            node_mask: (B, max_genes) binary mask
            struct_features: (B, struct_feat_dim) structural features
        """
        B, N, _ = adj_matrix.shape

        # Initial node features
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

        # Concatenate with structural features
        h_graph = mx.concatenate([h_mean, h_max, struct_features], axis=-1)

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

    def __call__(
        self,
        h: mx.array,
        adj_act: mx.array,
        adj_inh: mx.array,
        node_mask: mx.array,
    ) -> mx.array:
        msg_act = self.msg_act(h)
        agg_act = mx.matmul(adj_act, msg_act)

        msg_inh = self.msg_inh(h)
        agg_inh = mx.matmul(adj_inh, msg_inh)

        combined = mx.concatenate([h, agg_act, agg_inh], axis=-1)
        h_new = self.node_update(combined)
        h_new = self.norm(h + h_new)
        h_new = h_new * node_mask[:, :, None]

        return h_new


def topology_to_adj_and_features(
    edges: List,
    n_genes: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Convert topology to adjacency matrix and structural features."""
    adj = np.zeros((max_genes, max_genes), dtype=np.float32)

    for src, tgt, etype in edges:
        adj[src, tgt] = float(etype)

    mask = np.zeros(max_genes, dtype=np.float32)
    mask[:n_genes] = 1.0

    struct_feat = compute_structural_features(adj, n_genes)

    return mx.array(adj), mx.array(mask), mx.array(struct_feat)


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
            adj, mask, feat = topology_to_adj_and_features(
                circuit['edges'],
                circuit['n_genes'],
                max_genes,
            )
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
    model: EnhancedGNN,
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
    model: EnhancedGNN,
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
            adj, mask, feat = topology_to_adj_and_features(
                circuit['edges'], circuit['n_genes'], max_genes
            )
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
    model: EnhancedGNN,
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
    oracle_matches_label = 0
    total = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))

        for circuit in sampled:
            adj, mask, feat = topology_to_adj_and_features(
                circuit['edges'], circuit['n_genes'], max_genes
            )

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
            if oracle_pred == phenotype:
                oracle_matches_label += 1
            total += 1

    results = {
        'model_oracle_agreement': model_matches_oracle / total,
        'model_label_agreement': model_matches_label / total,
        'oracle_label_agreement': oracle_matches_label / total,
    }

    if logger:
        logger.info(f"  Model-Oracle: {results['model_oracle_agreement']:.1%}")
        logger.info(f"  Model-Label: {results['model_label_agreement']:.1%}")
        logger.info(f"  Oracle-Label: {results['oracle_label_agreement']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced GNN v4")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gnn_v4")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("gnn_v4", log_file=f"{checkpoint_dir}/train.log")

    # Load data
    logger.info(f"Loading circuits from {args.verified_circuits}")
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    # Create model
    model = EnhancedGNN(
        max_genes=args.max_genes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    logger.info("Model created")

    # Optimizer with warmup
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
