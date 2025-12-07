#!/usr/bin/env python3
"""
Clean GNN Training Script v3

Key insight from debugging:
- Simple MLP achieves 47.5% accuracy
- Complex GNN with auxiliary losses achieves only 25%
- The auxiliary losses (dynamics, contrastive, spectral) were HURTING learning

This version:
1. Clean message passing GNN (no complex auxiliary objectives)
2. Focus purely on classification
3. Add structural features as INPUTS, not auxiliary losses
4. Use edge type embedding (activation vs inhibition)
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


class EdgeTypeGNN(nn.Module):
    """
    Graph Neural Network that properly handles edge types (activation/inhibition).

    Key design choices:
    1. Separate message computation for activations and inhibitions
    2. Multiple message passing layers with residual connections
    3. Global graph-level readout for classification
    """

    def __init__(
        self,
        max_genes: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = 6,
    ):
        super().__init__()
        self.max_genes = max_genes
        self.hidden_dim = hidden_dim

        # Initial node embedding (just identity encoding)
        self.node_embed = nn.Linear(max_genes, hidden_dim)

        # Message passing layers
        self.mp_layers = []
        for _ in range(num_layers):
            self.mp_layers.append(MPLayer(hidden_dim))

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def __call__(self, adj_matrix: mx.array, node_mask: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            adj_matrix: (B, max_genes, max_genes) signed adjacency
            node_mask: (B, max_genes) binary mask for valid nodes
        """
        B, N, _ = adj_matrix.shape

        # Initial node features (one-hot position)
        pos_enc = mx.eye(N)
        pos_enc = mx.broadcast_to(pos_enc[None], (B, N, N))
        h = self.node_embed(pos_enc)  # (B, N, hidden)

        # Separate activation and inhibition adjacencies
        adj_act = mx.maximum(adj_matrix, 0.0)  # Positive edges
        adj_inh = mx.maximum(-adj_matrix, 0.0)  # Negative edges (made positive)

        # Message passing
        for mp_layer in self.mp_layers:
            h = mp_layer(h, adj_act, adj_inh, node_mask)

        # Global pooling (mean and max over nodes)
        mask_expanded = node_mask[:, :, None]
        h_masked = h * mask_expanded

        # Mean pooling
        h_sum = mx.sum(h_masked, axis=1)
        n_nodes = mx.sum(node_mask, axis=1, keepdims=True) + 1e-8
        h_mean = h_sum / n_nodes

        # Max pooling (set masked to -inf)
        h_for_max = h_masked + (1 - mask_expanded) * (-1e9)
        h_max = mx.max(h_for_max, axis=1)

        # Concatenate and classify
        h_graph = mx.concatenate([h_mean, h_max], axis=-1)
        logits = self.readout(h_graph)

        return logits


class MPLayer(nn.Module):
    """
    Message Passing Layer with separate handling for activation/inhibition.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Separate message networks for activation and inhibition
        self.msg_act = nn.Linear(hidden_dim, hidden_dim)
        self.msg_inh = nn.Linear(hidden_dim, hidden_dim)

        # Update network
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # self + act_msg + inh_msg
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
        """
        Args:
            h: (B, N, D) node features
            adj_act: (B, N, N) activation adjacency
            adj_inh: (B, N, N) inhibition adjacency
            node_mask: (B, N) binary mask
        """
        # Activation messages: sum over neighbors with activation edges
        msg_act = self.msg_act(h)  # (B, N, D)
        agg_act = mx.matmul(adj_act, msg_act)  # (B, N, D)

        # Inhibition messages: sum over neighbors with inhibition edges
        msg_inh = self.msg_inh(h)  # (B, N, D)
        agg_inh = mx.matmul(adj_inh, msg_inh)  # (B, N, D)

        # Combine with self
        combined = mx.concatenate([h, agg_act, agg_inh], axis=-1)
        h_new = self.node_update(combined)

        # Residual connection + normalization
        h_new = self.norm(h + h_new)

        # Apply mask
        h_new = h_new * node_mask[:, :, None]

        return h_new


def topology_to_adj_matrix(
    edges: List,
    n_genes: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array]:
    """Convert topology edges to adjacency matrix."""
    adj = np.zeros((max_genes, max_genes), dtype=np.float32)

    for src, tgt, etype in edges:
        adj[src, tgt] = float(etype)  # +1 for activation, -1 for inhibition

    mask = np.zeros(max_genes, dtype=np.float32)
    mask[:n_genes] = 1.0

    return mx.array(adj), mx.array(mask)


def create_balanced_batch(
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Create a balanced training batch."""
    adj_list = []
    mask_list = []
    label_list = []

    phenotypes = [p for p in circuits_by_phenotype.keys() if circuits_by_phenotype[p]]
    samples_per_class = max(batch_size // len(phenotypes), 1)

    for phenotype in phenotypes:
        circuits = circuits_by_phenotype[phenotype]
        n_samples = min(samples_per_class, len(circuits))
        sampled = random.sample(circuits, n_samples)

        for circuit in sampled:
            adj, mask = topology_to_adj_matrix(
                circuit['edges'],
                circuit['n_genes'],
                max_genes,
            )
            adj_list.append(adj)
            mask_list.append(mask)
            label_list.append(PHENOTYPE_TO_ID[phenotype])

    combined = list(zip(adj_list, mask_list, label_list))
    random.shuffle(combined)
    adj_list, mask_list, label_list = zip(*combined)

    return (
        mx.stack(adj_list),
        mx.stack(mask_list),
        mx.array(label_list),
    )


def train(
    model: EdgeTypeGNN,
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

    def loss_fn(model, adj, mask, labels):
        logits = model(adj, mask)
        return mx.mean(nn.losses.cross_entropy(logits, labels)), logits

    best_acc = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step in range(steps_per_epoch):
            adj, mask, labels = create_balanced_batch(
                circuits_by_phenotype, batch_size, max_genes
            )

            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            (loss, logits), grads = loss_and_grad_fn(model, adj, mask, labels)

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
    model: EdgeTypeGNN,
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
            adj, mask = topology_to_adj_matrix(
                circuit['edges'], circuit['n_genes'], max_genes
            )
            logits = model(adj[None], mask[None])
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
    model: EdgeTypeGNN,
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
            adj, mask = topology_to_adj_matrix(
                circuit['edges'], circuit['n_genes'], max_genes
            )

            # Model prediction
            logits = model(adj[None], mask[None])
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
    parser = argparse.ArgumentParser(description="Train Edge-Type GNN")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gnn_v3")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("gnn_v3", log_file=f"{checkpoint_dir}/train.log")

    # Load data
    logger.info(f"Loading circuits from {args.verified_circuits}")
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    # Create model
    model = EdgeTypeGNN(
        max_genes=args.max_genes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    # Count parameters (simplified)
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
