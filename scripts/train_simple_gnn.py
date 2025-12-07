#!/usr/bin/env python3
"""
Minimal GNN training script to debug learning issues.

This is a drastically simplified version to verify the model CAN learn.
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


class SimpleGNN(nn.Module):
    """Minimal GNN for debugging - just MLP on flattened adjacency."""

    def __init__(self, max_genes: int = 10, hidden_dim: int = 128, num_classes: int = 6):
        super().__init__()
        self.max_genes = max_genes
        input_dim = max_genes * max_genes  # Flattened adjacency

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def __call__(self, adj_matrix: mx.array, node_mask: mx.array) -> mx.array:
        B = adj_matrix.shape[0]
        flat = adj_matrix.reshape(B, -1)
        return self.net(flat)


def topology_to_adj_matrix(
    edges: List,
    n_genes: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array]:
    """Convert topology edges to adjacency matrix."""
    adj = np.zeros((max_genes, max_genes), dtype=np.float32)

    for src, tgt, etype in edges:
        if etype == 1:
            adj[src, tgt] = 1.0
        elif etype == -1:
            adj[src, tgt] = -1.0

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


def main():
    parser = argparse.ArgumentParser(description="Train Simple GNN")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    logger = setup_logger("simple_gnn")

    # Load data
    logger.info(f"Loading circuits from {args.verified_circuits}")
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    # Create model
    model = SimpleGNN(max_genes=args.max_genes)
    optimizer = optim.Adam(learning_rate=args.lr)

    total_circuits = sum(len(v) for v in circuits_by_phenotype.values())
    steps_per_epoch = max(total_circuits // args.batch_size, 10)

    logger.info(f"\nTraining with {steps_per_epoch} steps/epoch")

    def loss_fn(model, adj_matrix, node_mask, labels):
        logits = model(adj_matrix, node_mask)
        return mx.mean(nn.losses.cross_entropy(logits, labels)), logits

    best_acc = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step in range(steps_per_epoch):
            adj, mask, labels = create_balanced_batch(
                circuits_by_phenotype, args.batch_size, args.max_genes
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
            logger.info(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, acc={accuracy:.1%}, best={best_acc:.1%}")

    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("Final Evaluation (per phenotype)")
    logger.info("="*60)

    for phenotype, circuits in circuits_by_phenotype.items():
        correct = 0
        sampled = random.sample(circuits, min(200, len(circuits)))

        for circuit in sampled:
            adj, mask = topology_to_adj_matrix(
                circuit['edges'], circuit['n_genes'], args.max_genes
            )
            logits = model(adj[None], mask[None])
            pred = int(mx.argmax(logits[0]))

            if ID_TO_PHENOTYPE[pred] == phenotype:
                correct += 1

        logger.info(f"  {phenotype}: {correct}/{len(sampled)} ({correct/len(sampled):.1%})")


if __name__ == "__main__":
    main()
