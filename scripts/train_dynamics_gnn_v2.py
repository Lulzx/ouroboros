#!/usr/bin/env python3
"""
Training script v2 for Dynamics-Informed Graph Neural Network.

KEY CHANGES from v1:
1. Focus on CLASSIFICATION FIRST - no auxiliary losses initially
2. BALANCED sampling across phenotypes (undersampling majority class)
3. Warmup learning rate schedule
4. Much simpler training loop
5. Add auxiliary losses only AFTER classification converges

The insight: auxiliary losses were overwhelming the main classification signal.
We need the model to first learn to classify correctly, THEN fine-tune with
dynamics-aware losses.
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from src.model.graph_dynamics import (
    GraphDynamicsArgs,
    DynamicsInformedGNN,
)
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


def load_verified_circuits(path: str) -> Dict:
    """Load verified circuit database."""
    with open(path) as f:
        return json.load(f)


def topology_to_adj_matrix(
    edges: List,
    n_genes: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array]:
    """Convert topology edges to adjacency matrix."""
    adj = np.zeros((max_genes, max_genes), dtype=np.float32)

    for src, tgt, etype in edges:
        if etype == 1:  # activation
            adj[src, tgt] = 1.0
        elif etype == -1:  # inhibition
            adj[src, tgt] = -1.0

    mask = np.zeros(max_genes, dtype=np.float32)
    mask[:n_genes] = 1.0

    return mx.array(adj), mx.array(mask)


def create_balanced_batch(
    circuits_by_phenotype: Dict[str, List],
    batch_size: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Create a BALANCED training batch.

    Key insight: Sample equal numbers from each phenotype to prevent
    majority class (oscillator) from dominating.
    """
    adj_list = []
    mask_list = []
    label_list = []

    # Get available phenotypes
    phenotypes = [p for p in circuits_by_phenotype.keys() if circuits_by_phenotype[p]]
    samples_per_class = max(batch_size // len(phenotypes), 1)

    for phenotype in phenotypes:
        circuits = circuits_by_phenotype[phenotype]

        # Sample from this class
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

    # Shuffle
    combined = list(zip(adj_list, mask_list, label_list))
    random.shuffle(combined)
    adj_list, mask_list, label_list = zip(*combined)

    return (
        mx.stack(adj_list),
        mx.stack(mask_list),
        mx.array(label_list),
    )


def compute_class_weights(circuits_by_phenotype: Dict[str, List]) -> mx.array:
    """Compute inverse frequency class weights."""
    counts = []
    for i in range(len(PHENOTYPE_TO_ID)):
        phenotype = ID_TO_PHENOTYPE[i]
        counts.append(len(circuits_by_phenotype.get(phenotype, [])) + 1)  # +1 to avoid div by zero

    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]

    # Normalize
    max_w = max(weights)
    weights = [w / max_w for w in weights]

    return mx.array(weights)


def train_classification_phase(
    model: DynamicsInformedGNN,
    optimizer: optim.Optimizer,
    circuits_by_phenotype: Dict[str, List],
    epochs: int,
    batch_size: int,
    max_genes: int,
    class_weights: mx.array,
    logger,
) -> float:
    """
    Phase 1: Pure classification training.

    No auxiliary losses - just learn to classify correctly.
    """
    logger.info("\n" + "="*60)
    logger.info("Phase 1: Pure Classification")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
    logger.info("="*60)

    total_circuits = sum(len(v) for v in circuits_by_phenotype.values())
    steps_per_epoch = max(total_circuits // batch_size, 10)

    best_acc = 0.0
    patience = 0
    max_patience = 15

    def loss_fn(model, adj_matrix, node_mask, phenotype_ids):
        """Pure classification loss with class weighting."""
        outputs = model(adj_matrix, node_mask)

        # Weighted cross-entropy
        logits = outputs['phenotype_logits']
        ce = nn.losses.cross_entropy(logits, phenotype_ids, reduction='none')

        # Apply class weights
        sample_weights = class_weights[phenotype_ids]
        weighted_ce = ce * sample_weights

        return mx.mean(weighted_ce), outputs

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step in range(steps_per_epoch):
            adj_matrix, node_mask, phenotype_ids = create_balanced_batch(
                circuits_by_phenotype, batch_size, max_genes
            )

            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            (loss, outputs), grads = loss_and_grad_fn(
                model, adj_matrix, node_mask, phenotype_ids
            )

            # Gradient clipping
            grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)

            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)
            predictions = mx.argmax(outputs['phenotype_logits'], axis=-1)
            correct = mx.sum(predictions == phenotype_ids)
            epoch_correct += int(correct)
            epoch_total += phenotype_ids.shape[0]

        avg_loss = epoch_loss / steps_per_epoch
        accuracy = epoch_correct / max(epoch_total, 1)

        if accuracy > best_acc:
            best_acc = accuracy
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                f"acc={accuracy:.1%}, best={best_acc:.1%}"
            )

        # Early stopping
        if patience >= max_patience and best_acc > 0.6:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    return best_acc


def train_with_dynamics(
    model: DynamicsInformedGNN,
    optimizer: optim.Optimizer,
    circuits_by_phenotype: Dict[str, List],
    epochs: int,
    batch_size: int,
    max_genes: int,
    class_weights: mx.array,
    logger,
) -> float:
    """
    Phase 2: Add dynamics-aware losses.

    Now that classification works, add auxiliary losses to improve.
    """
    logger.info("\n" + "="*60)
    logger.info("Phase 2: Dynamics-Aware Fine-tuning")
    logger.info(f"Epochs: {epochs}")
    logger.info("="*60)

    total_circuits = sum(len(v) for v in circuits_by_phenotype.values())
    steps_per_epoch = max(total_circuits // batch_size, 10)

    best_acc = 0.0

    def loss_fn(model, adj_matrix, node_mask, phenotype_ids):
        """Classification + dynamics consistency."""
        outputs = model(adj_matrix, node_mask, return_features=True)

        # Classification loss (weighted)
        logits = outputs['phenotype_logits']
        ce = nn.losses.cross_entropy(logits, phenotype_ids, reduction='none')
        sample_weights = class_weights[phenotype_ids]
        ce_loss = mx.mean(ce * sample_weights)

        # Light dynamics consistency (very small weight)
        dynamics_loss = compute_light_dynamics_loss(
            outputs['attractor_features'],
            outputs['motif_features'],
            phenotype_ids,
        )

        # Contrastive loss (small weight)
        contrastive_loss = compute_contrastive_loss(
            outputs['contrastive_features'],
            phenotype_ids,
        )

        total = ce_loss + 0.1 * dynamics_loss + 0.1 * contrastive_loss

        return total, outputs

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step in range(steps_per_epoch):
            adj_matrix, node_mask, phenotype_ids = create_balanced_batch(
                circuits_by_phenotype, batch_size, max_genes
            )

            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            (loss, outputs), grads = loss_and_grad_fn(
                model, adj_matrix, node_mask, phenotype_ids
            )

            grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_loss += float(loss)
            predictions = mx.argmax(outputs['phenotype_logits'], axis=-1)
            correct = mx.sum(predictions == phenotype_ids)
            epoch_correct += int(correct)
            epoch_total += phenotype_ids.shape[0]

        avg_loss = epoch_loss / steps_per_epoch
        accuracy = epoch_correct / max(epoch_total, 1)

        if accuracy > best_acc:
            best_acc = accuracy

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                f"acc={accuracy:.1%}, best={best_acc:.1%}"
            )

    return best_acc


def compute_light_dynamics_loss(
    attractor_features: Dict[str, mx.array],
    motif_features: Dict[str, mx.array],
    phenotype_ids: mx.array,
) -> mx.array:
    """Light dynamics consistency - phenotype-specific priors."""
    B = phenotype_ids.shape[0]

    fixed_point = attractor_features['fixed_point']
    oscillation = attractor_features['oscillation']
    mutual_inhib = motif_features.get('mutual_inhibition', mx.zeros((B,)))

    # Soft priors (not hard constraints)
    osc_mask = (phenotype_ids == 0).astype(mx.float32)
    toggle_mask = (phenotype_ids == 1).astype(mx.float32)
    stable_mask = (phenotype_ids == 5).astype(mx.float32)

    loss = (
        osc_mask * mx.maximum(0.0, 0.5 - oscillation) +
        toggle_mask * mx.maximum(0.0, 0.5 - mutual_inhib) +
        stable_mask * mx.maximum(0.0, 0.5 - fixed_point)
    )

    return mx.mean(loss)


def compute_contrastive_loss(
    features: mx.array,
    labels: mx.array,
    temperature: float = 0.1,
) -> mx.array:
    """Supervised contrastive loss."""
    B = features.shape[0]
    if B < 2:
        return mx.array(0.0)

    # Normalize features
    features = features / (mx.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)

    sim = mx.matmul(features, features.T) / temperature

    labels_expand = labels[:, None]
    positive_mask = (labels_expand == labels[None, :]).astype(mx.float32)

    eye_mask = mx.eye(B)
    positive_mask = positive_mask * (1 - eye_mask)

    exp_sim = mx.exp(sim - mx.max(sim, axis=1, keepdims=True)) * (1 - eye_mask)
    log_prob = sim - mx.log(mx.sum(exp_sim, axis=1, keepdims=True) + 1e-8)

    num_positives = mx.sum(positive_mask, axis=1) + 1e-8
    loss = -mx.sum(positive_mask * log_prob, axis=1) / num_positives

    return mx.mean(loss)


def evaluate_model(
    model: DynamicsInformedGNN,
    circuits_by_phenotype: Dict[str, List],
    max_genes: int = 10,
    samples_per_phenotype: int = 200,
    logger=None,
) -> Dict[str, float]:
    """Evaluate model accuracy per phenotype."""
    results = {}
    all_correct = 0
    all_total = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))
        correct = 0

        for circuit in sampled:
            adj, mask = topology_to_adj_matrix(
                circuit['edges'],
                circuit['n_genes'],
                max_genes,
            )

            outputs = model(adj[None], mask[None])
            pred_id = int(mx.argmax(outputs['phenotype_logits'][0]))
            pred_phenotype = ID_TO_PHENOTYPE[pred_id]

            if pred_phenotype == phenotype:
                correct += 1

        accuracy = correct / len(sampled)
        results[phenotype] = accuracy
        all_correct += correct
        all_total += len(sampled)

        if logger:
            logger.info(f"  {phenotype}: {accuracy:.1%} ({correct}/{len(sampled)})")

    results['overall'] = all_correct / max(all_total, 1)
    if logger:
        logger.info(f"  Overall: {results['overall']:.1%}")

    return results


def oracle_evaluate(
    model: DynamicsInformedGNN,
    circuits_by_phenotype: Dict[str, List],
    classifier: BehaviorClassifier,
    max_genes: int = 10,
    samples_per_phenotype: int = 50,
    logger=None,
) -> Dict[str, float]:
    """Evaluate using actual Boolean network simulation."""
    model_matches_oracle = 0
    total = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))

        for circuit in sampled:
            adj, mask = topology_to_adj_matrix(
                circuit['edges'],
                circuit['n_genes'],
                max_genes,
            )

            # Model prediction
            outputs = model(adj[None], mask[None])
            pred_id = int(mx.argmax(outputs['phenotype_logits'][0]))
            model_pred = ID_TO_PHENOTYPE[pred_id]

            # Oracle prediction
            gene_names = [f"gene_{i}" for i in range(circuit['n_genes'])]
            interactions = []
            for src, tgt, etype in circuit['edges']:
                interactions.append({
                    'source': gene_names[src],
                    'target': gene_names[tgt],
                    'type': 'activates' if etype == 1 else 'inhibits',
                })

            circuit_dict = {'interactions': interactions}
            network = BooleanNetwork.from_circuit(circuit_dict)
            oracle_pred, _ = classifier.classify(network)

            if model_pred == oracle_pred:
                model_matches_oracle += 1
            total += 1

    agreement = model_matches_oracle / max(total, 1)
    if logger:
        logger.info(f"Model-Oracle Agreement: {agreement:.1%} ({model_matches_oracle}/{total})")

    return {'model_oracle_agreement': agreement}


def main():
    parser = argparse.ArgumentParser(description="Train Dynamics-Informed GNN v2")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dynamics_gnn_v2")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-mp-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--phase1-epochs", type=int, default=100)
    parser.add_argument("--phase2-epochs", type=int, default=50)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("dynamics_gnn_v2", log_file=f"{checkpoint_dir}/train.log")

    # Load data
    logger.info(f"Loading verified circuits from {args.verified_circuits}")
    verified_db = load_verified_circuits(args.verified_circuits)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})
    total = sum(len(v) for v in circuits_by_phenotype.values())
    logger.info(f"Loaded {total} verified circuits across {len(circuits_by_phenotype)} phenotypes")

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    # Compute class weights
    class_weights = compute_class_weights(circuits_by_phenotype)
    logger.info(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # Create model
    model_args = GraphDynamicsArgs(
        max_genes=args.max_genes,
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        num_sim_steps=20,
        use_spectral=True,
    )
    model = DynamicsInformedGNN(model_args)

    logger.info(f"Model parameters: {model.num_parameters:,}")

    # Phase 1: Pure classification with warmup
    lr_schedule = optim.join_schedules(
        [optim.linear_schedule(1e-5, args.learning_rate, 10),  # Warmup
         optim.cosine_decay(args.learning_rate, args.phase1_epochs - 10)],
        [10]
    )
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)

    phase1_acc = train_classification_phase(
        model, optimizer, circuits_by_phenotype,
        epochs=args.phase1_epochs,
        batch_size=args.batch_size,
        max_genes=args.max_genes,
        class_weights=class_weights,
        logger=logger,
    )

    # Phase 2: Dynamics-aware fine-tuning (lower LR)
    optimizer2 = optim.AdamW(learning_rate=args.learning_rate * 0.1, weight_decay=0.01)

    phase2_acc = train_with_dynamics(
        model, optimizer2, circuits_by_phenotype,
        epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        max_genes=args.max_genes,
        class_weights=class_weights,
        logger=logger,
    )

    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("Final Evaluation")
    logger.info("="*60)

    results = evaluate_model(
        model, circuits_by_phenotype,
        max_genes=args.max_genes, logger=logger
    )

    # Oracle evaluation
    logger.info("\n" + "="*60)
    logger.info("Oracle Validation")
    logger.info("="*60)

    classifier = BehaviorClassifier(
        num_initial_conditions=20,
        max_steps=200,
        rule="constitutive",
    )
    oracle_results = oracle_evaluate(
        model, circuits_by_phenotype, classifier,
        max_genes=args.max_genes, logger=logger
    )

    # Save model
    def flatten_params(params, prefix=""):
        flat = {}
        if isinstance(params, dict):
            for k, v in params.items():
                new_key = f"{prefix}.{k}" if prefix else k
                flat.update(flatten_params(v, new_key))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                new_key = f"{prefix}.{i}" if prefix else str(i)
                flat.update(flatten_params(v, new_key))
        else:
            flat[prefix] = params
        return flat

    flat_weights = flatten_params(model.parameters())
    save_path = checkpoint_dir / "best.safetensors"
    mx.save_safetensors(str(save_path), flat_weights)
    logger.info(f"Model saved to {save_path}")

    # Save results
    all_results = {
        'classification': results,
        'oracle': oracle_results,
        'phase1_best_acc': phase1_acc,
        'phase2_best_acc': phase2_acc,
    }
    results_path = checkpoint_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Phase 1 best accuracy: {phase1_acc:.1%}")
    logger.info(f"Phase 2 best accuracy: {phase2_acc:.1%}")
    logger.info(f"Final overall accuracy: {results['overall']:.1%}")
    logger.info(f"Model-Oracle agreement: {oracle_results['model_oracle_agreement']:.1%}")


if __name__ == "__main__":
    main()
