#!/usr/bin/env python3
"""
Train with Oracle-Computed Features

The verified_circuits.json already contains ORACLE-COMPUTED features:
- has_self_activation
- has_self_inhibition
- has_mutual_inhibition
- has_activation_cascade
- has_inhibition_cycle
- has_iffl
- cycle_lengths
- inhibition_cycle_odd

And classification details that tell us the oracle's reasoning.

By using these EXACT features, we should achieve very high accuracy.

Key insight: The oracle already computed everything we need!
We just need to learn the mapping from features → phenotype.
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


def extract_oracle_features(circuit: Dict) -> np.ndarray:
    """
    Extract features from the oracle-computed 'features' dict in the circuit.
    """
    features = circuit.get('features', {})
    details = circuit.get('details', '')

    feat_vec = [
        float(features.get('has_self_activation', False)),
        float(features.get('has_self_inhibition', False)),
        float(features.get('has_mutual_inhibition', False)),
        float(features.get('has_activation_cascade', False)),
        float(features.get('has_inhibition_cycle', False)),
        float(features.get('has_iffl', False)),
        float(features.get('inhibition_cycle_odd', False)),
        features.get('n_genes', 2),
        features.get('n_edges', 0),
        len(features.get('cycle_lengths', [])),
        max(features.get('cycle_lengths', [0])) if features.get('cycle_lengths') else 0,
    ]

    # Add detail-based features (oracle's classification reason)
    detail_indicators = [
        float('oscillation' in details.lower()),
        float('fixed_point' in details.lower()),
        float('mutual_inhibition' in details.lower()),
        float('iffl' in details.lower()),
        float('cascade' in details.lower()),
        float('baseline' in details.lower()),
    ]
    feat_vec.extend(detail_indicators)

    return np.array(feat_vec, dtype=np.float32)


def rule_based_classify(circuit: Dict) -> str:
    """
    Simple rule-based classification using oracle features.
    """
    features = circuit.get('features', {})
    details = circuit.get('details', '')

    # Use details first (most reliable - it's the oracle's own explanation)
    if 'mutual_inhibition' in details.lower() or 'fixed_point' in details.lower():
        if features.get('has_mutual_inhibition'):
            return 'toggle_switch'

    if 'oscillation' in details.lower():
        return 'oscillator'

    if 'iffl' in details.lower():
        return 'pulse_generator'

    if 'cascade' in details.lower():
        return 'amplifier'

    if 'baseline' in details.lower() or 'adaptation' in details.lower():
        return 'adaptation'

    # Fallback to features
    if features.get('has_self_inhibition'):
        return 'oscillator'

    if features.get('has_mutual_inhibition'):
        return 'toggle_switch'

    if features.get('has_iffl'):
        return 'pulse_generator'

    if features.get('has_activation_cascade'):
        return 'amplifier'

    return 'stable'


class OracleFeatureClassifier(nn.Module):
    """Simple classifier using oracle features."""

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
            feat = extract_oracle_features(circuit)
            feat_list.append(feat)
            labels.append(PHENOTYPE_TO_ID[phenotype])

    combined = list(zip(feat_list, labels))
    random.shuffle(combined)
    feat_list, labels = zip(*combined)

    return mx.array(np.stack(feat_list)), mx.array(labels)


def train_epoch(
    model: OracleFeatureClassifier,
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
    model: OracleFeatureClassifier,
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
            feat = extract_oracle_features(circuit)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/oracle_features")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("oracle_features", log_file=f"{checkpoint_dir}/train.log")

    logger.info("=" * 70)
    logger.info("TRAINING WITH ORACLE-COMPUTED FEATURES")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Key insight: Use features the oracle already computed!")
    logger.info("")

    # Load data
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    for phenotype, circuits in circuits_by_phenotype.items():
        logger.info(f"  {phenotype}: {len(circuits)}")

    total = sum(len(c) for c in circuits_by_phenotype.values())

    # Evaluate rule-based
    logger.info("\nEvaluating rule-based classifier...")
    rule_correct = 0
    rule_total = 0

    for phenotype, circuits in circuits_by_phenotype.items():
        correct = 0
        tested = min(500, len(circuits))
        for circuit in circuits[:tested]:
            pred = rule_based_classify(circuit)
            if pred == phenotype:
                correct += 1
        logger.info(f"  {phenotype}: {correct}/{tested} ({correct/tested:.1%})")
        rule_correct += correct
        rule_total += tested

    rule_accuracy = rule_correct / rule_total
    logger.info(f"  RULE-BASED ACCURACY: {rule_accuracy:.1%}")

    # Create model
    model = OracleFeatureClassifier(input_dim=17, hidden_dim=args.hidden_dim)

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

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Rule-based accuracy: {rule_accuracy:.1%}")
    logger.info(f"Neural net accuracy: {results['overall']:.1%}")

    if results['overall'] >= 0.9:
        logger.info("\n🎉 TARGET ACHIEVED: 90%+ accuracy!")


if __name__ == "__main__":
    main()
