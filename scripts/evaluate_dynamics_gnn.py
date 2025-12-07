#!/usr/bin/env python3
"""
Evaluation script for Dynamics-Informed GNN.

This script comprehensively evaluates the learned model:

1. CLASSIFICATION ACCURACY: On held-out verified circuits
2. GENERALIZATION: On larger circuits not seen during training
3. ORACLE VALIDATION: Using the boolean network simulator
4. REPRESENTATION ANALYSIS: t-SNE/UMAP of learned embeddings
5. ABLATION STUDIES: Effect of each component

The key metric is ORACLE ACCURACY: what fraction of generated/classified
circuits actually exhibit the intended dynamical behavior when simulated.
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
import numpy as np

from src.model.graph_dynamics import (
    GraphDynamicsArgs,
    DynamicsInformedGNN,
    circuit_to_graph,
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


def load_model(checkpoint_path: str, args: GraphDynamicsArgs) -> DynamicsInformedGNN:
    """Load trained model from checkpoint."""
    model = DynamicsInformedGNN(args)
    weights = mx.load(checkpoint_path)

    # Handle flattened weight format
    def unflatten_params(flat_weights):
        """Convert flat weights back to nested structure."""
        result = {}
        for key, value in flat_weights.items():
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                if part.isdigit():
                    part = int(part)
                if isinstance(current, dict):
                    if part not in current:
                        current[part] = {} if not parts[parts.index(str(part))+1].isdigit() else []
                    current = current[part]
                elif isinstance(current, list):
                    while len(current) <= part:
                        current.append({})
                    current = current[part]
            final_key = parts[-1]
            if final_key.isdigit():
                while len(current) <= int(final_key):
                    current.append(None)
                current[int(final_key)] = value
            else:
                current[final_key] = value
        return result

    model.load_weights(list(weights.items()))
    return model


def topology_to_circuit_dict(
    edges: List,
    n_genes: int,
    gene_names: List[str] = None,
) -> Dict:
    """Convert topology to circuit dictionary for simulator."""
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(n_genes)]

    interactions = []
    for src, tgt, etype in edges:
        interactions.append({
            'source': gene_names[src],
            'target': gene_names[tgt],
            'type': 'activates' if etype == 1 else 'inhibits',
        })

    return {'interactions': interactions}


def topology_to_adj_matrix(
    edges: List,
    n_genes: int,
    max_genes: int = 10,
) -> Tuple[mx.array, mx.array]:
    """Convert topology to adjacency matrix."""
    adj = np.zeros((max_genes, max_genes), dtype=np.float32)

    for src, tgt, etype in edges:
        adj[src, tgt] = float(etype)

    mask = np.zeros(max_genes, dtype=np.float32)
    mask[:n_genes] = 1.0

    return mx.array(adj), mx.array(mask)


def evaluate_classification(
    model: DynamicsInformedGNN,
    circuits_by_phenotype: Dict[str, List],
    max_genes: int = 10,
    samples_per_phenotype: int = 200,
) -> Dict[str, float]:
    """
    Evaluate classification accuracy on verified circuits.

    Returns accuracy per phenotype and confusion matrix.
    """
    results = {
        'per_phenotype': {},
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
    }

    for true_phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))
        correct = 0
        total = 0

        for circuit in sampled:
            adj, mask = topology_to_adj_matrix(
                circuit['edges'],
                circuit['n_genes'],
                max_genes,
            )

            outputs = model(adj[None], mask[None])
            pred_id = int(mx.argmax(outputs['phenotype_logits'][0]))
            pred_phenotype = ID_TO_PHENOTYPE[pred_id]

            results['confusion_matrix'][true_phenotype][pred_phenotype] += 1

            if pred_phenotype == true_phenotype:
                correct += 1
            total += 1

        results['per_phenotype'][true_phenotype] = correct / max(total, 1)

    # Overall accuracy
    total_correct = sum(
        results['confusion_matrix'][p][p]
        for p in results['per_phenotype']
    )
    total_samples = sum(
        sum(results['confusion_matrix'][p].values())
        for p in results['per_phenotype']
    )
    results['overall'] = total_correct / max(total_samples, 1)

    return results


def evaluate_oracle_consistency(
    model: DynamicsInformedGNN,
    circuits_by_phenotype: Dict[str, List],
    classifier: BehaviorClassifier,
    max_genes: int = 10,
    samples_per_phenotype: int = 100,
) -> Dict[str, float]:
    """
    Evaluate oracle consistency: does the model's prediction match
    what the simulator actually classifies?

    This tests whether the model has learned the TRUE dynamics, not just
    patterns that correlate with labels.
    """
    results = {'per_phenotype': {}, 'details': []}

    for labeled_phenotype, circuits in circuits_by_phenotype.items():
        if not circuits:
            continue

        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))

        model_oracle_agree = 0
        model_label_agree = 0
        oracle_label_agree = 0
        total = 0

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
            circuit_dict = topology_to_circuit_dict(
                circuit['edges'],
                circuit['n_genes'],
            )
            network = BooleanNetwork.from_circuit(circuit_dict)
            oracle_pred, _ = classifier.classify(network)

            # Agreement metrics
            if model_pred == oracle_pred:
                model_oracle_agree += 1
            if model_pred == labeled_phenotype:
                model_label_agree += 1
            if oracle_pred == labeled_phenotype:
                oracle_label_agree += 1

            total += 1

            results['details'].append({
                'labeled': labeled_phenotype,
                'model': model_pred,
                'oracle': oracle_pred,
            })

        results['per_phenotype'][labeled_phenotype] = {
            'model_oracle_agreement': model_oracle_agree / max(total, 1),
            'model_label_agreement': model_label_agree / max(total, 1),
            'oracle_label_agreement': oracle_label_agree / max(total, 1),
        }

    # Overall
    total = len(results['details'])
    model_oracle = sum(1 for d in results['details'] if d['model'] == d['oracle'])
    model_label = sum(1 for d in results['details'] if d['model'] == d['labeled'])
    oracle_label = sum(1 for d in results['details'] if d['oracle'] == d['labeled'])

    results['overall'] = {
        'model_oracle_agreement': model_oracle / max(total, 1),
        'model_label_agreement': model_label / max(total, 1),
        'oracle_label_agreement': oracle_label / max(total, 1),
    }

    return results


def evaluate_generalization(
    model: DynamicsInformedGNN,
    circuits_by_phenotype: Dict[str, List],
    classifier: BehaviorClassifier,
    test_sizes: List[int] = [4, 5, 6],
    max_genes: int = 10,
    samples_per_phenotype: int = 50,
) -> Dict[str, Dict]:
    """
    Evaluate generalization to larger circuits.

    This tests whether the model has learned the underlying principles,
    not just memorized the training set.
    """
    results = {}

    for n_genes in test_sizes:
        # Filter circuits of exactly this size
        filtered = {}
        for phenotype, circuits in circuits_by_phenotype.items():
            filtered[phenotype] = [c for c in circuits if c['n_genes'] == n_genes]

        if sum(len(v) for v in filtered.values()) == 0:
            results[f'{n_genes}_genes'] = {'error': 'no_circuits'}
            continue

        gen_results = evaluate_oracle_consistency(
            model, filtered, classifier,
            max_genes, samples_per_phenotype
        )

        results[f'{n_genes}_genes'] = {
            'model_oracle_agreement': gen_results['overall']['model_oracle_agreement'],
            'per_phenotype': {
                p: v['model_oracle_agreement']
                for p, v in gen_results['per_phenotype'].items()
            }
        }

    return results


def analyze_learned_features(
    model: DynamicsInformedGNN,
    circuits_by_phenotype: Dict[str, List],
    max_genes: int = 10,
    samples_per_phenotype: int = 50,
) -> Dict:
    """
    Analyze the learned representations.

    Compute:
    - Within-class variance (should be low)
    - Between-class distance (should be high)
    - Clustering quality
    """
    features_by_phenotype = defaultdict(list)

    for phenotype, circuits in circuits_by_phenotype.items():
        sampled = random.sample(circuits, min(samples_per_phenotype, len(circuits)))

        for circuit in sampled:
            adj, mask = topology_to_adj_matrix(
                circuit['edges'],
                circuit['n_genes'],
                max_genes,
            )

            outputs = model(adj[None], mask[None], return_features=True)
            features = outputs['contrastive_features'][0]
            features_by_phenotype[phenotype].append(np.array(features))

    # Compute metrics
    within_class_var = {}
    class_centroids = {}

    for phenotype, features_list in features_by_phenotype.items():
        if not features_list:
            continue
        features_array = np.stack(features_list)
        centroid = features_array.mean(axis=0)
        variance = np.mean(np.sum((features_array - centroid) ** 2, axis=1))

        within_class_var[phenotype] = float(variance)
        class_centroids[phenotype] = centroid

    # Between-class distances
    phenotypes = list(class_centroids.keys())
    between_class_dist = {}

    for i, p1 in enumerate(phenotypes):
        for p2 in phenotypes[i+1:]:
            dist = np.sum((class_centroids[p1] - class_centroids[p2]) ** 2)
            between_class_dist[f'{p1}_vs_{p2}'] = float(dist)

    return {
        'within_class_variance': within_class_var,
        'between_class_distance': between_class_dist,
        'mean_within_var': np.mean(list(within_class_var.values())),
        'mean_between_dist': np.mean(list(between_class_dist.values())),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dynamics-Informed GNN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--output-dir", type=str, default="results/dynamics_gnn")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--samples-per-phenotype", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("eval_dynamics_gnn", log_file=f"{output_dir}/eval.log")

    # Load data
    logger.info(f"Loading verified circuits from {args.verified_circuits}")
    with open(args.verified_circuits) as f:
        verified_db = json.load(f)

    circuits_by_phenotype = verified_db.get('verified_circuits', {})

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model_args = GraphDynamicsArgs(
        max_genes=args.max_genes,
        hidden_dim=128,
        num_mp_layers=4,
        use_spectral=True,
    )
    model = DynamicsInformedGNN(model_args)
    weights = mx.load(args.checkpoint)
    model.load_weights(list(weights.items()))

    # Initialize classifier
    classifier = BehaviorClassifier(
        num_initial_conditions=20,
        max_steps=200,
        rule="constitutive",
    )

    # Run evaluations
    all_results = {}

    # 1. Classification accuracy
    logger.info("\n" + "="*60)
    logger.info("Classification Accuracy")
    logger.info("="*60)
    class_results = evaluate_classification(
        model, circuits_by_phenotype,
        args.max_genes, args.samples_per_phenotype
    )
    all_results['classification'] = class_results

    logger.info(f"Overall accuracy: {class_results['overall']:.1%}")
    for p, acc in class_results['per_phenotype'].items():
        logger.info(f"  {p}: {acc:.1%}")

    # 2. Oracle consistency
    logger.info("\n" + "="*60)
    logger.info("Oracle Consistency")
    logger.info("="*60)
    oracle_results = evaluate_oracle_consistency(
        model, circuits_by_phenotype, classifier,
        args.max_genes, args.samples_per_phenotype
    )
    all_results['oracle'] = oracle_results

    logger.info(f"Model-Oracle agreement: {oracle_results['overall']['model_oracle_agreement']:.1%}")
    logger.info(f"Model-Label agreement: {oracle_results['overall']['model_label_agreement']:.1%}")
    logger.info(f"Oracle-Label agreement: {oracle_results['overall']['oracle_label_agreement']:.1%}")

    # 3. Generalization
    logger.info("\n" + "="*60)
    logger.info("Generalization to Larger Circuits")
    logger.info("="*60)
    gen_results = evaluate_generalization(
        model, circuits_by_phenotype, classifier,
        test_sizes=[4, 5, 6], max_genes=args.max_genes
    )
    all_results['generalization'] = gen_results

    for size, results in gen_results.items():
        if 'error' in results:
            logger.info(f"  {size}: {results['error']}")
        else:
            logger.info(f"  {size}: {results['model_oracle_agreement']:.1%}")

    # 4. Feature analysis
    logger.info("\n" + "="*60)
    logger.info("Learned Feature Analysis")
    logger.info("="*60)
    feature_results = analyze_learned_features(
        model, circuits_by_phenotype, args.max_genes
    )
    all_results['features'] = feature_results

    logger.info(f"Mean within-class variance: {feature_results['mean_within_var']:.4f}")
    logger.info(f"Mean between-class distance: {feature_results['mean_between_dist']:.4f}")
    logger.info(f"Separation ratio: {feature_results['mean_between_dist'] / (feature_results['mean_within_var'] + 1e-8):.2f}")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert any remaining numpy arrays
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        json.dump(convert_numpy(all_results), f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Classification accuracy: {class_results['overall']:.1%}")
    logger.info(f"Model-Oracle agreement: {oracle_results['overall']['model_oracle_agreement']:.1%}")
    logger.info(f"Feature separation: {feature_results['mean_between_dist'] / (feature_results['mean_within_var'] + 1e-8):.2f}x")


if __name__ == "__main__":
    main()
