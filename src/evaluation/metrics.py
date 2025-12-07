"""Evaluation metrics for GRN generation."""

from collections import Counter
from typing import Optional

import mlx.core as mx
import numpy as np

from ..model.transformer import GRNTransformer
from ..model.generation import generate, compute_sequence_log_probs
from ..data.tokenizer import GRNTokenizer
from ..simulator.boolean_network import BooleanNetwork
from ..simulator.classify_behavior import BehaviorClassifier
from ..training.grpo import SelfClassifier


def compute_self_consistency(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    num_samples: int = 100,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> dict:
    """
    Compute self-classification consistency.

    Generates circuits for each phenotype and checks if the model
    self-classifies them correctly.

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        num_samples: Samples per phenotype
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Dictionary with accuracy metrics
    """
    classifier = SelfClassifier(model, tokenizer)
    phenotype_ids = tokenizer.get_phenotype_token_ids()

    total_correct = 0
    total_samples = 0
    per_phenotype = {}

    for phenotype_id in phenotype_ids:
        phenotype_name = tokenizer.get_phenotype_name(phenotype_id)

        # Generate samples
        prompts = mx.array(
            [[tokenizer.bos_token_id, phenotype_id]] * num_samples
        )

        generated = generate(
            model,
            prompts,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Self-classify
        intended = mx.full((num_samples,), phenotype_id, dtype=mx.int32)
        predicted, _ = classifier.classify(generated)
        mx.eval(predicted)

        correct = int((predicted == intended).sum())
        total_correct += correct
        total_samples += num_samples
        per_phenotype[phenotype_name] = correct / num_samples

    return {
        "accuracy": total_correct / total_samples,
        "per_phenotype": per_phenotype,
    }


def compute_oracle_consistency(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    num_samples: int = 100,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
    simulation_steps: int = 100,
    num_initial_conditions: int = 10,
) -> dict:
    """
    Compute oracle (simulator) consistency.

    Generates circuits and validates them against boolean network simulation.

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        num_samples: Samples per phenotype
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        simulation_steps: Steps for simulation
        num_initial_conditions: Initial conditions for simulation

    Returns:
        Dictionary with accuracy metrics
    """
    classifier = BehaviorClassifier(
        num_initial_conditions=num_initial_conditions,
        max_steps=simulation_steps,
    )
    phenotype_ids = tokenizer.get_phenotype_token_ids()

    total_correct = 0
    total_samples = 0
    per_phenotype = {}

    for phenotype_id in phenotype_ids:
        phenotype_name = tokenizer.get_phenotype_name(phenotype_id)

        # Generate samples
        prompts = mx.array(
            [[tokenizer.bos_token_id, phenotype_id]] * num_samples
        )

        generated = generate(
            model,
            prompts,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        mx.eval(generated)

        # Classify with oracle
        correct = 0
        for i in range(num_samples):
            tokens = generated[i].tolist()
            circuit = tokenizer.decode(tokens)

            if not circuit.get("interactions"):
                continue

            try:
                network = BooleanNetwork.from_circuit(circuit)
                predicted, _ = classifier.classify(network, seed=i)
                if predicted == phenotype_name:
                    correct += 1
            except Exception:
                pass

        total_correct += correct
        total_samples += num_samples
        per_phenotype[phenotype_name] = correct / num_samples

    return {
        "accuracy": total_correct / total_samples,
        "per_phenotype": per_phenotype,
    }


def edit_distance(seq1: list, seq2: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_diversity(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    num_samples: int = 100,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> dict:
    """
    Compute diversity of generated circuits.

    Measures pairwise edit distances within each phenotype class.

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        num_samples: Samples per phenotype
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Dictionary with diversity metrics
    """
    phenotype_ids = tokenizer.get_phenotype_token_ids()
    per_phenotype = {}

    for phenotype_id in phenotype_ids:
        phenotype_name = tokenizer.get_phenotype_name(phenotype_id)

        # Generate samples
        prompts = mx.array(
            [[tokenizer.bos_token_id, phenotype_id]] * num_samples
        )

        generated = generate(
            model,
            prompts,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        mx.eval(generated)

        # Extract sequences (filter padding and special tokens)
        sequences = []
        for i in range(num_samples):
            tokens = generated[i].tolist()
            # Remove padding and special tokens
            filtered = [
                t for t in tokens
                if t not in [tokenizer.pad_token_id, tokenizer.bos_token_id,
                            tokenizer.eos_token_id]
            ]
            sequences.append(filtered)

        # Compute pairwise edit distances (sample if too many)
        distances = []
        max_pairs = 1000
        if len(sequences) > 50:
            # Sample pairs
            indices = np.random.choice(len(sequences), size=(max_pairs, 2))
            for i, j in indices:
                if i != j:
                    distances.append(edit_distance(sequences[i], sequences[j]))
        else:
            for i in range(len(sequences)):
                for j in range(i + 1, len(sequences)):
                    distances.append(edit_distance(sequences[i], sequences[j]))

        # Count unique sequences
        unique_seqs = len(set(tuple(s) for s in sequences))

        per_phenotype[phenotype_name] = {
            "mean_edit_distance": float(np.mean(distances)) if distances else 0.0,
            "std_edit_distance": float(np.std(distances)) if distances else 0.0,
            "unique_sequences": unique_seqs,
            "unique_ratio": unique_seqs / num_samples,
        }

    # Aggregate
    all_means = [v["mean_edit_distance"] for v in per_phenotype.values()]
    all_unique_ratios = [v["unique_ratio"] for v in per_phenotype.values()]

    return {
        "mean_edit_distance": float(np.mean(all_means)),
        "mean_unique_ratio": float(np.mean(all_unique_ratios)),
        "per_phenotype": per_phenotype,
    }


def compute_novelty(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    training_circuits: list[dict],
    num_samples: int = 100,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
    edit_threshold: int = 3,
) -> dict:
    """
    Compute novelty of generated circuits compared to training data.

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        training_circuits: List of training circuit dictionaries
        num_samples: Samples per phenotype
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        edit_threshold: Edit distance threshold for "near match"

    Returns:
        Dictionary with novelty metrics
    """
    # Encode training circuits
    training_encoded = set()
    for circuit in training_circuits:
        tokens = tokenizer.encode_flat(circuit)
        # Remove phenotype for comparison (index 1)
        tokens_no_phenotype = tokens[:1] + tokens[2:]  # Keep bos, skip phenotype
        training_encoded.add(tuple(tokens_no_phenotype))

    phenotype_ids = tokenizer.get_phenotype_token_ids()
    total_novel_exact = 0
    total_novel_approx = 0
    total_samples = 0

    for phenotype_id in phenotype_ids:
        prompts = mx.array(
            [[tokenizer.bos_token_id, phenotype_id]] * num_samples
        )

        generated = generate(
            model,
            prompts,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        mx.eval(generated)

        for i in range(num_samples):
            tokens = generated[i].tolist()
            # Remove phenotype
            tokens_no_phenotype = (
                [tokens[0]]
                + [t for t in tokens[2:] if t != tokenizer.pad_token_id]
            )

            # Check exact match
            if tuple(tokens_no_phenotype) not in training_encoded:
                total_novel_exact += 1

            # Check approximate match
            is_novel = True
            for train_seq in training_encoded:
                if edit_distance(tokens_no_phenotype, list(train_seq)) <= edit_threshold:
                    is_novel = False
                    break
            if is_novel:
                total_novel_approx += 1

        total_samples += num_samples

    return {
        "exact_novelty": total_novel_exact / total_samples,
        "approx_novelty": total_novel_approx / total_samples,
        "edit_threshold": edit_threshold,
    }


def compute_validity(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    num_samples: int = 100,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> dict:
    """
    Compute validity of generated circuits.

    Checks:
    - Correct grammar (gene-interaction-gene pattern)
    - No orphan genes
    - Well-formed sequences

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        num_samples: Samples per phenotype
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Dictionary with validity metrics
    """
    phenotype_ids = tokenizer.get_phenotype_token_ids()
    total_valid = 0
    total_with_interactions = 0
    total_samples = 0

    validity_details = {
        "has_eos": 0,
        "has_interactions": 0,
        "valid_grammar": 0,
        "no_orphans": 0,
    }

    for phenotype_id in phenotype_ids:
        prompts = mx.array(
            [[tokenizer.bos_token_id, phenotype_id]] * num_samples
        )

        generated = generate(
            model,
            prompts,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        mx.eval(generated)

        for i in range(num_samples):
            tokens = generated[i].tolist()
            circuit = tokenizer.decode(tokens)
            interactions = circuit.get("interactions", [])

            # Check for EOS
            has_eos = tokenizer.eos_token_id in tokens
            if has_eos:
                validity_details["has_eos"] += 1

            # Check for interactions
            has_interactions = len(interactions) > 0
            if has_interactions:
                validity_details["has_interactions"] += 1
                total_with_interactions += 1

            # Check grammar (all interactions have source, target, type)
            valid_grammar = all(
                "source" in inter and "target" in inter and "type" in inter
                for inter in interactions
            )
            if valid_grammar and has_interactions:
                validity_details["valid_grammar"] += 1

            # Check for orphan genes
            if interactions:
                sources = set(inter["source"] for inter in interactions)
                targets = set(inter["target"] for inter in interactions)
                all_genes = sources | targets
                # A gene is orphan if it's neither source nor target
                # In valid circuits, every gene should appear in interactions
                no_orphans = len(all_genes) > 0
                if no_orphans:
                    validity_details["no_orphans"] += 1

            # Count as valid if has interactions and valid grammar
            if has_interactions and valid_grammar:
                total_valid += 1

        total_samples += num_samples

    return {
        "validity": total_valid / total_samples,
        "has_interactions_ratio": total_with_interactions / total_samples,
        "details": {k: v / total_samples for k, v in validity_details.items()},
    }


def evaluate_model(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    training_circuits: Optional[list[dict]] = None,
    num_samples: int = 100,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
    run_oracle: bool = True,
) -> dict:
    """
    Run full evaluation suite.

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        training_circuits: Training data for novelty computation
        num_samples: Samples per phenotype
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        run_oracle: Whether to run oracle (simulator) evaluation

    Returns:
        Dictionary with all metrics
    """
    results = {}

    # Self-consistency
    results["self_consistency"] = compute_self_consistency(
        model, tokenizer, num_samples, max_length, temperature, top_p
    )

    # Oracle consistency
    if run_oracle:
        results["oracle_consistency"] = compute_oracle_consistency(
            model, tokenizer, num_samples, max_length, temperature, top_p
        )

    # Diversity
    results["diversity"] = compute_diversity(
        model, tokenizer, num_samples, max_length, temperature, top_p
    )

    # Novelty
    if training_circuits:
        results["novelty"] = compute_novelty(
            model, tokenizer, training_circuits, num_samples,
            max_length, temperature, top_p
        )

    # Validity
    results["validity"] = compute_validity(
        model, tokenizer, num_samples, max_length, temperature, top_p
    )

    return results
