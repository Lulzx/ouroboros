"""Analysis utilities for generated circuits."""

from collections import Counter
from typing import Optional

import mlx.core as mx
import numpy as np

from ..model.transformer import GRNTransformer
from ..model.generation import generate
from ..data.tokenizer import GRNTokenizer
from ..simulator.boolean_network import BooleanNetwork
from ..simulator.classify_behavior import BehaviorClassifier
from ..training.grpo import SelfClassifier


def analyze_generated_circuits(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    num_samples: int = 50,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> dict:
    """
    Analyze properties of generated circuits.

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        num_samples: Samples per phenotype
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Dictionary with analysis results
    """
    phenotype_ids = tokenizer.get_phenotype_token_ids()
    analysis = {}

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

        # Analyze circuits
        num_genes_list = []
        num_interactions_list = []
        interaction_types = Counter()
        gene_frequencies = Counter()
        motif_patterns = Counter()

        for i in range(num_samples):
            tokens = generated[i].tolist()
            circuit = tokenizer.decode(tokens)
            interactions = circuit.get("interactions", [])

            if not interactions:
                continue

            # Count genes
            genes = set()
            for inter in interactions:
                genes.add(inter["source"])
                genes.add(inter["target"])
            num_genes_list.append(len(genes))

            # Count interactions
            num_interactions_list.append(len(interactions))

            # Interaction types
            for inter in interactions:
                interaction_types[inter["type"]] += 1
                gene_frequencies[inter["source"]] += 1
                gene_frequencies[inter["target"]] += 1

            # Detect motifs
            motifs = detect_motifs(circuit)
            for motif in motifs:
                motif_patterns[motif] += 1

        analysis[phenotype_name] = {
            "num_samples": num_samples,
            "samples_with_interactions": len(num_genes_list),
            "avg_num_genes": float(np.mean(num_genes_list)) if num_genes_list else 0,
            "std_num_genes": float(np.std(num_genes_list)) if num_genes_list else 0,
            "avg_num_interactions": (
                float(np.mean(num_interactions_list)) if num_interactions_list else 0
            ),
            "std_num_interactions": (
                float(np.std(num_interactions_list)) if num_interactions_list else 0
            ),
            "interaction_types": dict(interaction_types),
            "top_genes": dict(gene_frequencies.most_common(10)),
            "motif_patterns": dict(motif_patterns.most_common(10)),
        }

    return analysis


def detect_motifs(circuit: dict) -> list[str]:
    """
    Detect common network motifs in a circuit.

    Args:
        circuit: Circuit dictionary

    Returns:
        List of detected motif names
    """
    interactions = circuit.get("interactions", [])
    if not interactions:
        return []

    motifs = []

    # Build adjacency structure
    activates = {}  # source -> list of targets
    inhibits = {}

    for inter in interactions:
        src = inter["source"]
        tgt = inter["target"]
        if inter["type"] == "activates":
            if src not in activates:
                activates[src] = []
            activates[src].append(tgt)
        else:
            if src not in inhibits:
                inhibits[src] = []
            inhibits[src].append(tgt)

    # Detect autoregulation
    for src, targets in activates.items():
        if src in targets:
            motifs.append("positive_autoregulation")
    for src, targets in inhibits.items():
        if src in targets:
            motifs.append("negative_autoregulation")

    # Detect mutual inhibition (toggle switch)
    for src, targets in inhibits.items():
        for tgt in targets:
            if tgt in inhibits and src in inhibits[tgt]:
                if "mutual_inhibition" not in motifs:
                    motifs.append("mutual_inhibition")

    # Detect cascade (A->B->C)
    all_genes = set()
    for inter in interactions:
        all_genes.add(inter["source"])
        all_genes.add(inter["target"])

    for a in all_genes:
        a_targets = activates.get(a, [])
        for b in a_targets:
            b_targets = activates.get(b, [])
            if b_targets:
                if "activation_cascade" not in motifs:
                    motifs.append("activation_cascade")

    # Detect feed-forward loop
    for a in all_genes:
        a_act_targets = activates.get(a, [])
        a_inh_targets = inhibits.get(a, [])
        all_a_targets = set(a_act_targets + a_inh_targets)

        for b in all_a_targets:
            b_act_targets = activates.get(b, [])
            b_inh_targets = inhibits.get(b, [])
            all_b_targets = set(b_act_targets + b_inh_targets)

            # Check if A also regulates any of B's targets
            shared = all_a_targets & all_b_targets
            if shared:
                # Determine FFL type
                for c in shared:
                    a_to_b_act = b in a_act_targets
                    a_to_c_act = c in a_act_targets
                    b_to_c_act = c in b_act_targets

                    if a_to_b_act and a_to_c_act and not b_to_c_act:
                        if "incoherent_ffl" not in motifs:
                            motifs.append("incoherent_ffl")
                    elif a_to_b_act and a_to_c_act and b_to_c_act:
                        if "coherent_ffl" not in motifs:
                            motifs.append("coherent_ffl")

    # Detect negative feedback loop
    for start in all_genes:
        visited = {start}
        current = start
        path_length = 0

        while path_length < 10:
            # Follow activation edges
            next_genes = activates.get(current, [])
            found_next = False

            for ng in next_genes:
                if ng not in visited:
                    visited.add(ng)
                    current = ng
                    path_length += 1
                    found_next = True
                    break

            if not found_next:
                break

            # Check if current inhibits start
            if start in inhibits.get(current, []):
                if "negative_feedback_loop" not in motifs:
                    motifs.append("negative_feedback_loop")
                break

    # Detect repressilator-like (ring of inhibitions)
    ring_size = len(all_genes)
    if ring_size >= 3:
        # Check if we can form a ring of inhibitions
        inhibition_count = sum(len(v) for v in inhibits.values())
        if inhibition_count >= ring_size:
            # Simple check: each gene inhibits exactly one other
            gene_list = list(all_genes)
            is_ring = True
            for g in gene_list:
                inh_targets = inhibits.get(g, [])
                if len(inh_targets) != 1:
                    is_ring = False
                    break
            if is_ring and ring_size >= 3:
                motifs.append("repressilator_like")

    return motifs


def create_confusion_matrix(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    use_oracle: bool = False,
    num_samples: int = 50,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> dict:
    """
    Create confusion matrix for phenotype classification.

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        use_oracle: Use oracle (simulator) instead of self-classification
        num_samples: Samples per phenotype
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        Dictionary with confusion matrix data
    """
    phenotype_ids = tokenizer.get_phenotype_token_ids()
    phenotype_names = [tokenizer.get_phenotype_name(pid) for pid in phenotype_ids]

    if use_oracle:
        classifier = BehaviorClassifier()
    else:
        classifier = SelfClassifier(model, tokenizer)

    # Initialize confusion matrix
    matrix = {true: {pred: 0 for pred in phenotype_names} for true in phenotype_names}

    for phenotype_id in phenotype_ids:
        true_name = tokenizer.get_phenotype_name(phenotype_id)

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

        if use_oracle:
            # Oracle classification
            for i in range(num_samples):
                tokens = generated[i].tolist()
                circuit = tokenizer.decode(tokens)

                if not circuit.get("interactions"):
                    matrix[true_name]["stable"] += 1
                    continue

                try:
                    network = BooleanNetwork.from_circuit(circuit)
                    predicted, _ = classifier.classify(network, seed=i)
                    if predicted in phenotype_names:
                        matrix[true_name][predicted] += 1
                    else:
                        matrix[true_name]["stable"] += 1
                except Exception:
                    matrix[true_name]["stable"] += 1
        else:
            # Self-classification
            intended = mx.full((num_samples,), phenotype_id, dtype=mx.int32)
            predicted_ids, _ = classifier.classify(generated)
            mx.eval(predicted_ids)

            for i in range(num_samples):
                pred_name = tokenizer.get_phenotype_name(int(predicted_ids[i]))
                if pred_name in phenotype_names:
                    matrix[true_name][pred_name] += 1

    # Normalize
    normalized = {}
    for true_name in phenotype_names:
        total = sum(matrix[true_name].values())
        if total > 0:
            normalized[true_name] = {
                pred: count / total
                for pred, count in matrix[true_name].items()
            }
        else:
            normalized[true_name] = matrix[true_name]

    return {
        "raw": matrix,
        "normalized": normalized,
        "phenotypes": phenotype_names,
    }


def sample_circuits(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    phenotype: str,
    num_samples: int = 10,
    max_length: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> list[dict]:
    """
    Sample circuits for a given phenotype.

    Args:
        model: GRNTransformer model
        tokenizer: GRNTokenizer
        phenotype: Phenotype name (e.g., "oscillator")
        num_samples: Number of samples
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        List of circuit dictionaries
    """
    phenotype_token = tokenizer.phenotype_to_token(phenotype)
    phenotype_id = tokenizer.token_to_id.get(phenotype_token)

    if phenotype_id is None:
        raise ValueError(f"Unknown phenotype: {phenotype}")

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

    circuits = []
    for i in range(num_samples):
        tokens = generated[i].tolist()
        circuit = tokenizer.decode(tokens)
        circuit["generated_tokens"] = tokens
        circuits.append(circuit)

    return circuits
