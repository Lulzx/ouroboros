#!/usr/bin/env python3
"""
Train supervised model on balanced verified circuits.

Key insight: The model needs to learn the STRUCTURE of each phenotype,
not just the correlations. By training on verified circuits with:
1. Class balancing (equal samples per phenotype)
2. Heavy augmentation (gene name shuffling)
3. More epochs

We can teach the model the correct topological patterns.
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from src.data.tokenizer import GRNTokenizer
from src.data.dataset import GRNDataset, GRNDataLoader
from src.model.transformer import GRNTransformer, ModelArgs
from src.training.supervised import SupervisedTrainer
from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier
from src.utils.config import load_config
from src.utils.logging import setup_logger


def load_verified_circuits(path: str) -> dict:
    """Load verified circuit database."""
    with open(path) as f:
        return json.load(f)


def create_balanced_dataset(
    verified_db: dict,
    gene_vocab: list,
    samples_per_phenotype: int = 1000,
    seed: int = 42
) -> list:
    """Create balanced training set from verified circuits."""
    random.seed(seed)
    circuits = []

    phenotypes = ["oscillator", "toggle_switch", "adaptation",
                  "pulse_generator", "amplifier", "stable"]

    for phenotype in phenotypes:
        verified = verified_db["verified_circuits"].get(phenotype, [])
        if not verified:
            print(f"Warning: No verified circuits for {phenotype}")
            continue

        # Oversample to get enough examples
        for _ in range(samples_per_phenotype):
            topo = random.choice(verified)
            edges = topo["edges"]
            n_genes = topo["n_genes"]

            # Random gene names
            gene_names = random.sample(gene_vocab, min(n_genes, len(gene_vocab)))

            interactions = []
            for src, tgt, etype in edges:
                if src < len(gene_names) and tgt < len(gene_names):
                    interactions.append({
                        "source": gene_names[src],
                        "target": gene_names[tgt],
                        "type": "activates" if etype == 1 else "inhibits"
                    })

            circuit = {
                "interactions": interactions,
                "phenotype": phenotype,
                "id": f"verified_{phenotype}_{len(circuits)}"
            }
            circuits.append(circuit)

    random.shuffle(circuits)
    return circuits


def evaluate_model(
    model: GRNTransformer,
    tokenizer: GRNTokenizer,
    classifier: BehaviorClassifier,
    num_samples: int = 50,
    temperature: float = 0.8,
) -> dict:
    """Evaluate model with oracle."""
    from src.model.generation import generate

    phenotypes = ["oscillator", "toggle_switch", "adaptation",
                  "pulse_generator", "amplifier", "stable"]

    results = {}
    total_correct = 0
    total_samples = 0

    for phenotype in phenotypes:
        phenotype_token = f"<{phenotype}>"
        phenotype_id = tokenizer.token_to_id.get(phenotype_token, tokenizer.unk_token_id)

        prompts = mx.array([
            [tokenizer.bos_token_id, phenotype_id]
        ] * num_samples)

        generated = generate(
            model,
            prompts,
            max_length=64,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        correct = 0
        for i in range(num_samples):
            token_ids = generated[i].tolist()
            try:
                circuit = tokenizer.decode(token_ids)
                if circuit and circuit.get("interactions"):
                    network = BooleanNetwork.from_circuit(circuit)
                    predicted, _ = classifier.classify(network)
                    if predicted == phenotype:
                        correct += 1
            except:
                pass

        results[phenotype] = correct / num_samples
        total_correct += correct
        total_samples += num_samples

    results["overall"] = total_correct / total_samples
    return results


def main():
    parser = argparse.ArgumentParser(description="Train on balanced verified circuits")
    parser.add_argument("--verified-circuits", type=str, default="data/verified_circuits.json")
    parser.add_argument("--data-dir", type=str, default="data/processed_relabeled")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/balanced_supervised")
    parser.add_argument("--samples-per-phenotype", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=5)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("balanced_supervised", log_file=f"{checkpoint_dir}/train.log")

    # Load verified circuits
    logger.info(f"Loading verified circuits from {args.verified_circuits}")
    verified_db = load_verified_circuits(args.verified_circuits)

    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer = GRNTokenizer.load(data_dir / "tokenizer.json")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")

    # Extract gene vocabulary
    gene_vocab = [t for t in tokenizer.token_to_id.keys()
                  if not t.startswith("<") and t not in ["activates", "inhibits"]]
    logger.info(f"Gene vocabulary: {len(gene_vocab)} genes")

    # Create balanced dataset
    logger.info(f"Creating balanced dataset with {args.samples_per_phenotype} samples per phenotype")
    train_circuits = create_balanced_dataset(
        verified_db, gene_vocab,
        samples_per_phenotype=args.samples_per_phenotype,
        seed=42
    )
    logger.info(f"Total training circuits: {len(train_circuits)}")

    # Count per phenotype
    phenotype_counts = defaultdict(int)
    for c in train_circuits:
        phenotype_counts[c["phenotype"]] += 1
    for p, count in sorted(phenotype_counts.items()):
        logger.info(f"  {p}: {count}")

    # Create validation set
    val_circuits = create_balanced_dataset(
        verified_db, gene_vocab,
        samples_per_phenotype=100,
        seed=1000
    )

    # Create datasets
    train_dataset = GRNDataset(train_circuits, tokenizer, max_length=128, augment=True)
    val_dataset = GRNDataset(val_circuits, tokenizer, max_length=128, augment=False)

    train_loader = GRNDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GRNDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    config = load_config("configs")
    model_args = ModelArgs(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.model.embed_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_dim=config.model.mlp_dim,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
    )

    model = GRNTransformer(model_args)
    logger.info(f"Model parameters: {model.num_parameters:,}")

    # Create oracle
    classifier = BehaviorClassifier(
        num_initial_conditions=20,
        max_steps=200,
        rule="constitutive",
    )

    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        grad_clip=1.0,
        checkpoint_dir=str(checkpoint_dir),
        pad_token_id=tokenizer.pad_token_id,
    )

    # Training loop with oracle evaluation
    best_oracle_acc = 0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        val_loss = trainer.train(epochs=1, log_interval=50)
        logger.info(f"Validation loss: {val_loss:.4f}")

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            logger.info("Evaluating with oracle...")
            results = evaluate_model(model, tokenizer, classifier, num_samples=30)

            logger.info(f"Oracle accuracy: {results['overall']:.1%}")
            for phenotype in ["oscillator", "toggle_switch", "adaptation",
                             "pulse_generator", "amplifier", "stable"]:
                logger.info(f"  {phenotype}: {results.get(phenotype, 0):.1%}")

            if results["overall"] > best_oracle_acc:
                best_oracle_acc = results["overall"]
                # Save best model
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
                best_path = checkpoint_dir / "best.safetensors"
                mx.save_safetensors(str(best_path), flat_weights)

                state = {
                    "epoch": epoch,
                    "oracle_accuracy": best_oracle_acc,
                    "model_args": {
                        "vocab_size": model.args.vocab_size,
                        "embed_dim": model.args.embed_dim,
                        "num_layers": model.args.num_layers,
                        "num_heads": model.args.num_heads,
                        "mlp_dim": model.args.mlp_dim,
                        "max_seq_len": model.args.max_seq_len,
                        "dropout": model.args.dropout,
                    },
                }
                with open(best_path.with_suffix(".json"), "w") as f:
                    json.dump(state, f, indent=2)

                logger.info(f"New best oracle accuracy: {best_oracle_acc:.1%}")

    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Best oracle accuracy: {best_oracle_acc:.1%}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
