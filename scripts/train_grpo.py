#!/usr/bin/env python3
"""Train GRN transformer with GRPO (self-classification reward)."""

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from src.data.tokenizer import GRNTokenizer
from src.model.transformer import GRNTransformer, ModelArgs
from src.training.grpo import GRPOTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logger


def load_model(checkpoint_path: Path, tokenizer: GRNTokenizer) -> GRNTransformer:
    """Load model from checkpoint."""
    # Load state to get model args
    state_path = checkpoint_path.with_suffix(".json")
    if state_path.exists():
        import json
        with open(state_path) as f:
            state = json.load(f)
        model_args = ModelArgs(**state["model_args"])
    else:
        # Use defaults with tokenizer vocab size
        model_args = ModelArgs(vocab_size=tokenizer.vocab_size)

    model = GRNTransformer(model_args)

    # Load weights
    weights = mx.load(str(checkpoint_path))
    model.load_weights(list(weights.items()))

    return model


def main():
    parser = argparse.ArgumentParser(description="GRPO training with self-classification")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Configuration directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Processed data directory",
    )
    parser.add_argument(
        "--supervised-checkpoint",
        type=str,
        default="checkpoints/supervised/best.safetensors",
        help="Path to supervised pretrained checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/grpo",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of training steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=None,
        help="Override KL coefficient",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Override generations per phenotype",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate every N steps",
    )
    args = parser.parse_args()

    # Setup
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("grn.grpo", log_file=f"{args.checkpoint_dir}/train.log")
    config = load_config(args.config_dir)

    # Override config
    steps = args.steps or config.grpo.training_steps
    learning_rate = args.learning_rate or config.grpo.learning_rate
    kl_coef = args.kl_coef or config.grpo.kl_coef
    num_generations = args.num_generations or config.grpo.num_generations

    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer_path = data_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = GRNTokenizer.load(tokenizer_path)
    else:
        tokenizer = GRNTokenizer()

    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Load pretrained model
    checkpoint_path = Path(args.supervised_checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please run supervised pretraining first")
        sys.exit(1)

    logger.info(f"Loading pretrained model from {checkpoint_path}")
    model = load_model(checkpoint_path, tokenizer)
    logger.info(f"Model parameters: {model.num_parameters:,}")

    # Create reference model (frozen copy)
    logger.info("Creating reference model (frozen)")
    ref_model = load_model(checkpoint_path, tokenizer)

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        weight_decay=config.grpo.weight_decay,
        kl_coef=kl_coef,
        num_generations=num_generations,
        max_gen_length=config.generation.max_length,
        temperature=args.temperature,
        top_p=config.generation.top_p,
        grad_clip=config.supervised.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    logger.info(f"Starting GRPO training for {steps} steps")
    trainer.train(
        num_steps=steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=config.grpo.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
