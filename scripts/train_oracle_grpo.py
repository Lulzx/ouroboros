#!/usr/bin/env python3
"""Train GRN transformer with Oracle-augmented GRPO."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from src.data.tokenizer import GRNTokenizer
from src.model.transformer import GRNTransformer, ModelArgs
from src.training.oracle_grpo import OracleGRPOTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logger


def load_model(checkpoint_path: Path, tokenizer: GRNTokenizer) -> GRNTransformer:
    """Load model from checkpoint."""
    state_path = checkpoint_path.with_suffix(".json")
    if state_path.exists():
        import json
        with open(state_path) as f:
            state = json.load(f)
        model_args = ModelArgs(**state["model_args"])
    else:
        model_args = ModelArgs(vocab_size=tokenizer.vocab_size)

    model = GRNTransformer(model_args)
    weights = mx.load(str(checkpoint_path))
    model.load_weights(list(weights.items()))

    return model


def main():
    parser = argparse.ArgumentParser(description="Oracle GRPO training")
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
        "--checkpoint",
        type=str,
        default="checkpoints/grpo/best.safetensors",
        help="Path to GRPO checkpoint (or supervised if starting fresh)",
    )
    parser.add_argument(
        "--ref-checkpoint",
        type=str,
        default=None,
        help="Path to reference model checkpoint (defaults to same as --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/oracle_grpo",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of training steps",
    )
    parser.add_argument(
        "--self-class-weight",
        type=float,
        default=None,
        help="Override self-classification reward weight",
    )
    parser.add_argument(
        "--oracle-weight",
        type=float,
        default=None,
        help="Override oracle reward weight",
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
    logger = setup_logger("grn.oracle_grpo", log_file=f"{args.checkpoint_dir}/train.log")
    config = load_config(args.config_dir)

    # Override config
    steps = args.steps or config.grpo.training_steps
    self_class_weight = args.self_class_weight or config.oracle_grpo.self_class_weight
    oracle_weight = args.oracle_weight or config.oracle_grpo.oracle_weight

    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer_path = data_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = GRNTokenizer.load(tokenizer_path)
    else:
        tokenizer = GRNTokenizer()

    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, tokenizer)
    logger.info(f"Model parameters: {model.num_parameters:,}")

    # Load reference model
    ref_checkpoint = Path(args.ref_checkpoint) if args.ref_checkpoint else checkpoint_path
    logger.info(f"Loading reference model from {ref_checkpoint}")
    ref_model = load_model(ref_checkpoint, tokenizer)

    # Create trainer
    trainer = OracleGRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        learning_rate=config.grpo.learning_rate,
        weight_decay=config.grpo.weight_decay,
        kl_coef=config.grpo.kl_coef,
        self_class_weight=self_class_weight,
        oracle_weight=oracle_weight,
        num_generations=config.grpo.num_generations,
        max_gen_length=config.generation.max_length,
        temperature=args.temperature,
        top_p=config.generation.top_p,
        checkpoint_dir=args.checkpoint_dir,
        simulation_steps=config.oracle_grpo.simulation_steps,
        num_initial_conditions=config.oracle_grpo.num_initial_conditions,
    )

    # Train
    logger.info(f"Starting Oracle GRPO training for {steps} steps")
    logger.info(f"Reward weights: self={self_class_weight}, oracle={oracle_weight}")
    trainer.train(
        num_steps=steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=config.grpo.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
