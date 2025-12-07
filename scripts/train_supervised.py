#!/usr/bin/env python3
"""Train GRN transformer with supervised pretraining."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from src.data.tokenizer import GRNTokenizer
from src.data.dataset import GRNDataset, GRNDataLoader, load_circuits
from src.model.transformer import GRNTransformer, ModelArgs
from src.training.supervised import SupervisedTrainer
from src.utils.config import load_config
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Supervised pretraining")
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
        "--checkpoint-dir",
        type=str,
        default="checkpoints/supervised",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation",
    )
    args = parser.parse_args()

    # Setup
    logger = setup_logger("grn.supervised", log_file=f"{args.checkpoint_dir}/train.log")
    config = load_config(args.config_dir)

    # Override config with command line args
    epochs = args.epochs or config.supervised.epochs
    batch_size = args.batch_size or config.supervised.batch_size
    learning_rate = args.learning_rate or config.supervised.learning_rate

    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer_path = data_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = GRNTokenizer.load(tokenizer_path)
    else:
        logger.warning("Tokenizer not found, creating default")
        tokenizer = GRNTokenizer()

    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Load data
    train_circuits = load_circuits(data_dir / "splits" / "train.json")
    val_circuits = load_circuits(data_dir / "splits" / "val.json")

    logger.info(f"Training circuits: {len(train_circuits)}")
    logger.info(f"Validation circuits: {len(val_circuits)}")

    # Create datasets
    train_dataset = GRNDataset(
        train_circuits,
        tokenizer,
        max_length=config.model.max_seq_len,
        augment=args.augment,
    )
    val_dataset = GRNDataset(
        val_circuits,
        tokenizer,
        max_length=config.model.max_seq_len,
        augment=False,
    )

    # Create dataloaders
    train_loader = GRNDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GRNDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
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

    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=config.supervised.weight_decay,
        warmup_steps=config.supervised.warmup_steps,
        grad_clip=config.supervised.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Train
    logger.info(f"Starting training for {epochs} epochs")
    best_val_loss = trainer.train(epochs=epochs, log_interval=args.log_interval)

    logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
