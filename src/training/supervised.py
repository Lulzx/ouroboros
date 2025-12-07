"""Supervised pretraining for GRN transformer."""

import json
import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..model.transformer import GRNTransformer
from ..data.dataset import GRNDataLoader
from ..utils.logging import get_logger


def cross_entropy_loss(
    logits: mx.array,
    targets: mx.array,
    pad_token_id: int = 0,
) -> mx.array:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab)
        targets: Target token IDs of shape (batch, seq_len)
        pad_token_id: Token ID to ignore in loss computation

    Returns:
        Scalar loss value
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]

    # Flatten
    B, L, V = shift_logits.shape
    flat_logits = shift_logits.reshape(B * L, V)
    flat_targets = shift_targets.reshape(B * L)

    # Compute cross-entropy
    loss = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none")

    # Mask padding
    mask = (flat_targets != pad_token_id).astype(mx.float32)
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)

    return loss


class SupervisedTrainer:
    """Trainer for supervised pretraining."""

    def __init__(
        self,
        model: GRNTransformer,
        train_loader: GRNDataLoader,
        val_loader: Optional[GRNDataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        total_steps: Optional[int] = None,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints/supervised",
        pad_token_id: int = 0,
    ):
        """
        Initialize trainer.

        Args:
            model: GRNTransformer to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Peak learning rate
            weight_decay: Weight decay coefficient
            warmup_steps: Number of warmup steps
            total_steps: Total training steps (for LR schedule)
            grad_clip: Gradient clipping threshold
            checkpoint_dir: Directory for saving checkpoints
            pad_token_id: Padding token ID
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pad_token_id = pad_token_id

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        self.logger = get_logger("grn.supervised")

    def get_lr(self) -> float:
        """Get current learning rate with warmup and cosine decay."""
        if self.step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * (self.step + 1) / self.warmup_steps

        if self.total_steps is None:
            return self.learning_rate

        # Cosine decay
        progress = (self.step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def update_lr(self):
        """Update optimizer learning rate."""
        lr = self.get_lr()
        self.optimizer.learning_rate = lr
        return lr

    def loss_fn(self, model: GRNTransformer, batch: dict) -> mx.array:
        """Compute loss for a batch."""
        input_ids = batch["input_ids"]
        logits = model.forward(input_ids)
        return cross_entropy_loss(logits, input_ids, self.pad_token_id)

    def train_step(self, batch: dict) -> float:
        """Execute single training step."""
        # Update learning rate
        lr = self.update_lr()

        # Compute loss and gradients
        loss, grads = nn.value_and_grad(self.model, self.loss_fn)(self.model, batch)

        # Clip gradients
        grads, grad_norm = optim.clip_grad_norm(grads, self.grad_clip)

        # Update parameters
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        self.step += 1

        return float(loss)

    def validate(self) -> float:
        """Run validation and return average loss."""
        if self.val_loader is None:
            return 0.0

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            logits = self.model.forward(batch["input_ids"])
            loss = cross_entropy_loss(logits, batch["input_ids"], self.pad_token_id)
            total_loss += float(loss)
            num_batches += 1
            mx.eval(loss)

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: Optional[Path] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_epoch{self.epoch}.safetensors"

        # Save model weights - flatten nested dict for safetensors
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

        flat_weights = flatten_params(self.model.parameters())
        mx.save_safetensors(str(path), flat_weights)

        # Save training state
        state = {
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "model_args": {
                "vocab_size": self.model.args.vocab_size,
                "embed_dim": self.model.args.embed_dim,
                "num_layers": self.model.args.num_layers,
                "num_heads": self.model.args.num_heads,
                "mlp_dim": self.model.args.mlp_dim,
                "max_seq_len": self.model.args.max_seq_len,
                "dropout": self.model.args.dropout,
            },
        }
        state_path = path.with_suffix(".json")
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        if is_best:
            best_path = self.checkpoint_dir / "best.safetensors"
            mx.save_safetensors(str(best_path), flat_weights)
            with open(best_path.with_suffix(".json"), "w") as f:
                json.dump(state, f, indent=2)

        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        weights = mx.load(str(path))
        self.model.load_weights(list(weights.items()))

        state_path = path.with_suffix(".json")
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.step = state.get("step", 0)
            self.epoch = state.get("epoch", 0)
            self.best_val_loss = state.get("best_val_loss", float("inf"))

        self.logger.info(f"Loaded checkpoint from {path}")

    def train(self, epochs: int, log_interval: int = 10):
        """
        Run training loop.

        Args:
            epochs: Number of epochs to train
            log_interval: Log every N steps
        """
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Model parameters: {self.model.num_parameters:,}")

        # Calculate total steps for LR schedule
        steps_per_epoch = len(self.train_loader)
        if self.total_steps is None:
            self.total_steps = epochs * steps_per_epoch

        for epoch in range(epochs):
            self.epoch = epoch + 1
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1

                if self.step % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.get_lr()
                    self.logger.info(
                        f"Epoch {self.epoch} | Step {self.step} | "
                        f"Loss: {loss:.4f} | Avg: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

            # Epoch complete
            train_loss = epoch_loss / max(num_batches, 1)
            val_loss = self.validate()

            self.logger.info(
                f"Epoch {self.epoch} complete | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(is_best=is_best)

        self.logger.info("Training complete!")
        return self.best_val_loss
