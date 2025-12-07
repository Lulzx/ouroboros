"""GRPO (Group Relative Policy Optimization) training with self-classification."""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..model.transformer import GRNTransformer
from ..model.generation import generate, compute_sequence_log_probs
from ..data.tokenizer import GRNTokenizer
from ..utils.logging import get_logger


class SelfClassifier:
    """Self-classification using model likelihoods."""

    def __init__(self, model: GRNTransformer, tokenizer: GRNTokenizer):
        """
        Initialize self-classifier.

        Args:
            model: GRNTransformer model
            tokenizer: GRNTokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.phenotype_ids = tokenizer.get_phenotype_token_ids()

    def classify(self, sequences: mx.array) -> tuple[mx.array, mx.array]:
        """
        Self-classify sequences by computing likelihood under each phenotype.

        For each sequence (without phenotype), compute log-likelihood when
        prepending each phenotype token. The phenotype with highest likelihood
        is the predicted class.

        Args:
            sequences: Token sequences of shape (batch, seq_len)
                      Expected format: <bos> <phenotype> ... <eos>

        Returns:
            Tuple of (predicted_phenotype_ids, log_likelihoods)
            - predicted_phenotype_ids: Shape (batch,)
            - log_likelihoods: Shape (batch, num_phenotypes)
        """
        B, L = sequences.shape

        # Extract sequences without the phenotype token (keep bos and rest)
        # Format: <bos> gene interaction gene ... <eos>
        # We'll prepend each phenotype after <bos>
        bos = sequences[:, 0:1]  # (B, 1)
        rest = sequences[:, 2:]  # (B, L-2) - skip original phenotype

        log_likelihoods = []

        for phenotype_id in self.phenotype_ids:
            # Create sequence with this phenotype
            phenotype_token = mx.full((B, 1), phenotype_id, dtype=mx.int32)
            candidate_seq = mx.concatenate([bos, phenotype_token, rest], axis=1)

            # Compute log-likelihood
            log_prob = compute_sequence_log_probs(
                self.model,
                candidate_seq,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            log_likelihoods.append(log_prob)

        # Stack log-likelihoods: (batch, num_phenotypes)
        log_likelihoods = mx.stack(log_likelihoods, axis=1)

        # Predict class with highest likelihood
        predicted_idx = mx.argmax(log_likelihoods, axis=1)
        predicted_phenotype_ids = mx.array(self.phenotype_ids)[predicted_idx]

        return predicted_phenotype_ids, log_likelihoods

    def compute_rewards(
        self,
        sequences: mx.array,
        intended_phenotype_ids: mx.array,
    ) -> mx.array:
        """
        Compute rewards based on self-classification agreement and validity.

        Args:
            sequences: Generated sequences of shape (batch, seq_len)
            intended_phenotype_ids: The phenotype each sequence was generated for

        Returns:
            Rewards of shape (batch,) - 1.0 if valid and match, 0.0 otherwise
        """
        predicted_ids, _ = self.classify(sequences)

        # Check validity for each sequence
        batch_size = sequences.shape[0]
        validity_mask = []
        for i in range(batch_size):
            token_ids = sequences[i].tolist()
            is_valid = self.tokenizer.is_valid_circuit(token_ids)
            validity_mask.append(1.0 if is_valid else 0.0)
        validity_mask = mx.array(validity_mask)

        # Reward is 1 only if valid AND prediction matches intended phenotype
        classification_match = (predicted_ids == intended_phenotype_ids).astype(mx.float32)
        rewards = classification_match * validity_mask

        return rewards


class GRPOTrainer:
    """Trainer for GRPO with self-classification reward."""

    def __init__(
        self,
        model: GRNTransformer,
        ref_model: GRNTransformer,
        tokenizer: GRNTokenizer,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        kl_coef: float = 0.1,
        num_generations: int = 4,
        max_gen_length: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints/grpo",
        entropy_coef: float = 0.01,
    ):
        """
        Initialize GRPO trainer.

        Args:
            model: Model to train (policy)
            ref_model: Reference model for KL penalty (frozen)
            tokenizer: GRNTokenizer
            learning_rate: Learning rate
            weight_decay: Weight decay
            kl_coef: KL divergence penalty coefficient
            num_generations: Number of generations per phenotype
            max_gen_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            grad_clip: Gradient clipping threshold
            checkpoint_dir: Directory for checkpoints
            entropy_coef: Entropy bonus coefficient
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.num_generations = num_generations
        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.entropy_coef = entropy_coef

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize classifier and optimizer
        self.classifier = SelfClassifier(model, tokenizer)
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.phenotype_ids = mx.array(tokenizer.get_phenotype_token_ids())
        self.step = 0

        self.logger = get_logger("grn.grpo")

    def generate_samples(
        self,
        phenotype_ids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Generate samples for each phenotype.

        Args:
            phenotype_ids: Phenotype token IDs to generate for

        Returns:
            Tuple of (generated_sequences, intended_phenotypes)
            - generated_sequences: (num_phenotypes * num_generations, seq_len)
            - intended_phenotypes: (num_phenotypes * num_generations,)
        """
        all_sequences = []
        all_phenotypes = []

        for phenotype_id in phenotype_ids:
            # Create prompts: <bos> <phenotype>
            prompts = mx.array(
                [[self.tokenizer.bos_token_id, int(phenotype_id)]]
                * self.num_generations
            )

            # Generate
            generated = generate(
                self.model,
                prompts,
                max_length=self.max_gen_length,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            all_sequences.append(generated)
            all_phenotypes.extend([int(phenotype_id)] * self.num_generations)

        # Concatenate all sequences
        # Pad to same length
        max_len = max(s.shape[1] for s in all_sequences)
        padded = []
        for seq in all_sequences:
            if seq.shape[1] < max_len:
                pad = mx.full(
                    (seq.shape[0], max_len - seq.shape[1]),
                    self.tokenizer.pad_token_id,
                    dtype=mx.int32,
                )
                seq = mx.concatenate([seq, pad], axis=1)
            padded.append(seq)

        sequences = mx.concatenate(padded, axis=0)
        phenotypes = mx.array(all_phenotypes)

        return sequences, phenotypes

    def compute_advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        """
        Compute group-relative advantages.

        Args:
            rewards: Raw rewards of shape (batch,)
            group_size: Size of each group (num_generations)

        Returns:
            Advantages of shape (batch,)
        """
        num_groups = rewards.shape[0] // group_size
        advantages = []

        for i in range(num_groups):
            start = i * group_size
            end = start + group_size
            group_rewards = rewards[start:end]

            # Normalize within group
            mean = group_rewards.mean()
            std = group_rewards.std() + 1e-8
            group_advantages = (group_rewards - mean) / std
            advantages.append(group_advantages)

        return mx.concatenate(advantages)

    def compute_kl_penalty(
        self,
        sequences: mx.array,
        policy_log_probs: mx.array,
    ) -> mx.array:
        """
        Compute KL divergence penalty from reference model.

        Args:
            sequences: Generated sequences
            policy_log_probs: Log probs from current policy

        Returns:
            KL penalty per sequence
        """
        # Get reference model log probs
        ref_log_probs = compute_sequence_log_probs(
            self.ref_model,
            sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # KL = policy_log_prob - ref_log_prob
        # We want to penalize deviation from reference
        kl = policy_log_probs - ref_log_probs

        return kl

    def grpo_loss(
        self,
        model: GRNTransformer,
        sequences: mx.array,
        advantages: mx.array,
        ref_log_probs: mx.array,
    ) -> mx.array:
        """
        Compute GRPO loss.

        Args:
            model: Policy model
            sequences: Generated sequences
            advantages: Group-relative advantages
            ref_log_probs: Reference model log probabilities

        Returns:
            Scalar loss
        """
        # Get current policy log probs
        policy_log_probs = compute_sequence_log_probs(
            model, sequences, self.tokenizer.pad_token_id
        )

        # Policy gradient loss: -advantage * log_prob
        pg_loss = -(advantages * policy_log_probs).mean()

        # KL penalty
        kl = policy_log_probs - ref_log_probs
        kl_loss = self.kl_coef * kl.mean()

        # Total loss
        total_loss = pg_loss + kl_loss

        return total_loss

    def train_step(self) -> dict:
        """
        Execute single GRPO training step.

        Returns:
            Dictionary of metrics
        """
        # Generate samples for all phenotypes
        sequences, intended_phenotypes = self.generate_samples(self.phenotype_ids)
        mx.eval(sequences, intended_phenotypes)

        # Compute rewards via self-classification
        rewards = self.classifier.compute_rewards(sequences, intended_phenotypes)
        mx.eval(rewards)

        # Compute group-relative advantages
        advantages = self.compute_advantages(rewards, self.num_generations)

        # Get reference model log probs (frozen)
        ref_log_probs = compute_sequence_log_probs(
            self.ref_model,
            sequences,
            self.tokenizer.pad_token_id,
        )
        mx.eval(ref_log_probs)

        # Compute loss and gradients
        loss, grads = nn.value_and_grad(self.model, self.grpo_loss)(
            self.model, sequences, advantages, ref_log_probs
        )

        # Clip gradients
        grads, grad_norm = optim.clip_grad_norm(grads, self.grad_clip)

        # Update parameters
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        self.step += 1

        # Compute metrics
        accuracy = float(rewards.mean())
        avg_reward = float(rewards.mean())

        # Per-phenotype accuracy
        per_phenotype_acc = {}
        for i, pid in enumerate(self.phenotype_ids):
            start = i * self.num_generations
            end = start + self.num_generations
            phenotype_name = self.tokenizer.get_phenotype_name(int(pid))
            per_phenotype_acc[phenotype_name] = float(rewards[start:end].mean())

        return {
            "loss": float(loss),
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "per_phenotype_acc": per_phenotype_acc,
            "grad_norm": float(grad_norm),
        }

    def evaluate(self, num_samples: int = 100) -> dict:
        """
        Evaluate model on self-classification accuracy.

        Args:
            num_samples: Number of samples per phenotype

        Returns:
            Dictionary of metrics
        """
        total_correct = 0
        total_samples = 0
        per_phenotype = {}

        for phenotype_id in self.phenotype_ids:
            phenotype_name = self.tokenizer.get_phenotype_name(int(phenotype_id))

            # Generate samples
            prompts = mx.array(
                [[self.tokenizer.bos_token_id, int(phenotype_id)]] * num_samples
            )

            generated = generate(
                self.model,
                prompts,
                max_length=self.max_gen_length,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Compute rewards
            intended = mx.full((num_samples,), int(phenotype_id), dtype=mx.int32)
            rewards = self.classifier.compute_rewards(generated, intended)
            mx.eval(rewards)

            correct = int(rewards.sum())
            total_correct += correct
            total_samples += num_samples
            per_phenotype[phenotype_name] = correct / num_samples

        return {
            "accuracy": total_correct / total_samples,
            "per_phenotype": per_phenotype,
        }

    def save_checkpoint(self, path: Optional[Path] = None):
        """Save model checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_step{self.step}.safetensors"

        # Flatten nested dict for safetensors
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

        state = {
            "step": self.step,
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
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Saved checkpoint to {path}")

    def train(
        self,
        num_steps: int,
        log_interval: int = 10,
        eval_interval: int = 100,
        checkpoint_interval: int = 500,
    ):
        """
        Run GRPO training.

        Args:
            num_steps: Total training steps
            log_interval: Log every N steps
            eval_interval: Evaluate every N steps
            checkpoint_interval: Save checkpoint every N steps
        """
        self.logger.info(f"Starting GRPO training for {num_steps} steps")

        for _ in range(num_steps):
            metrics = self.train_step()

            if self.step % log_interval == 0:
                self.logger.info(
                    f"Step {self.step} | Loss: {metrics['loss']:.4f} | "
                    f"Acc: {metrics['accuracy']:.3f}"
                )

            if self.step % eval_interval == 0:
                eval_metrics = self.evaluate(num_samples=50)
                self.logger.info(
                    f"Eval @ Step {self.step} | Accuracy: {eval_metrics['accuracy']:.3f}"
                )
                for name, acc in eval_metrics["per_phenotype"].items():
                    self.logger.info(f"  {name}: {acc:.3f}")

            if self.step % checkpoint_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        self.logger.info("GRPO training complete!")
