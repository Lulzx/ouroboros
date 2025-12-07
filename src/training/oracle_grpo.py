"""Oracle-augmented GRPO training with boolean network simulation."""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..model.transformer import GRNTransformer
from ..model.generation import generate, compute_sequence_log_probs
from ..data.tokenizer import GRNTokenizer
from ..simulator.boolean_network import BooleanNetwork
from ..simulator.classify_behavior import BehaviorClassifier
from ..utils.logging import get_logger

from .grpo import SelfClassifier


class OracleGRPOTrainer:
    """GRPO trainer with oracle (simulator) validation."""

    def __init__(
        self,
        model: GRNTransformer,
        ref_model: GRNTransformer,
        tokenizer: GRNTokenizer,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        kl_coef: float = 0.1,
        self_class_weight: float = 0.5,
        oracle_weight: float = 0.5,
        num_generations: int = 4,
        max_gen_length: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints/oracle_grpo",
        simulation_steps: int = 100,
        num_initial_conditions: int = 10,
    ):
        """
        Initialize Oracle GRPO trainer.

        Args:
            model: Model to train (policy)
            ref_model: Reference model for KL penalty (frozen)
            tokenizer: GRNTokenizer
            learning_rate: Learning rate
            weight_decay: Weight decay
            kl_coef: KL divergence penalty coefficient
            self_class_weight: Weight for self-classification reward
            oracle_weight: Weight for oracle (simulator) reward
            num_generations: Number of generations per phenotype
            max_gen_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            grad_clip: Gradient clipping threshold
            checkpoint_dir: Directory for checkpoints
            simulation_steps: Steps for boolean simulation
            num_initial_conditions: Initial conditions for simulation
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.self_class_weight = self_class_weight
        self.oracle_weight = oracle_weight
        self.num_generations = num_generations
        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.simulation_steps = simulation_steps
        self.num_initial_conditions = num_initial_conditions

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.self_classifier = SelfClassifier(model, tokenizer)
        self.behavior_classifier = BehaviorClassifier(
            num_initial_conditions=num_initial_conditions,
            max_steps=simulation_steps,
        )
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.phenotype_ids = mx.array(tokenizer.get_phenotype_token_ids())
        self.step = 0

        self.logger = get_logger("grn.oracle_grpo")

    def decode_sequences(self, sequences: mx.array) -> list[dict]:
        """Decode token sequences to circuit dictionaries."""
        circuits = []
        for i in range(sequences.shape[0]):
            token_ids = sequences[i].tolist()
            circuit = self.tokenizer.decode(token_ids)
            circuits.append(circuit)
        return circuits

    def compute_oracle_rewards(
        self,
        sequences: mx.array,
        intended_phenotypes: mx.array,
        use_shaped_rewards: bool = True,
    ) -> mx.array:
        """
        Compute rewards based on oracle (simulator) classification.

        Args:
            sequences: Generated sequences
            intended_phenotypes: Intended phenotype token IDs
            use_shaped_rewards: If True, use partial credit rewards

        Returns:
            Rewards array (shaped or binary based on use_shaped_rewards)
        """
        circuits = self.decode_sequences(sequences)
        rewards = []

        for i, circuit in enumerate(circuits):
            intended_id = int(intended_phenotypes[i])
            intended_name = self.tokenizer.get_phenotype_name(intended_id)

            # First check validity - reject degenerate circuits
            token_ids = sequences[i].tolist()
            if not self.tokenizer.is_valid_circuit(token_ids):
                rewards.append(0.0)
                continue

            # Parse to boolean network and classify
            try:
                if not circuit.get("interactions"):
                    rewards.append(0.0)
                    continue

                network = BooleanNetwork.from_circuit(circuit)

                if use_shaped_rewards:
                    # Use shaped rewards with partial credit
                    reward, _ = self.behavior_classifier.compute_shaped_reward(
                        network, intended_name, seed=i
                    )
                    rewards.append(reward)
                else:
                    # Binary reward
                    predicted, _ = self.behavior_classifier.classify(network, seed=i)
                    if predicted == intended_name:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
            except Exception:
                rewards.append(0.0)

        return mx.array(rewards)

    def generate_samples(
        self,
        phenotype_ids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Generate samples for each phenotype."""
        all_sequences = []
        all_phenotypes = []

        for phenotype_id in phenotype_ids:
            prompts = mx.array(
                [[self.tokenizer.bos_token_id, int(phenotype_id)]]
                * self.num_generations
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

            all_sequences.append(generated)
            all_phenotypes.extend([int(phenotype_id)] * self.num_generations)

        # Pad and concatenate
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

    def compute_combined_rewards(
        self,
        sequences: mx.array,
        intended_phenotypes: mx.array,
    ) -> tuple[mx.array, dict]:
        """
        Compute combined self-classification and oracle rewards.

        Args:
            sequences: Generated sequences
            intended_phenotypes: Intended phenotype token IDs

        Returns:
            Tuple of (combined_rewards, metrics_dict)
        """
        # Self-classification reward
        self_rewards = self.self_classifier.compute_rewards(
            sequences, intended_phenotypes
        )
        mx.eval(self_rewards)

        # Oracle reward
        oracle_rewards = self.compute_oracle_rewards(sequences, intended_phenotypes)

        # Combined reward
        combined = (
            self.self_class_weight * self_rewards
            + self.oracle_weight * oracle_rewards
        )

        metrics = {
            "self_class_reward": float(self_rewards.mean()),
            "oracle_reward": float(oracle_rewards.mean()),
            "combined_reward": float(combined.mean()),
        }

        return combined, metrics

    def compute_advantages(self, rewards: mx.array, group_size: int) -> mx.array:
        """Compute group-relative advantages."""
        num_groups = rewards.shape[0] // group_size
        advantages = []

        for i in range(num_groups):
            start = i * group_size
            end = start + group_size
            group_rewards = rewards[start:end]

            mean = group_rewards.mean()
            std = group_rewards.std() + 1e-8
            group_advantages = (group_rewards - mean) / std
            advantages.append(group_advantages)

        return mx.concatenate(advantages)

    def grpo_loss(
        self,
        model: GRNTransformer,
        sequences: mx.array,
        advantages: mx.array,
        ref_log_probs: mx.array,
    ) -> mx.array:
        """Compute GRPO loss."""
        policy_log_probs = compute_sequence_log_probs(
            model, sequences, self.tokenizer.pad_token_id
        )

        # Policy gradient loss
        pg_loss = -(advantages * policy_log_probs).mean()

        # KL penalty
        kl = policy_log_probs - ref_log_probs
        kl_loss = self.kl_coef * kl.mean()

        return pg_loss + kl_loss

    def train_step(self) -> dict:
        """Execute single training step."""
        # Generate samples
        sequences, intended_phenotypes = self.generate_samples(self.phenotype_ids)
        mx.eval(sequences, intended_phenotypes)

        # Compute combined rewards
        rewards, reward_metrics = self.compute_combined_rewards(
            sequences, intended_phenotypes
        )

        # Compute advantages
        advantages = self.compute_advantages(rewards, self.num_generations)

        # Reference model log probs
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

        # Clip and update
        grads, grad_norm = optim.clip_grad_norm(grads, self.grad_clip)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        self.step += 1

        metrics = {
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            **reward_metrics,
        }

        return metrics

    def evaluate(self, num_samples: int = 50) -> dict:
        """Evaluate on both self-classification and oracle."""
        self_correct = 0
        oracle_correct = 0
        total = 0

        per_phenotype_self = {}
        per_phenotype_oracle = {}

        for phenotype_id in self.phenotype_ids:
            phenotype_name = self.tokenizer.get_phenotype_name(int(phenotype_id))

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

            intended = mx.full((num_samples,), int(phenotype_id), dtype=mx.int32)

            # Self-classification
            self_rewards = self.self_classifier.compute_rewards(generated, intended)
            mx.eval(self_rewards)
            self_correct += int(self_rewards.sum())

            # Oracle
            oracle_rewards = self.compute_oracle_rewards(generated, intended)
            oracle_correct += int(oracle_rewards.sum())

            total += num_samples

            per_phenotype_self[phenotype_name] = float(self_rewards.mean())
            per_phenotype_oracle[phenotype_name] = float(oracle_rewards.mean())

        return {
            "self_accuracy": self_correct / total,
            "oracle_accuracy": oracle_correct / total,
            "per_phenotype_self": per_phenotype_self,
            "per_phenotype_oracle": per_phenotype_oracle,
        }

    def save_checkpoint(self, path: Optional[Path] = None):
        """Save checkpoint."""
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
            "self_class_weight": self.self_class_weight,
            "oracle_weight": self.oracle_weight,
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
        """Run Oracle GRPO training."""
        self.logger.info(f"Starting Oracle GRPO training for {num_steps} steps")
        self.logger.info(
            f"Reward weights: self={self.self_class_weight}, oracle={self.oracle_weight}"
        )

        for _ in range(num_steps):
            metrics = self.train_step()

            if self.step % log_interval == 0:
                self.logger.info(
                    f"Step {self.step} | Loss: {metrics['loss']:.4f} | "
                    f"Self: {metrics['self_class_reward']:.3f} | "
                    f"Oracle: {metrics['oracle_reward']:.3f}"
                )

            if self.step % eval_interval == 0:
                eval_metrics = self.evaluate(num_samples=30)
                self.logger.info(
                    f"Eval @ Step {self.step} | "
                    f"Self Acc: {eval_metrics['self_accuracy']:.3f} | "
                    f"Oracle Acc: {eval_metrics['oracle_accuracy']:.3f}"
                )

            if self.step % checkpoint_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        self.logger.info("Oracle GRPO training complete!")
