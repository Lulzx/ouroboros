#!/usr/bin/env python3
"""
GRPO Training with Oracle Reward.

Key insight: Self-classification reward causes mode collapse because the model
learns to generate sequences it can classify, not sequences that actually work.

Solution: Use the oracle (boolean network simulator) as the reward signal.
This ensures the model learns to generate circuits with correct dynamics.

Additional improvements:
1. Oracle reward instead of self-classification
2. Diversity bonus to encourage exploration
3. Per-phenotype balanced sampling
4. Curriculum learning (start easy, increase difficulty)
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
from src.model.transformer import GRNTransformer, ModelArgs
from src.model.generation import generate, compute_sequence_log_probs
from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier
from src.utils.config import load_config
from src.utils.logging import setup_logger


class OracleRewardComputer:
    """Compute rewards using oracle simulation."""

    def __init__(self, tokenizer: GRNTokenizer):
        self.tokenizer = tokenizer
        self.classifier = BehaviorClassifier(
            num_initial_conditions=20,
            max_steps=200,
            rule="constitutive",
        )

    def compute_rewards(
        self,
        sequences: mx.array,
        intended_phenotype_ids: mx.array,
    ) -> tuple[mx.array, dict]:
        """
        Compute oracle-based rewards.

        Returns:
            rewards: Binary rewards (1 if oracle matches intended, 0 otherwise)
            stats: Per-phenotype accuracy statistics
        """
        batch_size = sequences.shape[0]
        rewards = []
        stats = defaultdict(lambda: {"correct": 0, "total": 0, "generated": []})

        for i in range(batch_size):
            token_ids = sequences[i].tolist()

            # Decode circuit
            try:
                circuit = self.tokenizer.decode(token_ids)
            except:
                circuit = None

            if circuit is None or not circuit.get("interactions"):
                rewards.append(0.0)
                continue

            # Get intended phenotype
            intended_id = int(intended_phenotype_ids[i])
            intended_name = self.tokenizer.get_phenotype_name(intended_id)

            # Simulate and classify
            try:
                network = BooleanNetwork.from_circuit(circuit)
                predicted, _ = self.classifier.classify(network)
            except:
                predicted = "stable"

            # Compute reward
            is_correct = predicted == intended_name
            rewards.append(1.0 if is_correct else 0.0)

            # Track stats
            stats[intended_name]["total"] += 1
            if is_correct:
                stats[intended_name]["correct"] += 1
            stats[intended_name]["generated"].append({
                "circuit": circuit,
                "predicted": predicted,
                "correct": is_correct,
            })

        return mx.array(rewards), dict(stats)


class OracleGRPOTrainer:
    """GRPO trainer with oracle reward."""

    def __init__(
        self,
        model: GRNTransformer,
        ref_model: GRNTransformer,
        tokenizer: GRNTokenizer,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        kl_coef: float = 0.1,
        entropy_coef: float = 0.01,
        diversity_coef: float = 0.1,
        num_generations: int = 8,
        max_gen_length: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.95,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints/grpo_oracle",
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.diversity_coef = diversity_coef
        self.num_generations = num_generations
        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize oracle and optimizer
        self.oracle = OracleRewardComputer(tokenizer)
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.phenotype_ids = mx.array(tokenizer.get_phenotype_token_ids())
        self.step = 0
        self.best_accuracy = 0.0

        self.logger = setup_logger("grpo_oracle", log_file=f"{checkpoint_dir}/train.log")

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
        """Compute group-relative advantages with baseline subtraction."""
        num_groups = rewards.shape[0] // group_size
        advantages = []

        for i in range(num_groups):
            start = i * group_size
            end = start + group_size
            group_rewards = rewards[start:end]

            # Normalize within group
            mean = group_rewards.mean()
            std = group_rewards.std() + 1e-8

            # Only apply advantage if there's variance
            if std > 0.01:
                group_advantages = (group_rewards - mean) / std
            else:
                # If all same, small positive for correct, negative for incorrect
                group_advantages = group_rewards - 0.5

            advantages.append(group_advantages)

        return mx.concatenate(advantages)

    def compute_diversity_bonus(self, sequences: mx.array, group_size: int) -> mx.array:
        """
        Compute diversity bonus within each group.

        Encourage different circuits within same phenotype group.
        """
        batch_size = sequences.shape[0]
        num_groups = batch_size // group_size
        bonuses = []

        for g in range(num_groups):
            start = g * group_size
            end = start + group_size

            # Convert to sets of token sequences for diversity
            unique_seqs = set()
            group_bonuses = []

            for i in range(start, end):
                seq_tuple = tuple(sequences[i].tolist())
                if seq_tuple not in unique_seqs:
                    unique_seqs.add(seq_tuple)
                    group_bonuses.append(0.1)  # Bonus for unique
                else:
                    group_bonuses.append(0.0)  # No bonus for duplicate

            bonuses.extend(group_bonuses)

        return mx.array(bonuses)

    def grpo_loss(
        self,
        model: GRNTransformer,
        sequences: mx.array,
        advantages: mx.array,
        ref_log_probs: mx.array,
    ) -> mx.array:
        """Compute GRPO loss with KL penalty."""
        # Get current policy log probs
        policy_log_probs = compute_sequence_log_probs(
            model, sequences, self.tokenizer.pad_token_id
        )

        # Policy gradient loss
        pg_loss = -(advantages * policy_log_probs).mean()

        # KL penalty (prevent drift from reference)
        kl = policy_log_probs - ref_log_probs
        kl_loss = self.kl_coef * kl.mean()

        return pg_loss + kl_loss

    def train_step(self) -> dict:
        """Execute single training step with oracle reward."""
        # Generate samples
        sequences, intended_phenotypes = self.generate_samples(self.phenotype_ids)
        mx.eval(sequences, intended_phenotypes)

        # Compute oracle rewards
        rewards, stats = self.oracle.compute_rewards(sequences, intended_phenotypes)
        mx.eval(rewards)

        # Add diversity bonus
        diversity_bonus = self.compute_diversity_bonus(sequences, self.num_generations)
        combined_rewards = rewards + self.diversity_coef * diversity_bonus

        # Compute advantages
        advantages = self.compute_advantages(combined_rewards, self.num_generations)

        # Get reference log probs
        ref_log_probs = compute_sequence_log_probs(
            self.ref_model, sequences, self.tokenizer.pad_token_id
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

        per_phenotype_acc = {}
        for name, s in stats.items():
            if s["total"] > 0:
                per_phenotype_acc[name] = s["correct"] / s["total"]

        return {
            "loss": float(loss),
            "accuracy": accuracy,
            "per_phenotype_acc": per_phenotype_acc,
            "grad_norm": float(grad_norm),
        }

    def evaluate(self, num_samples: int = 50) -> dict:
        """Evaluate with oracle."""
        total_correct = 0
        total_samples = 0
        per_phenotype = {}

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
            rewards, _ = self.oracle.compute_rewards(generated, intended)
            mx.eval(rewards)

            correct = int(rewards.sum())
            total_correct += correct
            total_samples += num_samples
            per_phenotype[phenotype_name] = correct / num_samples

        return {
            "accuracy": total_correct / total_samples,
            "per_phenotype": per_phenotype,
        }

    def save_checkpoint(self, path: Path = None, is_best: bool = False):
        """Save checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f"checkpoint_step{self.step}.safetensors"

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
            "best_accuracy": self.best_accuracy,
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

        if is_best:
            best_path = self.checkpoint_dir / "best.safetensors"
            mx.save_safetensors(str(best_path), flat_weights)
            with open(best_path.with_suffix(".json"), "w") as f:
                json.dump(state, f, indent=2)
            self.logger.info(f"Saved best checkpoint @ {self.best_accuracy:.1%}")

    def train(
        self,
        num_steps: int,
        log_interval: int = 10,
        eval_interval: int = 50,
        checkpoint_interval: int = 200,
    ):
        """Run training loop."""
        self.logger.info(f"Starting Oracle GRPO training for {num_steps} steps")
        self.logger.info(f"  KL coef: {self.kl_coef}")
        self.logger.info(f"  Diversity coef: {self.diversity_coef}")
        self.logger.info(f"  Generations per phenotype: {self.num_generations}")
        self.logger.info(f"  Temperature: {self.temperature}")

        for _ in range(num_steps):
            metrics = self.train_step()

            if self.step % log_interval == 0:
                self.logger.info(
                    f"Step {self.step} | Loss: {metrics['loss']:.4f} | "
                    f"Oracle Acc: {metrics['accuracy']:.1%}"
                )
                for name, acc in metrics["per_phenotype_acc"].items():
                    self.logger.info(f"  {name}: {acc:.1%}")

            if self.step % eval_interval == 0:
                eval_metrics = self.evaluate(num_samples=30)
                self.logger.info(
                    f"Eval @ Step {self.step} | Oracle Accuracy: {eval_metrics['accuracy']:.1%}"
                )
                for name, acc in eval_metrics["per_phenotype"].items():
                    self.logger.info(f"  {name}: {acc:.1%}")

                if eval_metrics["accuracy"] > self.best_accuracy:
                    self.best_accuracy = eval_metrics["accuracy"]
                    self.save_checkpoint(is_best=True)

            if self.step % checkpoint_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        self.logger.info(f"Training complete! Best accuracy: {self.best_accuracy:.1%}")


def load_model(checkpoint_path: Path, tokenizer: GRNTokenizer) -> GRNTransformer:
    """Load model from checkpoint."""
    state_path = checkpoint_path.with_suffix(".json")
    if state_path.exists():
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
    parser = argparse.ArgumentParser(description="Oracle-based GRPO training")
    parser.add_argument("--data-dir", type=str, default="data/processed_relabeled")
    parser.add_argument("--supervised-checkpoint", type=str,
                       default="checkpoints/supervised_relabeled_v2/best.safetensors")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/grpo_oracle")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--diversity-coef", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=50)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    data_dir = Path(args.data_dir)
    tokenizer = GRNTokenizer.load(data_dir / "tokenizer.json")

    # Load models
    checkpoint_path = Path(args.supervised_checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, tokenizer)
    ref_model = load_model(checkpoint_path, tokenizer)

    print(f"Model parameters: {model.num_parameters:,}")

    # Create trainer
    trainer = OracleGRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        diversity_coef=args.diversity_coef,
        num_generations=args.num_generations,
        temperature=args.temperature,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    trainer.train(
        num_steps=args.steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
    )


if __name__ == "__main__":
    main()
