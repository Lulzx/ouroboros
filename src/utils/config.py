"""Configuration utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 150
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.1


@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    max_length: int = 64


@dataclass
class SupervisedConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    epochs: int = 20
    grad_clip: float = 1.0


@dataclass
class GRPOConfig:
    batch_size: int = 16
    num_generations: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    kl_coef: float = 0.1
    training_steps: int = 5000
    eval_interval: int = 100
    checkpoint_interval: int = 500


@dataclass
class OracleGRPOConfig:
    self_class_weight: float = 0.5
    oracle_weight: float = 0.5
    simulation_steps: int = 100
    num_initial_conditions: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    oracle_grpo: OracleGRPOConfig = field(default_factory=OracleGRPOConfig)


def load_config(config_dir: str | Path = "configs") -> Config:
    """Load configuration from YAML files."""
    config_dir = Path(config_dir)
    config = Config()

    # Load model config
    model_path = config_dir / "model.yaml"
    if model_path.exists():
        with open(model_path) as f:
            data = yaml.safe_load(f)
            if "model" in data:
                config.model = ModelConfig(**data["model"])
            if "generation" in data:
                config.generation = GenerationConfig(**data["generation"])

    # Load training config
    training_path = config_dir / "training.yaml"
    if training_path.exists():
        with open(training_path) as f:
            data = yaml.safe_load(f)
            if "supervised" in data:
                config.supervised = SupervisedConfig(**data["supervised"])
            if "grpo" in data:
                config.grpo = GRPOConfig(**data["grpo"])
            if "oracle_grpo" in data:
                config.oracle_grpo = OracleGRPOConfig(**data["oracle_grpo"])

    return config
