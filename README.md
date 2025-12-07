# Ouroboros

**Self-Classifying Autoregressive Generation of Gene Regulatory Circuits**

Ouroboros is a framework for generating gene regulatory network (GRN) circuits conditioned on desired dynamical behaviors. The system uses a single autoregressive transformer that both generates and classifies circuits, enabling training through Group Relative Policy Optimization (GRPO) with self-consistency as the reward signal.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Method](#method)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Data](#data)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Project Structure](#project-structure)
10. [Citation](#citation)
11. [License](#license)

---

## Introduction

Designing gene regulatory circuits with specific dynamical behaviors is a fundamental challenge in synthetic biology. Given a target phenotype such as oscillation or bistability, the goal is to produce a circuit topology that reliably exhibits that behavior.

Ouroboros addresses this problem through conditional autoregressive generation. The key insight is that a single model can serve as both generator and classifier: by computing the likelihood of a circuit under different phenotype labels, the model can classify its own outputs. This self-referential property enables a reinforcement learning approach where the reward signal derives from the model's own consistency, optionally augmented by an external simulator oracle.

The name refers to the ancient symbol of a snake eating its own tail, representing the self-referential nature of the training objective and the cyclic feedback loops characteristic of gene regulatory networks.

### Key Contributions

- A unified generative-discriminative architecture for conditional circuit generation
- Self-classification as a training signal for reinforcement learning
- Integration of boolean network simulation as an oracle reward
- Demonstration that small models (8-10M parameters) suffice for this task
- A complete pipeline from raw regulatory databases to trained generators

---

## Method

### Circuit Representation

Gene regulatory circuits are represented as token sequences:

```
<bos> <phenotype> gene_1 interaction_1 gene_2 interaction_2 gene_3 ... <eos>
```

For example, the Repressilator, a canonical synthetic oscillator:

```
<bos> <oscillator> lacI inhibits tetR inhibits cI inhibits lacI <eos>
```

The vocabulary consists of:
- Special tokens: `<bos>`, `<eos>`, `<pad>`, `<unk>`
- Phenotype class tokens: `<oscillator>`, `<toggle_switch>`, `<adaptation>`, etc.
- Interaction tokens: `activates`, `inhibits`
- Gene tokens: approximately 100 genes from curated databases

### Self-Classification

The same autoregressive model serves as both generator and classifier. For classification, we compute the log-likelihood of a circuit under each possible phenotype label and select the label with highest likelihood:

```
classify(circuit) = argmax_{phenotype} P(phenotype, circuit | model)
```

This requires no architectural changes or additional parameters.

### Training Pipeline

Training proceeds in three phases:

**Phase 1: Supervised Pretraining**

Standard next-token prediction on circuits with known phenotype labels. This establishes basic generation capability.

**Phase 2: Self-Classification GRPO**

For each phenotype label:
1. Generate N circuits conditioned on the label
2. Self-classify each generated circuit
3. Assign reward: 1 if classification matches intended label, 0 otherwise
4. Compute group-relative advantages
5. Update policy with KL penalty from reference model

**Phase 3: Oracle-Augmented GRPO**

Extend Phase 2 with boolean network simulation:
1. Parse generated circuits into network structure
2. Simulate dynamics from multiple initial conditions
3. Classify observed behavior (oscillation, bistability, etc.)
4. Combine oracle reward with self-classification reward

### Supported Phenotypes

| Phenotype | Description |
|-----------|-------------|
| `<oscillator>` | Sustained periodic dynamics |
| `<toggle_switch>` | Bistability with two stable states |
| `<adaptation>` | Return to baseline after perturbation |
| `<pulse_generator>` | Transient response followed by return to baseline |
| `<amplifier>` | Output signal amplitude exceeds input |
| `<stable>` | Single stable steady state |

---

## Installation

### Requirements

- Python 3.10 or higher
- macOS with Apple Silicon (M1/M2/M3/M4) recommended
- 16GB RAM minimum, 24GB recommended

### Setup

```bash
git clone https://github.com/lulzx/ouroboros.git
cd ouroboros

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

The framework is built on MLX for Apple Silicon optimization:

```
mlx>=0.20.0
mlx-lm>=0.20.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## Quick Start

### Run the Full Pipeline

```bash
./scripts/run_pipeline.sh
```

This runs preprocessing, supervised training, GRPO training, and evaluation.

### Individual Steps

```bash
# 1. Preprocess data
python scripts/preprocess.py --expand 10

# 2. Supervised pretraining
python scripts/train_supervised.py --epochs 20 --augment

# 3. GRPO training
python scripts/train_grpo.py --steps 2000

# 4. Oracle GRPO (optional)
python scripts/train_oracle_grpo.py --steps 1000

# 5. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/grpo/checkpoint_step2000.safetensors

# 6. Generate circuits
python scripts/generate.py --checkpoint checkpoints/grpo/checkpoint_step2000.safetensors --phenotype oscillator --simulate
```

### Using the Python API

```python
from src.data.tokenizer import GRNTokenizer
from src.model.transformer import GRNTransformer, ModelArgs
from src.model.generation import generate
from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier
import mlx.core as mx

# Load tokenizer and model
tokenizer = GRNTokenizer.load("data/processed/tokenizer.json")
model = GRNTransformer(ModelArgs(vocab_size=tokenizer.vocab_size))
weights = mx.load("checkpoints/grpo/checkpoint_step2000.safetensors")
model.load_weights(list(weights.items()))

# Generate an oscillator circuit
phenotype_id = tokenizer.token_to_id["<oscillator>"]
prompt = mx.array([[tokenizer.bos_token_id, phenotype_id]])
generated = generate(model, prompt, max_length=64, temperature=1.0)
circuit = tokenizer.decode(generated[0].tolist())
print(circuit)

# Validate with simulation
network = BooleanNetwork.from_circuit(circuit)
classifier = BehaviorClassifier()
predicted_behavior, details = classifier.classify(network)
print(f"Simulated behavior: {predicted_behavior}")
```

---

## Data

### Sources

Ouroboros uses regulatory interaction data from:

- **RegulonDB**: Escherichia coli transcriptional regulatory network
- **TRRUST v2**: Human transcriptional regulatory interactions
- **BioModels**: Curated mathematical models with annotated dynamics

Additionally, a set of approximately 50-100 classic circuits with known behaviors is manually curated from the synthetic biology literature.

### Download and Preprocessing

```bash
python scripts/download_data.py
python scripts/preprocess.py
```

This produces:
- `data/processed/circuits.json`: All circuits in standardized format
- `data/processed/vocabulary.json`: Gene and token vocabularies
- `data/processed/splits/`: Train, validation, and test splits

### Data Format

Raw circuit format:

```json
{
  "id": "circuit_001",
  "phenotype": "oscillator",
  "interactions": [
    {"source": "lacI", "target": "tetR", "type": "inhibits"},
    {"source": "tetR", "target": "cI", "type": "inhibits"},
    {"source": "cI", "target": "lacI", "type": "inhibits"}
  ],
  "source": "biomodels",
  "reference": "PMID:10659856"
}
```

---

## Training

### Phase 1: Supervised Pretraining

```bash
python scripts/train_supervised.py --config configs/supervised.yaml
```

Configuration:
- Batch size: 32
- Learning rate: 3e-4
- Optimizer: AdamW
- Epochs: 20

### Phase 2: Self-Classification GRPO

```bash
python scripts/train_grpo.py --config configs/grpo.yaml
```

Configuration:
- Generations per phenotype: 4
- Learning rate: 1e-5
- KL coefficient: 0.1
- Training steps: 5000

### Phase 3: Oracle-Augmented GRPO

```bash
python scripts/train_grpo.py --config configs/oracle_grpo.yaml
```

Configuration:
- Same as Phase 2
- Oracle weight: 0.5
- Simulation steps: 100

### Computational Requirements

On Apple M4 Pro with 24GB unified memory:
- Phase 1: approximately 2 hours
- Phase 2: approximately 4 hours
- Phase 3: approximately 6 hours

---

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/grpo/best
```

### Metrics

**Self-Consistency**: Fraction of generated circuits where self-classification matches the intended phenotype.

**Oracle Consistency**: Fraction of generated circuits where simulated behavior matches the intended phenotype.

**Diversity**: Mean pairwise edit distance between generated circuits within each phenotype class.

**Novelty**: Fraction of generated circuits not present in the training set.

**Validity**: Fraction of generated circuits that are well-formed (correct grammar, no orphan genes).

---

## Results

### Main Results

| Metric | Supervised | + Self-GRPO | + Oracle-GRPO |
|--------|------------|-------------|---------------|
| Self-Consistency | 0.537 | 0.780* | 0.337 |
| Oracle Consistency | 0.165 | 0.168 | 0.202 |
| Diversity | 13.06 | 27.62* | 6.22 |
| Novelty | 0.978 | 1.000 | 0.988 |
| Validity | 1.000 | 0.985* | 1.000 |

*Self-GRPO achieved high self-consistency through degenerate token patterns (interaction tokens used as genes), not valid circuits.

### Per-Phenotype Self-Consistency (Supervised)

| Phenotype | Accuracy |
|-----------|----------|
| Oscillator | 0.710 |
| Toggle Switch | 0.630 |
| Adaptation | 0.590 |
| Pulse Generator | 0.440 |
| Amplifier | 0.260 |
| Stable | 0.590 |

### Key Findings

1. **Supervised pretraining works well**: The model learns to generate valid, diverse circuits that self-classify correctly 54% of the time.

2. **Self-classification GRPO is vulnerable to reward hacking**: Without validity constraints, the model discovers degenerate patterns (`inhibits⊣inhibits`) that achieve high self-consistency but are semantically meaningless.

3. **Oracle reward is sparse**: The boolean network simulator classifies most circuits as "stable" unless they contain specific feedback structures. This makes RL optimization difficult for non-stable phenotypes.

4. **Mode collapse under RL**: Both GRPO variants eventually collapse to generating limited patterns (primarily oscillator-like or stable-like circuits regardless of the intended phenotype).

### Lessons Learned

- Validity constraints must be part of the reward function, not just evaluation
- Self-classification alone is insufficient; external grounding (oracle) is necessary but challenging
- The simulator's harsh classification creates sparse rewards that impede learning
- Curriculum learning or reward shaping may be needed for complex phenotypes

---

## Citation

```bibtex
@software{ouroboros2025,
  title = {Ouroboros: Self-Classifying Autoregressive Generation of Gene Regulatory Circuits},
  author = {Lulzx},
  year = {2025},
  url = {https://github.com/lulzx/ouroboros}
}
```

---

## License

This project is licensed under the MIT License. See LICENSE for details.

---

## Acknowledgments

This work uses data from RegulonDB, TRRUST, and BioModels. The boolean network simulation approach draws on classical systems biology methods. The GRPO training procedure is adapted from recent work in language model alignment.

---

## References

Gardner, T. S., Cantor, C. R., & Collins, J. J. (2000). Construction of a genetic toggle switch in Escherichia coli. Nature, 403(6767), 339-342.

Elowitz, M. B., & Leibler, S. (2000). A synthetic oscillatory network of transcriptional regulators. Nature, 403(6767), 335-338.

Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Shen-Orr, S. S., Milo, R., Mangan, S., & Alon, U. (2002). Network motifs in the transcriptional regulation network of Escherichia coli. Nature Genetics, 31(1), 64-68.
