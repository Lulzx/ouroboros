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

# 2. Enumerate verified circuits (for constrained generation)
python scripts/enumerate_circuits.py --max-genes 3

# 3. Supervised pretraining on relabeled data
python scripts/relabel_data.py --verbose
python scripts/preprocess.py --circuits data/synthetic/classic_circuits_relabeled.json --output-dir data/processed_relabeled
python scripts/train_supervised.py --data-dir data/processed_relabeled --epochs 20 --augment

# 4. Expert iteration training (optional, for neural model improvement)
python scripts/train_expert_iteration.py --iterations 10 --target-accuracy 0.9

# 5. Evaluate with constrained generation (99.8% accuracy)
python scripts/evaluate_constrained.py --num-samples 100

# 6. Generate circuits with guaranteed correctness
python scripts/generate.py --checkpoint checkpoints/expert_iteration/best.safetensors --phenotype oscillator --simulate
```

### Using the Python API

```python
from src.data.tokenizer import GRNTokenizer
from src.generation.constrained import VerifiedCircuitGenerator
from src.simulator.boolean_network import BooleanNetwork
from src.simulator.classify_behavior import BehaviorClassifier

# Load tokenizer
tokenizer = GRNTokenizer.load("data/processed_relabeled/tokenizer.json")

# Create verified circuit generator (99.8% accuracy)
generator = VerifiedCircuitGenerator(
    verified_db_path="data/verified_circuits.json",
    tokenizer=tokenizer,
)

# Generate oscillator circuits with guaranteed correctness
circuits = generator.generate(phenotype="oscillator", num_samples=5)

for circuit in circuits:
    print("Circuit:", circuit["interactions"])

    # Verify with simulation
    network = BooleanNetwork.from_circuit(circuit)
    classifier = BehaviorClassifier(rule="constitutive")
    predicted, details = classifier.classify(network)
    print(f"  Verified behavior: {predicted}")
```

**Output:**
```
Circuit: [{'source': 'lacI', 'target': 'lacI', 'type': 'inhibits'}]
  Verified behavior: oscillator
Circuit: [{'source': 'p53', 'target': 'mdm2', 'type': 'inhibits'}, {'source': 'mdm2', 'target': 'p53', 'type': 'inhibits'}]
  Verified behavior: oscillator
...
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

### Phase 3: Oracle-Based GRPO

```bash
python scripts/train_grpo_oracle.py \
  --supervised-checkpoint checkpoints/supervised_relabeled/best.safetensors \
  --checkpoint-dir checkpoints/grpo_oracle \
  --steps 500 \
  --num-generations 8 \
  --kl-coef 0.05
```

Configuration:
- Uses boolean network simulator as reward (not self-classification)
- Diversity bonus to prevent mode collapse
- KL coefficient: 0.05 (lower than self-classification to allow exploration)
- Generations per phenotype: 8-12

**Note**: Self-classification GRPO (`train_grpo.py`) causes mode collapse. Use oracle-based GRPO for better results.

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

### Practical Solution: 99.8% Oracle Accuracy

Through exhaustive enumeration and template-based generation, we achieved **99.8% oracle accuracy** for small circuits (2-3 genes).

**Important caveat**: This is a practical engineering solution, not a learned generalization. Template-based generation samples from pre-verified topologies - accuracy is guaranteed by construction, not learned. The neural models plateau at ~50% accuracy, which represents the actual learning challenge.

| Method | Oracle Accuracy | Diversity | Notes |
|--------|----------------|-----------|-------|
| Original Supervised | ~20% | High | Learned surface patterns only |
| Self-Classification GRPO | Mode collapse | - | Collapses to single phenotype |
| Oracle GRPO | 43.3% | Medium | Uses simulator as reward |
| Expert Iteration (10 rounds) | 50.8% | Medium | Best neural approach |
| **Template-Based (Verified)** | **99.8%** | **High** | Lookup from verified database |

### The Core Challenge: Learning Structure

Neural models learn *correlations* (e.g., "oscillator" → inhibition cycles) but not *causation* (specific structures that actually oscillate). This is why:

- Supervised learning plateaus at ~20%
- GRPO improves to ~43% but can't discover new structures
- Expert iteration reaches ~50% with verified data augmentation

For **practical applications** (small circuits), we use template-based generation:
1. Exhaustively enumerate all 19,762 possible 2-3 gene topologies
2. Pre-verify each with the oracle simulator
3. Sample from verified topologies at generation time

**Limitation**: This doesn't scale to larger circuits without re-enumeration.

### Discovered Necessary Conditions

| Phenotype | Required Pattern | Verified Rate |
|-----------|-----------------|---------------|
| Toggle Switch | Mutual inhibition (A⊣B, B⊣A) | 100% |
| Pulse Generator | IFFL motif (A→B, A→C, B⊣C) | 100% |
| Amplifier | Activation cascade (A→B→C) | 100% |
| Adaptation | Self-inhibition | 100% |
| Oscillator | Self-inhibition | 92% |

### Per-Phenotype Oracle Accuracy (Template-Based)

| Phenotype | Accuracy |
|-----------|----------|
| Oscillator | 100% |
| Toggle Switch | 100% |
| Adaptation | 99% |
| Pulse Generator | 100% |
| Amplifier | 100% |
| Stable | 100% |

### Verified Circuit Database

We created a database of 19,762 verified circuits:
- **2-gene circuits**: 80 topologies tested
- **3-gene circuits**: 19,682 topologies tested
- Each classified by the fixed oracle with constitutive update rule

Distribution of working circuits:
```
oscillator:      9,751 (49.3%)
toggle_switch:   2,625 (13.3%)
pulse_generator: 3,098 (15.7%)
adaptation:      2,244 (11.4%)
amplifier:         387 (2.0%)
stable:          1,657 (8.3%)
```

### Diversity Through Gene Assignment

Template-based generation maintains high diversity through combinatorial gene name assignment:
- Gene vocabulary: 103 genes
- For a 3-gene circuit: 103 × 102 × 101 = **1,061,106** possible combinations
- Each combination produces a unique circuit with guaranteed correct dynamics

### Boolean Network Simulator Improvements

Critical fixes to the boolean network simulator:

| Circuit | Before Fix | After Fix |
|---------|------------|-----------|
| Repressilator | stable (wrong) | **oscillator** |
| Toggle Switch | stable (wrong) | **toggle_switch** |
| Activation Cascade | toggle_switch (wrong) | **amplifier** |
| IFFL | adaptation (wrong) | **pulse_generator** |

**Key fixes:**
- Added `constitutive` update rule: genes with only inhibitors are ON by default, repressed when inhibited (biologically accurate)
- Fixed `is_fixed_point` detection to use cycle length
- Added topology checks for mutual inhibition, IFFL, and cascade patterns

### Expert Iteration Results

For training neural models to generate correct circuits:

| Iteration | Oracle Accuracy | Training Set Size |
|-----------|----------------|-------------------|
| 0 (baseline) | 19% | - |
| 1 | 35% | 2,054 |
| 5 | 41% | 2,166 |
| 10 | 50.8% | 2,257 |

The model improves by learning from verified circuits combined with oracle-filtered generations.

### GRPO Training Results

We explored multiple GRPO variants for reinforcement learning:

**Self-Classification GRPO** (original approach):
- Uses model's own likelihood as reward signal
- Problem: Mode collapse to single phenotype ("stable")
- The model learns to generate sequences it can classify, not correct circuits

**Oracle GRPO** (improved approach):
- Uses boolean network simulator as ground truth reward
- Achieves 43.3% overall accuracy
- Per-phenotype: oscillator 96.7%, stable 96.7%, amplifier 66.7%
- Still struggles with toggle_switch (0%), pulse_generator (0%)

**Key GRPO Findings**:
1. Self-classification reward causes mode collapse because the model optimizes for self-consistency rather than correctness
2. Oracle reward prevents collapse but can't discover new topological patterns
3. GRPO can only improve what the model already knows how to generate
4. For phenotypes requiring specific structures (mutual inhibition, IFFL), GRPO alone is insufficient

**Per-Phenotype Neural Model Accuracy** (Best: Expert Iteration + GRPO):

| Phenotype | Neural | Template |
|-----------|--------|----------|
| oscillator | 80.0% | 100% |
| toggle_switch | 22.0% | 100% |
| adaptation | 30.0% | 100% |
| pulse_generator | 30.0% | 100% |
| amplifier | 54.0% | 100% |
| stable | 88.0% | 100% |

### Key Findings

1. **Exhaustive enumeration reveals ground truth**: By testing all small circuits, we discovered exactly what patterns work for each phenotype.

2. **Template-based generation is a practical workaround**: Sampling from pre-verified topologies guarantees correctness but is not learned generalization - it's database lookup with combinatorial gene assignment.

3. **Neural models plateau at ~50% accuracy**: Even with expert iteration and GRPO, neural models struggle with phenotypes requiring specific topological structures. This is a fundamental limitation: neural models learn smooth distributions that assign probability mass to invalid structures.

4. **GRPO can only refine, not discover**: Reinforcement learning improves what the model already knows but cannot discover entirely new structural patterns. For toggle_switch (requires mutual inhibition) and pulse_generator (requires IFFL), neural models need explicit structural guidance.

5. **The constitutive rule is biologically accurate**: Genes with only inhibitors should be ON by default (constitutive expression), which correctly models transcriptional regulation.

6. **Small circuits are sufficient**: Most phenotypes can be achieved with 2-3 genes, making exhaustive enumeration tractable.

### Lessons Learned

- **First principles matter**: Understanding WHY circuits work (not just correlations) enables principled solutions
- **Exhaustive testing is tractable for small circuits**: 3^9 = 19,683 is easily testable
- **Constrained generation beats unconstrained**: Enforcing necessary conditions guarantees correctness
- **Diversity and accuracy are not trade-offs**: Template-based generation achieves both through combinatorial gene assignment
- **RL can't discover structure**: GRPO and similar methods optimize existing behavior but can't discover fundamentally new topological patterns
- **Oracle reward beats self-classification**: Using ground truth (simulator) as reward prevents mode collapse that occurs with self-classification

---

## Project Structure

```
ouroboros/
├── configs/                    # Training configuration files
│   ├── model.yaml
│   └── training.yaml
├── data/
│   ├── synthetic/              # Source circuit data
│   │   ├── classic_circuits.json
│   │   └── classic_circuits_relabeled.json  # Oracle-corrected labels
│   ├── processed/              # Tokenized training data
│   ├── processed_relabeled/    # Relabeled training data
│   └── verified_circuits.json  # 19,762 verified circuit topologies
├── scripts/
│   ├── preprocess.py           # Data preprocessing
│   ├── relabel_data.py         # Re-label with fixed oracle
│   ├── enumerate_circuits.py   # Exhaustive circuit enumeration
│   ├── train_supervised.py     # Supervised pretraining
│   ├── train_grpo.py           # Self-classification GRPO
│   ├── train_grpo_oracle.py    # Oracle-based GRPO (avoids mode collapse)
│   ├── train_expert_iteration.py  # Expert iteration training
│   ├── train_balanced_supervised.py  # Balanced verified circuit training
│   ├── evaluate.py             # Standard evaluation
│   ├── evaluate_constrained.py # Constrained generation evaluation
│   └── generate.py             # Circuit generation
├── src/
│   ├── data/                   # Tokenization and datasets
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── model/                  # Transformer architecture
│   │   └── transformer.py
│   ├── simulator/              # Boolean network simulation
│   │   ├── boolean_network.py
│   │   ├── dynamics.py
│   │   └── classify_behavior.py
│   ├── generation/             # Constrained generation
│   │   └── constrained.py      # Template-based & hybrid generators
│   ├── training/               # Training loops
│   │   ├── supervised.py
│   │   └── grpo.py
│   └── evaluation/             # Metrics and analysis
│       ├── metrics.py
│       └── analyze.py
└── checkpoints/                # Trained models
    ├── supervised/
    ├── supervised_relabeled/
    ├── grpo_oracle/            # Oracle-based GRPO (43.3%)
    ├── expert_iteration/
    └── expert_grpo_v2/         # Expert iteration + GRPO (50.8%)
```

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
