# Theoretical Foundations for Dynamics-Informed Learning

## The Core Problem

**Why do neural models plateau at ~50% accuracy on GRN phenotype prediction?**

### Mathematical Formulation

A gene regulatory network (GRN) with *n* genes can be modeled as a Boolean network:

- **State space**: S = {0,1}ⁿ (2ⁿ possible states)
- **Transition function**: f: S → S
- **Dynamics**: x_{t+1} = f(x_t)

The phenotype of a circuit is determined by its **attractor structure**:

| Phenotype | Attractor Property |
|-----------|-------------------|
| Oscillator | ∃ limit cycle with period ≥ 2 |
| Toggle Switch | ∃ at least 2 distinct fixed points (bistability) |
| Stable | Unique global fixed point |
| Adaptation | Returns to baseline after perturbation (homeostasis) |
| Pulse Generator | Transient response followed by return |
| Amplifier | Output amplitude exceeds input |

### Why Previous Approaches Fail

#### 1. Sequence Representation Mismatch

The previous approach represented circuits as sequences:
```
<bos> <oscillator> geneA inhibits geneB activates geneC <eos>
```

**Problem**: Circuits are **graphs**, not sequences. The same circuit can be written in multiple equivalent orderings:
- "A inhibits B, B inhibits C, C inhibits A"
- "B inhibits C, C inhibits A, A inhibits B"
- "C inhibits A, A inhibits B, B inhibits C"

These are identical graphs but different sequences. The model must learn this equivalence implicitly, wasting capacity.

#### 2. Token-Level Statistics vs. Graph-Level Invariants

Standard transformers learn **token-level statistics**:
- P(next_token | previous_tokens)

But the phenotype depends on **graph-level invariants**:
- Presence of cycles
- Mutual inhibition structures
- Feedback loop polarity (positive vs negative)

**Concrete Example**:

For toggle switches, the *necessary condition* is mutual inhibition (A inhibits B AND B inhibits A). This is a **structural invariant** that requires reasoning about pairs of edges globally, not predicting the next token locally.

#### 3. Smooth Distributions on Discrete Structures

Neural networks learn smooth probability distributions. But the space of valid circuits is **discrete with sharp boundaries**:

- A circuit either HAS mutual inhibition or it DOESN'T
- There's no "almost mutual inhibition"

When the model assigns probability to invalid structures (e.g., circuits without mutual inhibition but labeled "toggle_switch"), it learns spurious correlations.

#### 4. Learning Correlations, Not Causation

From training data, the model learns correlations like:
- "oscillator" → more "inhibits" tokens
- "toggle_switch" → gene pairs appear together

But it doesn't learn causation:
- **WHY** does a repressilator oscillate?
- **WHAT** makes a toggle switch bistable?

### The Solution: Dynamics-Informed Learning

## Theoretical Framework

### 1. Graph Representation with Permutation Equivariance

**Definition**: A function f is **permutation equivariant** if for any permutation π:
```
f(π(G)) = π(f(G))
```

**Key Property**: The phenotype of a circuit is invariant to gene relabeling. Therefore, our model should be permutation equivariant.

**Implementation**: Message Passing Neural Networks (MPNNs) are naturally permutation equivariant. For node i with neighbors N(i):

```
h_i^{(l+1)} = UPDATE(h_i^{(l)}, AGGREGATE({h_j^{(l)} : j ∈ N(i)}))
```

### 2. Differentiable Dynamics

**Key Innovation**: Make the Boolean network simulation differentiable.

**Standard Boolean Update** (non-differentiable):
```python
if activation_count > inhibition_count:
    next_state = 1
else:
    next_state = 0
```

**Soft Boolean Update** (differentiable):
```python
next_state = sigmoid(τ * (activation_signal - inhibition_signal))
```

As τ → ∞, this approaches the hard Boolean function. For finite τ, gradients can flow.

**Mathematical Insight**: The composition of T differentiable update steps:

```
x_{t+T} = f(f(...f(x_0)...)) = f^T(x_0)
```

is a differentiable function of the network parameters (adjacency matrix). This allows us to compute:

```
∂L/∂W = ∂L/∂x_T * ∂x_T/∂W
```

where W is the signed adjacency matrix.

### 3. Spectral Characterization of Structure

The **Laplacian spectrum** of a graph encodes structural properties:

- **L = D - A** where D is degree matrix, A is adjacency
- Eigenvalues λ₁ ≤ λ₂ ≤ ... ≤ λₙ

**Key Properties**:
- Number of zero eigenvalues = number of connected components
- Smallest non-zero eigenvalue (algebraic connectivity) = how "connected" the graph is
- Largest eigenvalue = related to maximum degree

For directed graphs (like GRNs), we use the **signed interaction matrix** W = A⁺ - A⁻ where A⁺ is activation and A⁻ is inhibition.

**Eigenvalue Interpretation**:
- Complex eigenvalues → oscillatory dynamics
- Positive real eigenvalues → stable dynamics
- Negative real eigenvalues → damped dynamics

### 4. Contrastive Learning for Phenotype Separation

**Objective**: Learn representations where:
- Same phenotype → similar representations
- Different phenotypes → dissimilar representations

**Supervised Contrastive Loss**:

```
L_contrast = -log(exp(z_i · z_j / τ) / Σ_{k≠i} exp(z_i · z_k / τ))
```

where z_j has the same phenotype as z_i.

**Why This Helps**:
- Forces the model to find discriminative features
- Encourages clustering by phenotype
- Regularizes representations to be informative

### 5. Structural Priors and Hard Constraints

Some phenotypes have **necessary conditions** that are analytically known:

| Phenotype | Necessary Condition |
|-----------|-------------------|
| Toggle Switch | Mutual inhibition (A⊣B, B⊣A) |
| Oscillator | Negative feedback loop (odd number of inhibitions in a cycle) |
| Amplifier | Activation cascade without feedback |
| Pulse Generator | IFFL motif (A→B, A→C, B⊣C) |

**Implementation**: We encode these as soft constraints in the loss:

```
L_constraint = λ * Σ_i [violation_i(G, phenotype_i)]
```

For toggle switch:
```
violation = (1 - has_mutual_inhibition(G)) * I[phenotype = toggle_switch]
```

## Curriculum Learning Strategy

### Stage 1: Simple Deterministic Cases

**Focus**: 2-gene circuits where topology → phenotype is deterministic

**Examples**:
- A⊣B, B⊣A → always toggle_switch
- A⊣A → always oscillator
- A→B → always stable

**Objective**: Learn the fundamental mappings without ambiguity

### Stage 2: Add Complexity

**Focus**: 3-gene circuits with more structural variety

**Key Patterns**:
- Repressilator (A⊣B⊣C⊣A) → oscillator
- IFFL (A→B, A→C, B⊣C) → pulse_generator
- Cascades (A→B→C) → amplifier

### Stage 3: Contrastive Learning

**Focus**: Learn to separate phenotype classes in representation space

**Training**: Add contrastive loss to classification loss

### Stage 4: Dynamics Fine-tuning

**Focus**: Align learned representations with actual dynamics

**Training**: Use differentiable simulation loss to ensure predicted phenotype matches simulated behavior

### Stage 5: Exploration

**Focus**: Discover novel structures through guided exploration

**Training**: RL-style exploration with oracle reward

## Expected Improvements

### Why 90%+ Accuracy Should Be Achievable

1. **Graph representation eliminates permutation ambiguity**: No more learning that "A inhibits B" and "B inhibits A" have the same meaning when in the same circuit.

2. **Differentiable simulation provides direct supervision**: The model learns from actual dynamics, not just correlated labels.

3. **Structural priors prevent impossible generations**: The model can't generate a toggle switch without mutual inhibition.

4. **Contrastive learning improves separation**: Phenotype classes become clearly distinguishable.

5. **Curriculum learning builds solid foundations**: Simple patterns are mastered before complex ones.

### Generalization to Larger Circuits

**Key Insight**: The structural features that determine phenotype are **local motifs**, not global size.

- Mutual inhibition is the same whether in a 3-gene or 10-gene circuit
- Negative feedback loops follow the same rules regardless of loop length
- Cascades have the same dynamics at any depth

By learning these local motifs, the model should generalize to circuits of any size.

## Comparison to Previous Approaches

| Aspect | Previous (Sequence) | New (Graph + Dynamics) |
|--------|-------------------|----------------------|
| Representation | Sequential tokens | Graph adjacency |
| Learning objective | Next-token prediction | Phenotype classification + dynamics |
| Structural priors | Implicit in data | Explicit constraints |
| Dynamics awareness | None | Differentiable simulation |
| Permutation handling | Learned implicitly | Built-in equivariance |
| Generalization | Limited | By design |

## Mathematical Guarantees

### Theorem 1: Permutation Equivariance

For any permutation matrix P and circuit graph G:
```
f_GNN(P^T G P) = P f_GNN(G)
```

The phenotype prediction is therefore invariant to gene relabeling.

### Theorem 2: Dynamics Consistency

If the differentiable simulation converges to the same attractor as the discrete simulation (in the limit of temperature τ → ∞), then optimizing the dynamics loss is equivalent to optimizing oracle accuracy.

### Theorem 3: Structural Constraint Satisfaction

When the constraint loss L_constraint = 0, all generated circuits satisfy the necessary conditions for their intended phenotype.

## Implementation Considerations

### MLX Optimization

- Use `mx.vmap` for batched operations over circuits
- Leverage `mx.graph` for efficient adjacency operations
- Compile frequently-called functions with `mx.compile`

### Memory Efficiency

- Adjacency matrices are sparse; use sparse representations when possible
- Limit maximum genes to 10-20 for practical training
- Use gradient checkpointing for deep message passing

### Training Stability

- Use LayerNorm between message passing layers
- Gradient clipping (max norm 1.0)
- Learning rate warmup followed by cosine decay

## Conclusion

The key insight is that **phenotype classification is fundamentally a dynamics problem**, not a pattern matching problem. By building dynamics awareness directly into the model architecture (through differentiable simulation and structural priors), we can achieve the kind of robust generalization that was impossible with sequence-based approaches.

The combination of:
1. Graph representation (correct structure)
2. Differentiable simulation (correct objective)
3. Structural priors (domain knowledge)
4. Contrastive learning (representation quality)
5. Curriculum learning (training stability)

should enable 90%+ accuracy through genuine learning, with generalization to larger circuits.
