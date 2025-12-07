"""
Graph Generator for GRN Circuits.

This module implements a generative model that produces circuits by learning
the structure of valid topologies conditioned on phenotype.

Key Innovation: Instead of generating tokens sequentially, we generate
the adjacency matrix directly using:

1. NODE GENERATION: Predict number of genes and their roles
2. EDGE GENERATION: Predict which edges exist and their types
3. DYNAMICS VALIDATION: Use the differentiable simulator to validate

The generator is trained with:
- Reconstruction loss (on known circuits)
- Dynamics loss (simulated behavior should match intended phenotype)
- Constraint loss (enforce necessary conditions per phenotype)

Mathematical Foundation:
The space of n-gene circuits is a product of edge decisions:
- For each pair (i,j): edge ∈ {none, activates, inhibits}
- This gives 3^(n²) possible topologies

We model this as:
P(circuit | phenotype) = ∏_{i,j} P(edge_{i,j} | phenotype, context)

The context includes:
- Phenotype embedding
- Already-decided edges
- Dynamics features from partial circuit
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .graph_dynamics import (
    GraphDynamicsArgs,
    DynamicsInformedGNN,
    DifferentiableSimulator,
    SpectralFeatures,
    circuit_to_graph,
)


@dataclass
class GeneratorArgs:
    """Configuration for graph generator."""

    # Graph dimensions
    max_genes: int = 10
    node_dim: int = 64
    edge_dim: int = 32
    hidden_dim: int = 256

    # Generation
    num_edge_types: int = 3  # none, activates, inhibits

    # Phenotype conditioning
    num_phenotypes: int = 6
    phenotype_dim: int = 64

    # Architecture
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    # Training
    soft_temp: float = 5.0


class PhenotypeEmbedding(nn.Module):
    """
    Learnable phenotype embeddings with structural priors.

    Each phenotype has:
    - Base embedding (learned)
    - Structural requirements (encoded as features)
    """

    STRUCTURAL_PRIORS = {
        'oscillator': {
            'needs_cycle': True,
            'needs_self_inhibition': True,
            'needs_mutual_inhibition': False,
        },
        'toggle_switch': {
            'needs_cycle': False,
            'needs_self_inhibition': False,
            'needs_mutual_inhibition': True,
        },
        'adaptation': {
            'needs_cycle': True,
            'needs_self_inhibition': True,
            'needs_mutual_inhibition': False,
        },
        'pulse_generator': {
            'needs_cycle': False,
            'needs_self_inhibition': False,
            'needs_mutual_inhibition': False,
            'needs_iffl': True,
        },
        'amplifier': {
            'needs_cycle': False,
            'needs_self_inhibition': False,
            'needs_mutual_inhibition': False,
            'needs_cascade': True,
        },
        'stable': {
            'needs_cycle': False,
            'needs_self_inhibition': False,
            'needs_mutual_inhibition': False,
        },
    }

    def __init__(self, args: GeneratorArgs):
        super().__init__()
        self.args = args

        # Learnable base embeddings
        self.embed = nn.Embedding(args.num_phenotypes, args.phenotype_dim)

        # Structural prior encoding (8 binary features)
        self.prior_proj = nn.Linear(8, args.phenotype_dim // 4)

    def __call__(self, phenotype_ids: mx.array) -> mx.array:
        """
        Get phenotype embeddings.

        Args:
            phenotype_ids: (batch,) integer phenotype IDs

        Returns:
            embeddings: (batch, phenotype_dim)
        """
        base = self.embed(phenotype_ids)
        return base

    def get_structural_requirements(self, phenotype_name: str) -> Dict[str, bool]:
        """Get structural requirements for a phenotype."""
        return self.STRUCTURAL_PRIORS.get(phenotype_name, {})


class EdgePredictor(nn.Module):
    """
    Predicts edge type for a given node pair.

    Input features:
    - Phenotype embedding
    - Source and target node features
    - Context from already-decided edges
    - Partial dynamics features

    Output: logits for [none, activates, inhibits]
    """

    def __init__(self, args: GeneratorArgs):
        super().__init__()
        self.args = args

        # Input: phenotype + 2 nodes + context
        input_dim = args.phenotype_dim + 2 * args.node_dim + args.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_dim // 2, args.num_edge_types),
        )

    def __call__(
        self,
        phenotype_emb: mx.array,
        source_features: mx.array,
        target_features: mx.array,
        context: mx.array,
    ) -> mx.array:
        """
        Predict edge type.

        Args:
            phenotype_emb: (batch, phenotype_dim)
            source_features: (batch, node_dim)
            target_features: (batch, node_dim)
            context: (batch, hidden_dim) aggregated context

        Returns:
            logits: (batch, num_edge_types)
        """
        combined = mx.concatenate([
            phenotype_emb,
            source_features,
            target_features,
            context,
        ], axis=-1)

        return self.mlp(combined)


class GraphTransformerLayer(nn.Module):
    """
    Transformer layer that operates on graph nodes.

    Uses self-attention to let nodes communicate, with:
    - Phenotype as additional context
    - Partial adjacency as attention mask/bias
    """

    def __init__(self, args: GeneratorArgs):
        super().__init__()
        self.args = args
        self.head_dim = args.node_dim // args.num_heads

        self.q_proj = nn.Linear(args.node_dim, args.node_dim, bias=False)
        self.k_proj = nn.Linear(args.node_dim, args.node_dim, bias=False)
        self.v_proj = nn.Linear(args.node_dim, args.node_dim, bias=False)
        self.o_proj = nn.Linear(args.node_dim, args.node_dim, bias=False)

        self.norm1 = nn.LayerNorm(args.node_dim)
        self.norm2 = nn.LayerNorm(args.node_dim)

        self.mlp = nn.Sequential(
            nn.Linear(args.node_dim, args.node_dim * 4),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.node_dim * 4, args.node_dim),
        )

        # Edge bias projection
        self.edge_bias = nn.Linear(args.num_edge_types, args.num_heads)

    def __call__(
        self,
        x: mx.array,
        edge_types: mx.array,
        node_mask: mx.array,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            x: (batch, n, node_dim) node features
            edge_types: (batch, n, n, num_edge_types) one-hot edge types
            node_mask: (batch, n) binary mask

        Returns:
            x: (batch, n, node_dim) updated features
        """
        B, n, d = x.shape
        h = self.args.num_heads

        # Self-attention with edge bias
        q = self.q_proj(x).reshape(B, n, h, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, n, h, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, n, h, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # Add edge bias
        edge_bias = self.edge_bias(edge_types)  # (B, n, n, num_heads)
        edge_bias = edge_bias.transpose(0, 3, 1, 2)  # (B, h, n, n)
        scores = scores + edge_bias

        # Mask for inactive nodes
        mask = node_mask[:, None, None, :]  # (B, 1, 1, n)
        scores = scores + (1 - mask) * float('-inf')

        weights = mx.softmax(scores, axis=-1)
        attn_out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, n, d)
        attn_out = self.o_proj(attn_out)

        # Residual and norm
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))

        return x


class CircuitGenerator(nn.Module):
    """
    Main generator model for GRN circuits.

    Generation process:
    1. Embed phenotype condition
    2. Initialize node features based on target gene count
    3. Iteratively generate edges using autoregressive decoding
    4. Validate with differentiable simulator

    Training:
    - Teacher forcing on known circuits (reconstruction)
    - REINFORCE/Gumbel for exploration
    - Dynamics loss for correctness

    The key insight is that we generate a GRAPH, not a sequence.
    This naturally handles permutation invariance.
    """

    def __init__(self, args: GeneratorArgs):
        super().__init__()
        self.args = args

        # Phenotype conditioning
        self.phenotype_emb = PhenotypeEmbedding(args)

        # Node initialization
        self.node_init = nn.Sequential(
            nn.Linear(args.phenotype_dim + 1, args.node_dim),  # +1 for position
            nn.ReLU(),
            nn.Linear(args.node_dim, args.node_dim),
        )

        # Graph transformer layers
        self.layers = [GraphTransformerLayer(args) for _ in range(args.num_layers)]

        # Edge predictor
        self.edge_predictor = EdgePredictor(args)

        # Context aggregator
        self.context_agg = nn.Sequential(
            nn.Linear(args.node_dim * args.max_genes, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )

        # Number of genes predictor
        self.num_genes_predictor = nn.Sequential(
            nn.Linear(args.phenotype_dim, args.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_dim // 2, args.max_genes - 1),  # 2 to max_genes
        )

        # Differentiable simulator for validation
        sim_args = GraphDynamicsArgs(
            max_genes=args.max_genes,
            hidden_dim=args.hidden_dim,
            num_sim_steps=20,
        )
        self.simulator = DifferentiableSimulator(sim_args)

    def init_nodes(
        self,
        phenotype_emb: mx.array,
        num_genes: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Initialize node features.

        Args:
            phenotype_emb: (batch, phenotype_dim)
            num_genes: (batch,) number of genes per sample

        Returns:
            node_features: (batch, max_genes, node_dim)
            node_mask: (batch, max_genes)
        """
        B = phenotype_emb.shape[0]
        n = self.args.max_genes

        # Create position features
        positions = mx.arange(n)[None, :] / n  # (1, n) normalized position

        # Broadcast phenotype to all nodes
        phenotype_expanded = phenotype_emb[:, None, :].broadcast_to((B, n, -1))

        # Combine phenotype with position
        pos_features = positions.broadcast_to((B, n))[:, :, None]
        combined = mx.concatenate([phenotype_expanded, pos_features], axis=-1)

        # Initialize node features
        node_features = self.node_init(combined)

        # Create mask based on num_genes
        indices = mx.arange(n)[None, :]  # (1, n)
        node_mask = (indices < num_genes[:, None]).astype(mx.float32)

        return node_features, node_mask

    def generate_edges(
        self,
        phenotype_emb: mx.array,
        node_features: mx.array,
        node_mask: mx.array,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """
        Generate edges autoregressively.

        Args:
            phenotype_emb: (batch, phenotype_dim)
            node_features: (batch, n, node_dim)
            node_mask: (batch, n)
            temperature: sampling temperature
            hard: if True, use hard sampling (for inference)

        Returns:
            edge_types: (batch, n, n, num_edge_types) soft one-hot
            adj_matrix: (batch, n, n) signed adjacency
        """
        B, n, d = node_features.shape
        num_edge_types = self.args.num_edge_types

        # Initialize edge types as "none" (first type)
        edge_types = mx.zeros((B, n, n, num_edge_types))
        edge_types = edge_types.at[:, :, :, 0].add(1)  # Start with all "none"

        # Process through transformer layers
        x = node_features
        for layer in self.layers:
            x = layer(x, edge_types, node_mask)

        # Generate edges in order (upper triangle first, then lower)
        adj_matrix = mx.zeros((B, n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # No self-loops initially (handled separately)

                # Context from current graph state
                context = self.context_agg(x.reshape(B, -1))

                # Predict edge type
                logits = self.edge_predictor(
                    phenotype_emb,
                    x[:, i, :],  # source
                    x[:, j, :],  # target
                    context,
                )

                # Apply temperature and sample
                logits = logits / temperature

                if hard:
                    # Hard sampling for inference
                    edge_idx = mx.argmax(logits, axis=-1)
                    one_hot = mx.zeros((B, num_edge_types))
                    for b in range(B):
                        one_hot = one_hot.at[b, edge_idx[b]].set(1)
                else:
                    # Gumbel-softmax for training
                    one_hot = self._gumbel_softmax(logits, temperature)

                # Update edge types
                edge_types = edge_types.at[:, i, j, :].set(one_hot)

                # Update adjacency (0: none, 1: activates, -1: inhibits)
                # edge_idx 0 = none, 1 = activates, 2 = inhibits
                adj_val = one_hot[:, 1] - one_hot[:, 2]  # +1 or -1
                adj_matrix = adj_matrix.at[:, i, j].set(adj_val)

                # Mask by node existence
                edge_mask = node_mask[:, i] * node_mask[:, j]
                adj_matrix = adj_matrix.at[:, i, j].multiply(edge_mask)

        return edge_types, adj_matrix

    def _gumbel_softmax(
        self,
        logits: mx.array,
        temperature: float = 1.0,
    ) -> mx.array:
        """Gumbel-softmax for differentiable sampling."""
        # Sample from Gumbel(0, 1)
        u = mx.random.uniform(shape=logits.shape)
        gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)

        # Add Gumbel noise and apply softmax
        y = mx.softmax((logits + gumbel) / temperature, axis=-1)

        return y

    def forward(
        self,
        phenotype_ids: mx.array,
        num_genes: Optional[mx.array] = None,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, mx.array]:
        """
        Generate circuits conditioned on phenotype.

        Args:
            phenotype_ids: (batch,) phenotype IDs
            num_genes: (batch,) optional number of genes (sampled if None)
            temperature: sampling temperature
            hard: hard sampling for inference

        Returns:
            Dictionary with:
            - adj_matrix: (batch, n, n) signed adjacency
            - edge_types: (batch, n, n, num_edge_types) one-hot
            - node_mask: (batch, n)
            - num_genes: (batch,)
        """
        B = phenotype_ids.shape[0]

        # Get phenotype embedding
        phenotype_emb = self.phenotype_emb(phenotype_ids)

        # Predict number of genes if not provided
        if num_genes is None:
            num_genes_logits = self.num_genes_predictor(phenotype_emb)
            if hard:
                num_genes = mx.argmax(num_genes_logits, axis=-1) + 2  # 2 to max_genes
            else:
                num_genes_probs = mx.softmax(num_genes_logits, axis=-1)
                num_genes = mx.argmax(num_genes_probs, axis=-1) + 2
        num_genes = num_genes.astype(mx.int32)

        # Initialize nodes
        node_features, node_mask = self.init_nodes(phenotype_emb, num_genes)

        # Generate edges
        edge_types, adj_matrix = self.generate_edges(
            phenotype_emb,
            node_features,
            node_mask,
            temperature,
            hard,
        )

        return {
            'adj_matrix': adj_matrix,
            'edge_types': edge_types,
            'node_mask': node_mask,
            'num_genes': num_genes,
            'phenotype_emb': phenotype_emb,
        }

    def compute_loss(
        self,
        phenotype_ids: mx.array,
        target_adj: mx.array,
        target_mask: mx.array,
        target_dynamics: Optional[Dict[str, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Compute training loss.

        Components:
        1. Edge reconstruction: cross-entropy on edge types
        2. Dynamics loss: simulated behavior should match phenotype
        3. Constraint loss: enforce structural priors

        Args:
            phenotype_ids: (batch,) phenotype IDs
            target_adj: (batch, n, n) ground truth signed adjacency
            target_mask: (batch, n) node mask
            target_dynamics: optional dict with dynamics labels

        Returns:
            total_loss: scalar
            loss_components: dict
        """
        B = phenotype_ids.shape[0]
        n = self.args.max_genes

        # Get phenotype embedding
        phenotype_emb = self.phenotype_emb(phenotype_ids)

        # Count target genes
        num_genes = mx.sum(target_mask, axis=1).astype(mx.int32)

        # Initialize and process
        node_features, node_mask = self.init_nodes(phenotype_emb, num_genes)

        # Convert target adjacency to one-hot edge types
        target_edge_types = mx.zeros((B, n, n, 3))
        no_edge = (mx.abs(target_adj) < 0.5)
        activates = target_adj > 0.5
        inhibits = target_adj < -0.5

        target_edge_types = target_edge_types.at[:, :, :, 0].add(no_edge.astype(mx.float32))
        target_edge_types = target_edge_types.at[:, :, :, 1].add(activates.astype(mx.float32))
        target_edge_types = target_edge_types.at[:, :, :, 2].add(inhibits.astype(mx.float32))

        # Process through transformer
        x = node_features
        for layer in self.layers:
            x = layer(x, target_edge_types, node_mask)

        # Compute edge prediction loss (teacher forcing)
        edge_loss = mx.array(0.0)
        edge_count = 0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                context = self.context_agg(x.reshape(B, -1))
                logits = self.edge_predictor(
                    phenotype_emb,
                    x[:, i, :],
                    x[:, j, :],
                    context,
                )

                # Target edge type
                target = mx.argmax(target_edge_types[:, i, j, :], axis=-1)

                # Cross-entropy loss
                ce = nn.losses.cross_entropy(logits, target)

                # Mask by node existence
                mask = node_mask[:, i] * node_mask[:, j]
                edge_loss = edge_loss + mx.sum(ce * mask)
                edge_count += mx.sum(mask)

        edge_loss = edge_loss / (edge_count + 1e-8)

        # Dynamics loss: simulate and check consistency
        trajectory, dynamics_features = self.simulator.simulate(target_adj, node_mask)
        attractor_features = self.simulator.compute_attractor_features(trajectory, node_mask)

        dynamics_loss = self._compute_dynamics_loss(
            attractor_features,
            phenotype_ids,
        )

        # Constraint loss: enforce structural priors
        constraint_loss = self._compute_constraint_loss(
            target_adj,
            node_mask,
            phenotype_ids,
        )

        # Total loss
        total_loss = edge_loss + 0.5 * dynamics_loss + 0.3 * constraint_loss

        return total_loss, {
            'edge': edge_loss,
            'dynamics': dynamics_loss,
            'constraint': constraint_loss,
        }

    def _compute_dynamics_loss(
        self,
        attractor_features: Dict[str, mx.array],
        phenotype_ids: mx.array,
    ) -> mx.array:
        """Encourage dynamics features to match phenotype."""
        B = phenotype_ids.shape[0]

        fixed_point = attractor_features['fixed_point']
        oscillation = attractor_features['oscillation']

        # Phenotype-specific losses
        # 0: oscillator - should oscillate
        # 1: toggle_switch - should have fixed point
        # 5: stable - should have fixed point

        osc_mask = (phenotype_ids == 0).astype(mx.float32)
        toggle_mask = (phenotype_ids == 1).astype(mx.float32)
        stable_mask = (phenotype_ids == 5).astype(mx.float32)

        loss = (
            osc_mask * (1 - oscillation) +
            toggle_mask * (1 - fixed_point) +
            stable_mask * (1 - fixed_point)
        )

        return mx.mean(loss)

    def _compute_constraint_loss(
        self,
        adj_matrix: mx.array,
        node_mask: mx.array,
        phenotype_ids: mx.array,
    ) -> mx.array:
        """Enforce structural constraints per phenotype."""
        B = adj_matrix.shape[0]
        n = self.args.max_genes

        # Detect structural features
        # Mutual inhibition
        inhib = (adj_matrix < -0.5).astype(mx.float32)
        mutual_inhib = inhib * mx.transpose(inhib, (0, 2, 1))
        has_mutual = mx.max(mx.max(mutual_inhib, axis=1), axis=1)

        # Self inhibition
        diag = mx.diagonal(adj_matrix, axis1=1, axis2=2)
        has_self_inhib = mx.max((diag < -0.5).astype(mx.float32), axis=1)

        # Cascade (activation chain)
        activ = (adj_matrix > 0.5).astype(mx.float32)
        two_hop = mx.matmul(activ, activ)
        has_cascade = (mx.max(mx.max(two_hop, axis=1), axis=1) > 0.5).astype(mx.float32)

        # Phenotype constraints
        # Toggle switch MUST have mutual inhibition
        toggle_mask = (phenotype_ids == 1).astype(mx.float32)
        toggle_violation = toggle_mask * (1 - has_mutual)

        # Amplifier SHOULD have cascade
        amp_mask = (phenotype_ids == 4).astype(mx.float32)
        amp_violation = amp_mask * (1 - has_cascade)

        # Oscillator SHOULD have self-inhibition or negative feedback
        osc_mask = (phenotype_ids == 0).astype(mx.float32)
        osc_violation = osc_mask * (1 - has_self_inhib) * 0.5  # softer constraint

        loss = toggle_violation + amp_violation + osc_violation

        return mx.mean(loss)

    @property
    def num_parameters(self) -> int:
        """Count parameters."""
        def count_params(params):
            total = 0
            if isinstance(params, dict):
                for v in params.values():
                    total += count_params(v)
            elif isinstance(params, list):
                for v in params:
                    total += count_params(v)
            elif hasattr(params, 'size'):
                total += params.size
            return total
        return count_params(self.parameters())


def create_generator(
    max_genes: int = 10,
    hidden_dim: int = 256,
    num_layers: int = 6,
) -> CircuitGenerator:
    """Create a circuit generator with specified configuration."""
    args = GeneratorArgs(
        max_genes=max_genes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    return CircuitGenerator(args)
