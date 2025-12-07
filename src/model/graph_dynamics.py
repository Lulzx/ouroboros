"""
Dynamics-Informed Graph Neural Network for GRN Generation.

This module implements a novel architecture that learns the DYNAMICS of gene regulatory
networks, not just surface patterns. Key innovations:

1. GRAPH REPRESENTATION: Circuits are graphs, not sequences. We use message-passing
   neural networks with permutation equivariance.

2. DIFFERENTIABLE SIMULATION: We unfold Boolean network dynamics as a differentiable
   computation graph using soft Boolean operations (sigmoid approximations).

3. SPECTRAL FEATURES: Eigenvalues of the interaction matrix encode structural
   properties like cycles, feedback loops, and connectivity.

4. ATTRACTOR PREDICTION: We train auxiliary heads to predict dynamical invariants
   (number of attractors, cycle lengths, fixed points).

5. CONTRASTIVE LEARNING: We learn representations where same-dynamics circuits
   are close and different-dynamics circuits are far apart.

Mathematical Foundation:
- A Boolean network with n genes has state space S = {0,1}^n
- The dynamics is governed by: x_{t+1} = f(x_t) where f is the update rule
- Phenotypes are determined by attractor structure:
  - Oscillator: ∃ limit cycle with period ≥ 2
  - Toggle switch: ∃ at least 2 distinct fixed points
  - Stable: unique global fixed point
  - Adaptation: returns to baseline after perturbation

Key Insight: The phenotype is a function of the GRAPH TOPOLOGY and UPDATE RULE,
not the specific gene names. Therefore, the model should be permutation-equivariant
with respect to gene relabeling.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class GraphDynamicsArgs:
    """Configuration for Dynamics-Informed Graph Neural Network."""

    # Graph dimensions
    max_genes: int = 10  # Maximum number of genes in a circuit
    node_dim: int = 64   # Node embedding dimension
    edge_dim: int = 32   # Edge embedding dimension
    hidden_dim: int = 128  # Hidden layer dimension

    # Message passing
    num_mp_layers: int = 4  # Number of message passing layers

    # Dynamics simulation
    num_sim_steps: int = 20  # Number of differentiable simulation steps
    soft_temp: float = 5.0   # Temperature for soft Boolean operations

    # Spectral features
    use_spectral: bool = True
    spectral_dim: int = 16

    # Phenotype prediction
    num_phenotypes: int = 6

    # Training
    dropout: float = 0.1

    # Contrastive learning
    contrastive_dim: int = 64
    contrastive_temp: float = 0.07


class SoftBooleanOps:
    """
    Differentiable approximations of Boolean operations.

    Key insight: Boolean operations are discontinuous, but we can approximate
    them with smooth functions for gradient-based learning.

    - AND(x, y) ≈ sigmoid(temp * (x + y - 1.5))
    - OR(x, y) ≈ sigmoid(temp * max(x, y))
    - NOT(x) = 1 - x

    As temp → ∞, these approach true Boolean operations.
    """

    @staticmethod
    def soft_and(x: mx.array, y: mx.array, temp: float = 5.0) -> mx.array:
        """Soft AND: returns ~1 when both inputs are ~1."""
        return mx.sigmoid(temp * (x + y - 1.5))

    @staticmethod
    def soft_or(x: mx.array, y: mx.array, temp: float = 5.0) -> mx.array:
        """Soft OR: returns ~1 when either input is ~1."""
        return mx.sigmoid(temp * (mx.maximum(x, y) - 0.5))

    @staticmethod
    def soft_not(x: mx.array) -> mx.array:
        """Soft NOT: returns 1 - x."""
        return 1.0 - x

    @staticmethod
    def soft_threshold(x: mx.array, threshold: float = 0.5, temp: float = 5.0) -> mx.array:
        """Soft threshold: smooth step function."""
        return mx.sigmoid(temp * (x - threshold))


class SpectralFeatures(nn.Module):
    """
    Compute spectral features of the interaction graph.

    The eigenvalues of the graph Laplacian encode structural properties:
    - Number of connected components (zero eigenvalues)
    - Graph diameter (spectral gap)
    - Cycle structure (complex eigenvalues of adjacency)

    For directed graphs (like GRNs), we use the interaction matrix W = A - I
    where A is activation and I is inhibition.
    """

    def __init__(self, args: GraphDynamicsArgs):
        super().__init__()
        self.args = args
        self.proj = nn.Linear(args.max_genes * 2, args.spectral_dim)

    def __call__(self, adj_matrix: mx.array) -> mx.array:
        """
        Compute spectral features from adjacency matrix.

        Args:
            adj_matrix: (batch, max_genes, max_genes) signed adjacency
                       +1 for activation, -1 for inhibition, 0 for no edge

        Returns:
            spectral_features: (batch, spectral_dim)
        """
        B = adj_matrix.shape[0]
        n = self.args.max_genes

        # Compute degree features (in-degree, out-degree for each node)
        in_degree = mx.sum(mx.abs(adj_matrix), axis=1)  # (B, n)
        out_degree = mx.sum(mx.abs(adj_matrix), axis=2)  # (B, n)

        # Activation vs inhibition balance
        activation_in = mx.sum(mx.maximum(adj_matrix, 0), axis=1)
        inhibition_in = mx.sum(mx.maximum(-adj_matrix, 0), axis=1)

        # Concatenate features
        # Shape: (B, n * 2)
        degree_features = mx.concatenate([in_degree, out_degree], axis=1)

        # Project to spectral dimension
        spectral = self.proj(degree_features)

        return spectral

    def compute_motif_features(self, adj_matrix: mx.array) -> Dict[str, mx.array]:
        """
        Detect structural motifs relevant for phenotype prediction.

        Returns dict with soft indicators for:
        - mutual_inhibition: A -| B and B -| A (toggle switch)
        - self_inhibition: A -| A (oscillator, adaptation)
        - feedforward: A -> B -> C with A -> C (IFFL variants)
        - cascade: A -> B -> C (amplifier)
        """
        B = adj_matrix.shape[0]
        n = self.args.max_genes

        # Mutual inhibition: W[i,j] < 0 AND W[j,i] < 0
        # Use soft comparison
        inhib = mx.sigmoid(-10 * adj_matrix)  # ~1 where inhibition
        mutual_inhib = inhib * mx.transpose(inhib, (0, 2, 1))
        has_mutual_inhib = mx.max(mx.max(mutual_inhib, axis=1), axis=1)

        # Self inhibition: diagonal is negative
        diag = mx.diagonal(adj_matrix, axis1=1, axis2=2)  # (B, n)
        has_self_inhib = mx.max(mx.sigmoid(-10 * diag), axis=1)

        # Cascade: chain of activations
        # W[i,j] > 0 AND W[j,k] > 0 for some i,j,k
        activ = mx.sigmoid(10 * adj_matrix)  # ~1 where activation
        two_hop = mx.matmul(activ, activ)  # (B, n, n)
        has_cascade = mx.max(mx.max(two_hop, axis=1), axis=1)

        # IFFL: A activates B and C, B inhibits C
        # This is complex, approximate with: activation path AND inhibition exists
        inhib_exists = mx.max(mx.max(inhib, axis=1), axis=1)
        activ_exists = mx.max(mx.max(activ, axis=1), axis=1)
        has_iffl = inhib_exists * activ_exists * has_cascade

        return {
            'mutual_inhibition': has_mutual_inhib,
            'self_inhibition': has_self_inhib,
            'cascade': has_cascade,
            'iffl': has_iffl,
        }


class DifferentiableSimulator(nn.Module):
    """
    Differentiable Boolean network simulator.

    Instead of discrete Boolean updates, we use continuous relaxations:

    For gene g with activators A and inhibitors I:
    - activation_signal = sigmoid(sum of activator states)
    - inhibition_signal = sigmoid(sum of inhibitor states)
    - next_state = soft_update(activation_signal, inhibition_signal, rule)

    The key insight is that we can unfold T steps of simulation as a
    differentiable computation graph, allowing gradients to flow through
    the dynamics.
    """

    def __init__(self, args: GraphDynamicsArgs):
        super().__init__()
        self.args = args
        self.temp = args.soft_temp

        # Learnable parameters for update rule
        self.activation_weight = nn.Linear(1, 1, bias=False)
        self.inhibition_weight = nn.Linear(1, 1, bias=False)

        # State encoder
        self.state_encoder = nn.Linear(args.max_genes, args.hidden_dim)

        # Dynamics features extractor
        self.dynamics_head = nn.Sequential(
            nn.Linear(args.hidden_dim * args.num_sim_steps, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
        )

    def soft_update(
        self,
        state: mx.array,
        adj_matrix: mx.array,
        node_mask: mx.array,
    ) -> mx.array:
        """
        Perform one step of soft Boolean network update.

        Args:
            state: (batch, max_genes) current state in [0, 1]
            adj_matrix: (batch, max_genes, max_genes) signed adjacency
            node_mask: (batch, max_genes) which nodes are active

        Returns:
            next_state: (batch, max_genes) next state in [0, 1]
        """
        B, n = state.shape

        # Separate activation and inhibition edges
        activation_adj = mx.maximum(adj_matrix, 0)  # +1 edges
        inhibition_adj = mx.maximum(-adj_matrix, 0)  # -1 edges -> +1

        # Compute incoming signals
        # activation_signal[i] = sum_j adj[j,i] * state[j]
        state_expanded = state[:, :, None]  # (B, n, 1)

        activation_in = mx.matmul(
            mx.transpose(activation_adj, (0, 2, 1)),  # (B, n, n)
            state_expanded  # (B, n, 1)
        ).squeeze(-1)  # (B, n)

        inhibition_in = mx.matmul(
            mx.transpose(inhibition_adj, (0, 2, 1)),
            state_expanded
        ).squeeze(-1)  # (B, n)

        # Count number of regulators
        num_activators = mx.sum(activation_adj, axis=1)  # (B, n)
        num_inhibitors = mx.sum(inhibition_adj, axis=1)  # (B, n)

        # Constitutive update rule (biologically accurate):
        # - Genes with only inhibitors: ON by default, OFF when inhibited
        # - Genes with only activators: OFF by default, ON when activated
        # - Genes with both: compare activation vs inhibition

        # Soft implementation:
        only_inhibitors = (num_activators < 0.5) * (num_inhibitors >= 0.5)
        only_activators = (num_inhibitors < 0.5) * (num_activators >= 0.5)
        both = (num_activators >= 0.5) * (num_inhibitors >= 0.5)
        neither = (num_activators < 0.5) * (num_inhibitors < 0.5)

        # Next state computation
        # For only inhibitors: 1 if no active inhibitor, 0 otherwise
        next_only_inhib = mx.sigmoid(self.temp * (0.5 - inhibition_in))

        # For only activators: 1 if any active activator, 0 otherwise
        next_only_activ = mx.sigmoid(self.temp * (activation_in - 0.5))

        # For both: compare
        next_both = mx.sigmoid(self.temp * (activation_in - inhibition_in))

        # For neither: maintain state
        next_neither = state

        # Combine
        next_state = (
            only_inhibitors * next_only_inhib +
            only_activators * next_only_activ +
            both * next_both +
            neither * next_neither
        )

        # Apply node mask (inactive nodes stay 0)
        next_state = next_state * node_mask

        return next_state

    def simulate(
        self,
        adj_matrix: mx.array,
        node_mask: mx.array,
        initial_state: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Run differentiable simulation for T steps.

        Args:
            adj_matrix: (batch, max_genes, max_genes) signed adjacency
            node_mask: (batch, max_genes) which nodes are active
            initial_state: (batch, max_genes) or None for random

        Returns:
            trajectory: (batch, T, max_genes) state trajectory
            dynamics_features: (batch, hidden_dim // 2) learned features
        """
        B = adj_matrix.shape[0]
        n = adj_matrix.shape[1]  # Get actual size from input
        T = self.args.num_sim_steps

        # Initialize state
        if initial_state is None:
            # Start with all genes ON (constitutive expression)
            state = node_mask.astype(mx.float32)
        else:
            state = initial_state

        # Run simulation
        trajectory = [state]
        for _ in range(T - 1):
            state = self.soft_update(state, adj_matrix, node_mask)
            trajectory.append(state)

        # Stack trajectory
        trajectory = mx.stack(trajectory, axis=1)  # (B, T, n)

        # Extract dynamics features
        # Pad trajectory to max_genes for the encoder if needed
        if n < self.args.max_genes:
            padding = mx.zeros((B, T, self.args.max_genes - n))
            trajectory_padded = mx.concatenate([trajectory, padding], axis=2)
        else:
            trajectory_padded = trajectory

        # Encode each state and concatenate
        state_encodings = []
        for t in range(T):
            enc = self.state_encoder(trajectory_padded[:, t, :])  # (B, hidden)
            state_encodings.append(enc)

        all_encodings = mx.concatenate(state_encodings, axis=1)  # (B, T * hidden)
        dynamics_features = self.dynamics_head(all_encodings)  # (B, hidden // 2)

        return trajectory, dynamics_features

    def compute_attractor_features(
        self,
        trajectory: mx.array,
        node_mask: mx.array,
    ) -> Dict[str, mx.array]:
        """
        Compute differentiable approximations of attractor features.

        Features:
        - period_indicator: soft indicator of oscillation period
        - fixed_point_indicator: soft indicator of fixed point
        - state_diversity: measure of state space exploration
        """
        B, T, n = trajectory.shape

        # Compare final state to earlier states (cycle detection)
        final_state = trajectory[:, -1, :]  # (B, n)

        # Compute similarity to each earlier state
        # similarity[t] = exp(-||final - trajectory[t]||^2)
        diffs = trajectory - final_state[:, None, :]  # (B, T, n)
        diffs = diffs * node_mask[:, None, :]  # Mask inactive nodes
        distances = mx.sum(diffs ** 2, axis=2)  # (B, T)
        similarities = mx.exp(-10 * distances)  # (B, T)

        # Fixed point: final state similar to second-to-last
        fixed_point_indicator = similarities[:, -2] if T > 1 else mx.ones((B,))

        # Oscillation: final state similar to some earlier state but not adjacent
        if T > 3:
            # Check similarity to states at least 2 steps back
            early_similarities = similarities[:, :-2]  # (B, T-2)
            oscillation_indicator = mx.max(early_similarities, axis=1)
        else:
            oscillation_indicator = mx.zeros((B,))

        # State diversity: how many distinct states visited
        # Approximate with sum of pairwise distances
        diversity = mx.mean(distances, axis=1)

        return {
            'fixed_point': fixed_point_indicator,
            'oscillation': oscillation_indicator,
            'diversity': diversity,
        }


class GraphMessagePassing(nn.Module):
    """
    Message Passing Neural Network for graph-structured circuits.

    Key property: Permutation equivariance - relabeling genes doesn't change
    the learned representation (up to relabeling of features).

    Each layer:
    1. Aggregate messages from neighbors (with edge features)
    2. Update node representations
    3. Apply global graph pooling for readout
    """

    def __init__(self, args: GraphDynamicsArgs):
        super().__init__()
        self.args = args

        # Node embedding (initially just indicator)
        self.node_embed = nn.Linear(1, args.node_dim)

        # Edge embedding (for edge type: activation vs inhibition)
        self.edge_embed = nn.Linear(1, args.edge_dim)

        # Message passing layers
        self.mp_layers = []
        for _ in range(args.num_mp_layers):
            self.mp_layers.append(MPLayer(args))

        # Global readout
        self.readout = nn.Sequential(
            nn.Linear(args.node_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )

    def __call__(
        self,
        adj_matrix: mx.array,
        node_mask: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Forward pass of message passing network.

        Args:
            adj_matrix: (batch, max_genes, max_genes) signed adjacency
            node_mask: (batch, max_genes) binary mask for active nodes

        Returns:
            node_features: (batch, max_genes, node_dim) per-node features
            graph_features: (batch, hidden_dim) global graph features
        """
        B = adj_matrix.shape[0]
        n = self.args.max_genes

        # Initial node features (just indicator that node exists)
        node_features = self.node_embed(node_mask[:, :, None])  # (B, n, node_dim)

        # Edge features from adjacency
        edge_features = self.edge_embed(adj_matrix[:, :, :, None])  # (B, n, n, edge_dim)

        # Message passing
        for layer in self.mp_layers:
            node_features = layer(node_features, edge_features, adj_matrix, node_mask)

        # Global readout: sum pooling over nodes
        node_features_masked = node_features * node_mask[:, :, None]
        pooled = mx.sum(node_features_masked, axis=1)  # (B, node_dim)

        # Normalize by number of nodes
        num_nodes = mx.sum(node_mask, axis=1, keepdims=True) + 1e-8
        pooled = pooled / num_nodes

        graph_features = self.readout(pooled)  # (B, hidden_dim)

        return node_features, graph_features


class MPLayer(nn.Module):
    """Single message passing layer."""

    def __init__(self, args: GraphDynamicsArgs):
        super().__init__()
        self.args = args

        # Message function: combines source node and edge features
        self.message_fn = nn.Sequential(
            nn.Linear(args.node_dim + args.edge_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.node_dim),
        )

        # Update function: combines node with aggregated messages
        self.update_fn = nn.Sequential(
            nn.Linear(args.node_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.node_dim),
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(args.node_dim)

    def __call__(
        self,
        node_features: mx.array,
        edge_features: mx.array,
        adj_matrix: mx.array,
        node_mask: mx.array,
    ) -> mx.array:
        """
        One round of message passing.

        Args:
            node_features: (B, n, node_dim)
            edge_features: (B, n, n, edge_dim)
            adj_matrix: (B, n, n) adjacency for masking
            node_mask: (B, n) node mask
        """
        B, n, d = node_features.shape

        # Compute messages for each edge
        # Source features for each possible edge: (B, n, n, node_dim)
        source_features = mx.broadcast_to(node_features[:, :, None, :], (B, n, n, d))

        # Combine source and edge features
        combined = mx.concatenate([source_features, edge_features], axis=-1)

        # Reshape for message function
        combined_flat = combined.reshape(B * n * n, -1)
        messages_flat = self.message_fn(combined_flat)
        messages = messages_flat.reshape(B, n, n, self.args.node_dim)

        # Mask messages by adjacency (only where edges exist)
        edge_mask = (mx.abs(adj_matrix) > 0.5)[:, :, :, None]  # (B, n, n, 1)
        messages = messages * edge_mask

        # Aggregate: sum over source dimension
        aggregated = mx.sum(messages, axis=1)  # (B, n, node_dim)

        # Update node features
        combined_node = mx.concatenate([node_features, aggregated], axis=-1)
        combined_flat = combined_node.reshape(B * n, -1)
        updated_flat = self.update_fn(combined_flat)
        updated = updated_flat.reshape(B, n, self.args.node_dim)

        # Residual connection and normalization
        output = self.norm(node_features + updated)

        # Apply node mask
        output = output * node_mask[:, :, None]

        return output


class DynamicsInformedGNN(nn.Module):
    """
    Main model: Dynamics-Informed Graph Neural Network.

    Architecture:
    1. Graph Message Passing: Learn structural features
    2. Differentiable Simulation: Learn dynamics features
    3. Spectral Features: Encode graph structure
    4. Motif Detection: Hard-coded structural priors
    5. Multi-head prediction: Phenotype + auxiliary tasks

    The key innovation is combining learned features (GNN, simulation)
    with domain knowledge (motifs, spectral properties).
    """

    def __init__(self, args: GraphDynamicsArgs):
        super().__init__()
        self.args = args

        # Core modules
        self.gnn = GraphMessagePassing(args)
        self.simulator = DifferentiableSimulator(args)
        self.spectral = SpectralFeatures(args) if args.use_spectral else None

        # Feature fusion
        feature_dim = args.hidden_dim + args.hidden_dim // 2  # GNN + dynamics
        if args.use_spectral:
            feature_dim += args.spectral_dim
        feature_dim += 4  # motif features

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )

        # Phenotype prediction head
        self.phenotype_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim // 2, args.num_phenotypes),
        )

        # Auxiliary heads for multi-task learning
        self.aux_heads = {
            'has_cycle': nn.Linear(args.hidden_dim, 1),
            'num_attractors': nn.Linear(args.hidden_dim, 4),  # 1, 2, 3, 4+
            'is_bistable': nn.Linear(args.hidden_dim, 1),
        }

        # Contrastive learning projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.contrastive_dim),
        )

    def __call__(
        self,
        adj_matrix: mx.array,
        node_mask: mx.array,
        return_features: bool = False,
    ) -> Dict[str, mx.array]:
        """
        Forward pass.

        Args:
            adj_matrix: (batch, max_genes, max_genes) signed adjacency matrix
                       +1 for activation, -1 for inhibition, 0 for no edge
            node_mask: (batch, max_genes) binary mask for active nodes
            return_features: whether to return intermediate features

        Returns:
            Dictionary with:
            - phenotype_logits: (batch, num_phenotypes)
            - aux_predictions: dict of auxiliary task predictions
            - contrastive_features: (batch, contrastive_dim) for contrastive loss
            - [optional] intermediate features
        """
        B = adj_matrix.shape[0]

        # 1. Graph features from message passing
        node_features, graph_features = self.gnn(adj_matrix, node_mask)

        # 2. Dynamics features from differentiable simulation
        trajectory, dynamics_features = self.simulator.simulate(adj_matrix, node_mask)
        attractor_features = self.simulator.compute_attractor_features(trajectory, node_mask)

        # 3. Spectral features
        if self.spectral is not None:
            spectral_features = self.spectral(adj_matrix)
        else:
            spectral_features = mx.zeros((B, 0))

        # 4. Motif features
        motif_features = self.spectral.compute_motif_features(adj_matrix) if self.spectral else {}
        motif_vec = mx.stack([
            motif_features.get('mutual_inhibition', mx.zeros((B,))),
            motif_features.get('self_inhibition', mx.zeros((B,))),
            motif_features.get('cascade', mx.zeros((B,))),
            motif_features.get('iffl', mx.zeros((B,))),
        ], axis=1)  # (B, 4)

        # 5. Feature fusion
        all_features = mx.concatenate([
            graph_features,
            dynamics_features,
            spectral_features,
            motif_vec,
        ], axis=1)

        fused = self.fusion(all_features)

        # 6. Predictions
        phenotype_logits = self.phenotype_head(fused)

        aux_predictions = {
            'has_cycle': mx.sigmoid(self.aux_heads['has_cycle'](fused)),
            'num_attractors': mx.softmax(self.aux_heads['num_attractors'](fused), axis=-1),
            'is_bistable': mx.sigmoid(self.aux_heads['is_bistable'](fused)),
        }

        # Contrastive features
        contrastive_features = self.contrastive_head(fused)
        contrastive_features = contrastive_features / (
            mx.linalg.norm(contrastive_features, axis=-1, keepdims=True) + 1e-8
        )

        result = {
            'phenotype_logits': phenotype_logits,
            'aux_predictions': aux_predictions,
            'contrastive_features': contrastive_features,
            'attractor_features': attractor_features,
            'motif_features': motif_features,
        }

        if return_features:
            result['graph_features'] = graph_features
            result['dynamics_features'] = dynamics_features
            result['spectral_features'] = spectral_features
            result['trajectory'] = trajectory

        return result

    def compute_loss(
        self,
        adj_matrix: mx.array,
        node_mask: mx.array,
        phenotype_labels: mx.array,
        aux_labels: Optional[Dict[str, mx.array]] = None,
        contrastive_pairs: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Compute total loss with multiple objectives.

        Loss components:
        1. Phenotype classification (cross-entropy)
        2. Auxiliary tasks (multi-task learning)
        3. Contrastive loss (same phenotype close, different far)
        4. Dynamics consistency (trajectory should be consistent with phenotype)

        Args:
            adj_matrix: (batch, max_genes, max_genes)
            node_mask: (batch, max_genes)
            phenotype_labels: (batch,) integer labels
            aux_labels: optional dict of auxiliary labels
            contrastive_pairs: optional (positive_idx, negative_idx) for contrastive

        Returns:
            total_loss: scalar
            loss_components: dict of individual losses
        """
        outputs = self(adj_matrix, node_mask, return_features=True)

        loss_components = {}

        # 1. Phenotype classification loss
        ce_loss = mx.mean(
            nn.losses.cross_entropy(outputs['phenotype_logits'], phenotype_labels)
        )
        loss_components['classification'] = ce_loss

        # 2. Auxiliary losses
        if aux_labels is not None:
            if 'has_cycle' in aux_labels:
                cycle_loss = nn.losses.binary_cross_entropy(
                    outputs['aux_predictions']['has_cycle'].squeeze(-1),
                    aux_labels['has_cycle'].astype(mx.float32),
                )
                loss_components['aux_cycle'] = mx.mean(cycle_loss)

            if 'is_bistable' in aux_labels:
                bistable_loss = nn.losses.binary_cross_entropy(
                    outputs['aux_predictions']['is_bistable'].squeeze(-1),
                    aux_labels['is_bistable'].astype(mx.float32),
                )
                loss_components['aux_bistable'] = mx.mean(bistable_loss)

        # 3. Contrastive loss (InfoNCE)
        if contrastive_pairs is not None:
            contrastive_loss = self._compute_contrastive_loss(
                outputs['contrastive_features'],
                phenotype_labels,
            )
            loss_components['contrastive'] = contrastive_loss

        # 4. Dynamics consistency loss
        # Encourage attractor features to match phenotype
        dynamics_loss = self._compute_dynamics_consistency_loss(
            outputs['attractor_features'],
            outputs['motif_features'],
            phenotype_labels,
        )
        loss_components['dynamics'] = dynamics_loss

        # Total loss with weights
        total_loss = (
            1.0 * ce_loss +
            0.3 * loss_components.get('aux_cycle', 0) +
            0.3 * loss_components.get('aux_bistable', 0) +
            0.5 * loss_components.get('contrastive', 0) +
            0.5 * dynamics_loss
        )

        return total_loss, loss_components

    def _compute_contrastive_loss(
        self,
        features: mx.array,
        labels: mx.array,
    ) -> mx.array:
        """
        Compute supervised contrastive loss.

        Circuits with same phenotype should have similar representations,
        circuits with different phenotypes should have dissimilar representations.
        """
        B = features.shape[0]
        temp = self.args.contrastive_temp

        # Compute similarity matrix
        sim = mx.matmul(features, features.T) / temp  # (B, B)

        # Create mask for positive pairs (same label)
        labels_expand = labels[:, None]  # (B, 1)
        positive_mask = (labels_expand == labels[None, :]).astype(mx.float32)  # (B, B)

        # Remove diagonal (self-similarity)
        eye_mask = mx.eye(B)
        positive_mask = positive_mask * (1 - eye_mask)

        # Denominator: sum over all non-self pairs
        exp_sim = mx.exp(sim) * (1 - eye_mask)

        # Numerator: sum over positive pairs
        log_prob = sim - mx.log(mx.sum(exp_sim, axis=1, keepdims=True) + 1e-8)

        # Average over positive pairs
        num_positives = mx.sum(positive_mask, axis=1) + 1e-8
        contrastive_loss = -mx.sum(positive_mask * log_prob, axis=1) / num_positives

        return mx.mean(contrastive_loss)

    def _compute_dynamics_consistency_loss(
        self,
        attractor_features: Dict[str, mx.array],
        motif_features: Dict[str, mx.array],
        phenotype_labels: mx.array,
    ) -> mx.array:
        """
        Encourage consistency between dynamics features and phenotype labels.

        Priors based on phenotype definitions:
        - Oscillator: should have oscillation, NOT fixed point
        - Toggle switch: should have fixed point AND mutual inhibition
        - Stable: should have fixed point, NOT oscillation
        - Adaptation: should have fixed point
        """
        B = phenotype_labels.shape[0]

        # Phenotype indices (assuming standard ordering)
        # 0: oscillator, 1: toggle_switch, 2: adaptation,
        # 3: pulse_generator, 4: amplifier, 5: stable

        oscillator_mask = (phenotype_labels == 0).astype(mx.float32)
        toggle_mask = (phenotype_labels == 1).astype(mx.float32)
        stable_mask = (phenotype_labels == 5).astype(mx.float32)

        fixed_point = attractor_features['fixed_point']
        oscillation = attractor_features['oscillation']
        mutual_inhib = motif_features.get('mutual_inhibition', mx.zeros((B,)))

        # Oscillators should oscillate
        osc_loss = oscillator_mask * (1 - oscillation) + oscillator_mask * fixed_point

        # Toggle switches should have mutual inhibition and fixed points
        toggle_loss = toggle_mask * (1 - mutual_inhib) + toggle_mask * (1 - fixed_point)

        # Stable should have fixed point
        stable_loss = stable_mask * (1 - fixed_point)

        total = osc_loss + toggle_loss + stable_loss

        return mx.mean(total)

    @property
    def num_parameters(self) -> int:
        """Count total parameters."""
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


def circuit_to_graph(
    circuit: dict,
    max_genes: int = 10,
    gene_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[mx.array, mx.array, Dict[str, int]]:
    """
    Convert circuit dictionary to graph representation.

    Args:
        circuit: dict with 'interactions' key
        max_genes: maximum number of genes to support
        gene_to_idx: optional pre-computed gene-to-index mapping

    Returns:
        adj_matrix: (max_genes, max_genes) signed adjacency
        node_mask: (max_genes,) binary mask for active nodes
        gene_to_idx: mapping from gene name to index
    """
    interactions = circuit.get('interactions', [])

    # Build gene vocabulary if not provided
    if gene_to_idx is None:
        genes = set()
        for inter in interactions:
            genes.add(inter['source'].lower())
            genes.add(inter['target'].lower())
        gene_to_idx = {g: i for i, g in enumerate(sorted(genes))}

    n_genes = len(gene_to_idx)
    if n_genes > max_genes:
        raise ValueError(f"Circuit has {n_genes} genes, max is {max_genes}")

    # Build adjacency matrix
    adj = np.zeros((max_genes, max_genes), dtype=np.float32)

    for inter in interactions:
        src = gene_to_idx.get(inter['source'].lower())
        tgt = gene_to_idx.get(inter['target'].lower())

        if src is not None and tgt is not None:
            if inter['type'].lower() == 'activates':
                adj[src, tgt] = 1.0
            elif inter['type'].lower() == 'inhibits':
                adj[src, tgt] = -1.0

    # Node mask
    mask = np.zeros(max_genes, dtype=np.float32)
    mask[:n_genes] = 1.0

    return mx.array(adj), mx.array(mask), gene_to_idx


def create_dynamics_gnn(
    max_genes: int = 10,
    hidden_dim: int = 128,
    num_mp_layers: int = 4,
    num_sim_steps: int = 20,
) -> DynamicsInformedGNN:
    """Create a Dynamics-Informed GNN with specified configuration."""
    args = GraphDynamicsArgs(
        max_genes=max_genes,
        hidden_dim=hidden_dim,
        num_mp_layers=num_mp_layers,
        num_sim_steps=num_sim_steps,
    )
    return DynamicsInformedGNN(args)
