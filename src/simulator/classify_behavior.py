"""Classify dynamical behavior of boolean networks."""

from typing import Optional

import numpy as np

try:
    from .boolean_network import BooleanNetwork
    from .dynamics import simulate, detect_dynamics, test_adaptation
except ImportError:
    from boolean_network import BooleanNetwork
    from dynamics import simulate, detect_dynamics, test_adaptation


class BehaviorClassifier:
    """Classifier for boolean network dynamical behavior."""

    PHENOTYPES = [
        "oscillator",
        "toggle_switch",
        "adaptation",
        "pulse_generator",
        "amplifier",
        "stable",
    ]

    def __init__(
        self,
        num_initial_conditions: int = 20,
        max_steps: int = 200,
        rule: str = "constitutive",
    ):
        """
        Initialize classifier.

        Args:
            num_initial_conditions: Number of initial states to test
            max_steps: Maximum simulation steps
            rule: Boolean update rule (default: "constitutive" for biologically
                  accurate simulation of transcriptional regulation)
        """
        self.num_initial_conditions = num_initial_conditions
        self.max_steps = max_steps
        self.rule = rule

    def classify(
        self,
        network: BooleanNetwork,
        seed: int = 42,
    ) -> tuple[str, dict]:
        """
        Classify network behavior.

        Args:
            network: BooleanNetwork to classify
            seed: Random seed for reproducibility

        Returns:
            Tuple of (phenotype_name, details_dict)
        """
        if network.num_genes == 0:
            return "stable", {"reason": "empty_network"}

        # Analyze dynamics
        dynamics = detect_dynamics(
            network,
            num_initial_conditions=self.num_initial_conditions,
            max_steps=self.max_steps,
            rule=self.rule,
            seed=seed,
        )

        # Check for bistability (toggle switch)
        # Must have multiple fixed points AND mutual inhibition topology
        if dynamics["num_fixed_points"] >= 2:
            # Verify it has mutual inhibition (true toggle switch)
            if self._has_mutual_inhibition(network):
                return "toggle_switch", {
                    "reason": "multiple_fixed_points_with_mutual_inhibition",
                    "num_fixed_points": dynamics["num_fixed_points"],
                    "fixed_points": dynamics["fixed_points"],
                }

        # Check for oscillations (only if not bistable)
        if dynamics["num_oscillations"] > 0:
            # Has oscillatory attractors
            periods = dynamics["oscillation_periods"]
            avg_period = np.mean(periods) if periods else 0

            if avg_period >= 2:
                return "oscillator", {
                    "reason": "sustained_oscillation",
                    "periods": periods,
                    "num_attractors": dynamics["num_attractors"],
                }

        # Check for pulse generator pattern (IFFL-like) BEFORE adaptation
        # IFFL also shows adaptation-like behavior, but the topology is specific
        if self._is_pulse_generator_topology(network):
            return "pulse_generator", {
                "reason": "iffl_topology",
            }

        # Check for amplifier pattern (cascade) BEFORE adaptation
        # Cascades are stable but shouldn't be classified as adaptation
        if self._is_amplifier_topology(network):
            return "amplifier", {
                "reason": "cascade_topology",
            }

        # Check for adaptation (only for non-cascade, non-IFFL circuits)
        # Adaptation requires negative feedback that creates transient response
        if self._has_negative_feedback_loop(network):
            rng = np.random.default_rng(seed)
            adaptation_tests = []
            for gene in network.genes[:3]:  # Test a few genes
                result = test_adaptation(
                    network,
                    perturbation_gene=gene,
                    rule=self.rule,
                    rng=rng,
                )
                adaptation_tests.append(result)

            if any(t["is_adaptive"] for t in adaptation_tests):
                return "adaptation", {
                    "reason": "returns_to_baseline",
                    "tests": adaptation_tests,
                }

        # Default: stable
        return "stable", {
            "reason": "single_attractor_or_default",
            "num_fixed_points": dynamics["num_fixed_points"],
        }

    def _is_pulse_generator_topology(self, network: BooleanNetwork) -> bool:
        """
        Check if network has incoherent feed-forward loop topology.

        IFFL: A->B, A->C, B-|C (A activates both B and C, B inhibits C)
        """
        # Look for IFFL pattern
        for gene_a in network.genes:
            # Find genes that A activates
            targets_of_a = set()
            for target, activators in network.activators.items():
                if gene_a in activators:
                    targets_of_a.add(target)

            # Check if any pair forms IFFL
            for gene_b in targets_of_a:
                for gene_c in targets_of_a:
                    if gene_b != gene_c:
                        # Check if B inhibits C
                        if gene_b in network.inhibitors.get(gene_c, []):
                            return True

        return False

    def _is_amplifier_topology(self, network: BooleanNetwork) -> bool:
        """
        Check if network has cascade amplifier topology.

        Cascade: A->B->C->... (chain of activations)
        """
        if network.num_genes < 2:
            return False

        # Check for activation chain without negative feedback
        has_chain = False
        has_negative_feedback = False

        # Look for activation chains
        for start_gene in network.genes:
            visited = {start_gene}
            current = start_gene
            chain_length = 0

            while True:
                # Find next in chain
                next_gene = None
                for target, activators in network.activators.items():
                    if current in activators and target not in visited:
                        next_gene = target
                        break

                if next_gene is None:
                    break

                visited.add(next_gene)
                current = next_gene
                chain_length += 1

            if chain_length >= 2:
                has_chain = True

        # Check for negative feedback (would make it oscillator)
        for gene in network.genes:
            for inhibitor in network.inhibitors.get(gene, []):
                # Check if inhibitor is downstream
                if self._is_downstream(network, gene, inhibitor):
                    has_negative_feedback = True
                    break

        return has_chain and not has_negative_feedback

    def _has_mutual_inhibition(self, network: BooleanNetwork) -> bool:
        """
        Check if network has mutual inhibition (A inhibits B AND B inhibits A).

        This is the hallmark of toggle switch topology.
        """
        for gene_a in network.genes:
            for gene_b in network.genes:
                if gene_a != gene_b:
                    # Check if A inhibits B
                    a_inhibits_b = gene_a in network.inhibitors.get(gene_b, [])
                    # Check if B inhibits A
                    b_inhibits_a = gene_b in network.inhibitors.get(gene_a, [])

                    if a_inhibits_b and b_inhibits_a:
                        return True
        return False

    def _has_negative_feedback_loop(self, network: BooleanNetwork) -> bool:
        """
        Check if network has a negative feedback loop.

        A negative feedback loop is a cycle with an odd number of inhibitions.
        This is required for true adaptation behavior.
        """
        # Look for cycles that contain at least one inhibition
        for start_gene in network.genes:
            # BFS to find cycles back to start
            visited = set()
            # Track (current_gene, inhibition_count)
            queue = [(start_gene, 0)]

            while queue:
                current, inhib_count = queue.pop(0)
                if current in visited and current != start_gene:
                    continue

                # Check outgoing edges
                for target, activators in network.activators.items():
                    if current in activators:
                        if target == start_gene and inhib_count > 0:
                            # Found cycle back to start with some inhibitions
                            return True
                        if target not in visited:
                            queue.append((target, inhib_count))
                            visited.add(target)

                for target, inhibitors in network.inhibitors.items():
                    if current in inhibitors:
                        if target == start_gene:
                            # Found cycle back to start through inhibition
                            return True
                        if target not in visited:
                            queue.append((target, inhib_count + 1))
                            visited.add(target)

        return False

    def _is_downstream(
        self,
        network: BooleanNetwork,
        start: str,
        target: str,
        max_depth: int = 10,
    ) -> bool:
        """Check if target is downstream of start in activation cascade."""
        if start == target:
            return True

        visited = {start}
        frontier = [start]

        for _ in range(max_depth):
            new_frontier = []
            for current in frontier:
                for gene, activators in network.activators.items():
                    if current in activators and gene not in visited:
                        if gene == target:
                            return True
                        visited.add(gene)
                        new_frontier.append(gene)
            frontier = new_frontier
            if not frontier:
                break

        return False

    def classify_batch(
        self,
        networks: list[BooleanNetwork],
        seed: int = 42,
    ) -> list[tuple[str, dict]]:
        """Classify multiple networks."""
        return [self.classify(net, seed=seed + i) for i, net in enumerate(networks)]

    def compute_shaped_reward(
        self,
        network: BooleanNetwork,
        intended_phenotype: str,
        seed: int = 42,
    ) -> tuple[float, dict]:
        """
        Compute shaped reward with partial credit for behaviors.

        Instead of binary 0/1, gives partial credit based on how close
        the dynamics are to the intended phenotype.

        Args:
            network: BooleanNetwork to evaluate
            intended_phenotype: Target phenotype name
            seed: Random seed

        Returns:
            Tuple of (reward, details_dict)
            Reward is in range [0, 1]
        """
        if network.num_genes == 0:
            return 0.0, {"reason": "empty_network"}

        # Get dynamics info
        dynamics = detect_dynamics(
            network,
            num_initial_conditions=self.num_initial_conditions,
            max_steps=self.max_steps,
            rule=self.rule,
            seed=seed,
        )

        # Get actual classification
        predicted, details = self.classify(network, seed=seed)

        # Perfect match gets full reward
        if predicted == intended_phenotype:
            return 1.0, {"match": True, "predicted": predicted}

        # Compute partial reward based on intended phenotype
        reward = 0.0
        reason = "no_match"

        if intended_phenotype == "oscillator":
            # Partial credit for:
            # - Having any oscillations (even if not dominant)
            # - Having multiple attractors (complexity)
            # - Having cycles with period > 1
            if dynamics["num_oscillations"] > 0:
                periods = dynamics["oscillation_periods"]
                avg_period = np.mean(periods) if periods else 0
                # More credit for longer periods (up to period 10)
                period_score = min(avg_period / 10.0, 1.0) * 0.5
                # Credit for having oscillations at all
                osc_ratio = dynamics["num_oscillations"] / max(self.num_initial_conditions, 1)
                osc_score = osc_ratio * 0.3
                reward = period_score + osc_score
                reason = "partial_oscillation"
            elif dynamics["num_attractors"] > 1:
                # Multiple attractors is somewhat oscillator-like (complex)
                reward = 0.1
                reason = "multiple_attractors"

        elif intended_phenotype == "toggle_switch":
            # Partial credit for:
            # - Having any fixed points
            # - Having multiple attractors
            num_fp = dynamics["num_fixed_points"]
            if num_fp >= 2:
                reward = 1.0  # This should match, but just in case
                reason = "bistable"
            elif num_fp == 1:
                # Has stability, just not bistable
                reward = 0.3
                reason = "monostable"
            elif dynamics["num_attractors"] > 1:
                reward = 0.2
                reason = "multiple_attractors"

        elif intended_phenotype == "adaptation":
            # Partial credit for returning to baseline-ish
            rng = np.random.default_rng(seed)
            adaptation_scores = []
            for gene in network.genes[:3]:
                result = test_adaptation(
                    network,
                    perturbation_gene=gene,
                    rule=self.rule,
                    rng=rng,
                )
                if result["is_adaptive"]:
                    adaptation_scores.append(1.0)
                elif result.get("baseline") and result.get("recovered"):
                    # Partial credit if it went somewhere stable
                    adaptation_scores.append(0.2)
                else:
                    adaptation_scores.append(0.0)
            if adaptation_scores:
                reward = np.mean(adaptation_scores) * 0.8
                reason = "partial_adaptation"

        elif intended_phenotype == "pulse_generator":
            # Partial credit for IFFL-like topology
            if self._is_pulse_generator_topology(network):
                reward = 0.8  # Has topology but dynamics didn't match
                reason = "iffl_topology_present"
            elif self._has_feedforward_motif(network):
                reward = 0.3
                reason = "has_feedforward"

        elif intended_phenotype == "amplifier":
            # Partial credit for cascade topology
            if self._is_amplifier_topology(network):
                reward = 0.8
                reason = "cascade_topology_present"
            elif self._has_activation_chain(network):
                reward = 0.3
                reason = "has_activation_chain"

        elif intended_phenotype == "stable":
            # Partial credit for having fixed points
            if dynamics["num_fixed_points"] >= 1:
                reward = 0.5  # Has stability
                reason = "has_fixed_point"

        return reward, {
            "match": False,
            "predicted": predicted,
            "intended": intended_phenotype,
            "partial_reason": reason,
        }

    def _has_feedforward_motif(self, network: BooleanNetwork) -> bool:
        """Check if network has any feedforward motif (A->B, A->C, B->C)."""
        for gene_a in network.genes:
            targets = set()
            for target, activators in network.activators.items():
                if gene_a in activators:
                    targets.add(target)
            for target, inhibitors in network.inhibitors.items():
                if gene_a in inhibitors:
                    targets.add(target)

            # Check if any two targets are connected
            for gene_b in targets:
                for gene_c in targets:
                    if gene_b != gene_c:
                        # B -> C or B -| C
                        if gene_b in network.activators.get(gene_c, []):
                            return True
                        if gene_b in network.inhibitors.get(gene_c, []):
                            return True
        return False

    def _has_activation_chain(self, network: BooleanNetwork) -> bool:
        """Check if network has an activation chain of length >= 2."""
        for start_gene in network.genes:
            visited = {start_gene}
            current = start_gene
            chain_length = 0

            while chain_length < 5:  # Limit search depth
                next_gene = None
                for target, activators in network.activators.items():
                    if current in activators and target not in visited:
                        next_gene = target
                        break
                if next_gene is None:
                    break
                visited.add(next_gene)
                current = next_gene
                chain_length += 1

            if chain_length >= 2:
                return True
        return False


def classify_phenotype(
    circuit: dict,
    num_initial_conditions: int = 20,
    max_steps: int = 200,
    rule: str = "majority",
    seed: int = 42,
) -> tuple[str, dict]:
    """
    Convenience function to classify a circuit's phenotype.

    Args:
        circuit: Circuit dictionary with 'interactions'
        num_initial_conditions: Number of initial states to test
        max_steps: Maximum simulation steps
        rule: Boolean update rule
        seed: Random seed

    Returns:
        Tuple of (phenotype_name, details_dict)
    """
    network = BooleanNetwork.from_circuit(circuit)
    classifier = BehaviorClassifier(
        num_initial_conditions=num_initial_conditions,
        max_steps=max_steps,
        rule=rule,
    )
    return classifier.classify(network, seed=seed)


def validate_circuit_phenotype(
    circuit: dict,
    expected_phenotype: str,
    **kwargs,
) -> bool:
    """
    Check if circuit's simulated behavior matches expected phenotype.

    Args:
        circuit: Circuit dictionary
        expected_phenotype: Expected phenotype string
        **kwargs: Arguments for classify_phenotype

    Returns:
        True if predicted phenotype matches expected
    """
    predicted, _ = classify_phenotype(circuit, **kwargs)
    return predicted == expected_phenotype
