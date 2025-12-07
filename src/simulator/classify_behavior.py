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
        rule: str = "majority",
    ):
        """
        Initialize classifier.

        Args:
            num_initial_conditions: Number of initial states to test
            max_steps: Maximum simulation steps
            rule: Boolean update rule
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

        # Check for oscillations
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

        # Check for bistability (toggle switch)
        if dynamics["num_fixed_points"] >= 2:
            return "toggle_switch", {
                "reason": "multiple_fixed_points",
                "num_fixed_points": dynamics["num_fixed_points"],
                "fixed_points": dynamics["fixed_points"],
            }

        # Check for adaptation
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

        # Check for pulse generator pattern (IFFL-like)
        if self._is_pulse_generator_topology(network):
            return "pulse_generator", {
                "reason": "iffl_topology",
            }

        # Check for amplifier pattern (cascade)
        if self._is_amplifier_topology(network):
            return "amplifier", {
                "reason": "cascade_topology",
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
