"""Boolean network representation and parsing."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BooleanNetwork:
    """
    Boolean network representation of a gene regulatory circuit.

    Each gene has a boolean state (0 or 1) and an update rule based on
    its activators and inhibitors.
    """

    genes: list[str]
    activators: dict[str, list[str]] = field(default_factory=dict)
    inhibitors: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize empty lists for genes without regulators
        for gene in self.genes:
            if gene not in self.activators:
                self.activators[gene] = []
            if gene not in self.inhibitors:
                self.inhibitors[gene] = []

    @classmethod
    def from_circuit(cls, circuit: dict) -> "BooleanNetwork":
        """
        Create BooleanNetwork from circuit dictionary.

        Args:
            circuit: Dictionary with 'interactions' key containing list of
                    {'source', 'target', 'type'} dictionaries

        Returns:
            BooleanNetwork instance
        """
        genes = set()
        activators: dict[str, list[str]] = {}
        inhibitors: dict[str, list[str]] = {}

        for interaction in circuit.get("interactions", []):
            source = interaction["source"].lower()
            target = interaction["target"].lower()
            int_type = interaction["type"].lower()

            genes.add(source)
            genes.add(target)

            if target not in activators:
                activators[target] = []
            if target not in inhibitors:
                inhibitors[target] = []

            if int_type == "activates":
                activators[target].append(source)
            elif int_type == "inhibits":
                inhibitors[target].append(source)

        return cls(
            genes=sorted(list(genes)),
            activators=activators,
            inhibitors=inhibitors,
        )

    @classmethod
    def from_tokens(
        cls,
        tokens: list[str],
        skip_phenotype: bool = True,
    ) -> "BooleanNetwork":
        """
        Create BooleanNetwork from token sequence.

        Args:
            tokens: List of token strings (gene, interaction type)
            skip_phenotype: Whether to skip the first phenotype token

        Returns:
            BooleanNetwork instance
        """
        # Filter special tokens and phenotype
        filtered = []
        for t in tokens:
            if t in ["<bos>", "<eos>", "<pad>", "<unk>"]:
                continue
            if skip_phenotype and t.startswith("<") and t.endswith(">"):
                continue
            filtered.append(t)

        # Parse triplets: source type target
        genes = set()
        activators: dict[str, list[str]] = {}
        inhibitors: dict[str, list[str]] = {}

        i = 0
        while i + 2 < len(filtered):
            source = filtered[i]
            int_type = filtered[i + 1]
            target = filtered[i + 2]

            if int_type not in ["activates", "inhibits"]:
                i += 1
                continue

            genes.add(source)
            genes.add(target)

            if target not in activators:
                activators[target] = []
            if target not in inhibitors:
                inhibitors[target] = []

            if int_type == "activates":
                activators[target].append(source)
            else:
                inhibitors[target].append(source)

            i += 3

        return cls(
            genes=sorted(list(genes)),
            activators=activators,
            inhibitors=inhibitors,
        )

    def update_gene(
        self,
        gene: str,
        state: dict[str, int],
        rule: str = "majority",
    ) -> int:
        """
        Compute next state for a gene.

        Args:
            gene: Gene name
            state: Current state dictionary {gene: 0|1}
            rule: Update rule:
                - "majority": majority of regulators wins
                - "activation_wins": inhibitors block, else activators activate
                - "inhibition_wins": any inhibitor turns gene off
                - "constitutive": genes are ON by default, repressed by inhibitors
                  (biologically accurate for transcriptional regulation)

        Returns:
            Next state (0 or 1)
        """
        # Count active activators and inhibitors
        active_activators = sum(
            state.get(a, 0) for a in self.activators.get(gene, [])
        )
        active_inhibitors = sum(
            state.get(i, 0) for i in self.inhibitors.get(gene, [])
        )

        has_activators = len(self.activators.get(gene, [])) > 0
        has_inhibitors = len(self.inhibitors.get(gene, [])) > 0

        if rule == "constitutive":
            # Biologically accurate rule:
            # - Genes with only inhibitors: ON by default, OFF when repressed
            # - Genes with only activators: OFF by default, ON when activated
            # - Genes with both: ON if activators > inhibitors
            # - Genes with no regulators: maintain state

            if not has_activators and not has_inhibitors:
                return state.get(gene, 0)

            if not has_activators:
                # Only inhibitors: constitutively ON, repressed when inhibited
                return 0 if active_inhibitors > 0 else 1

            if not has_inhibitors:
                # Only activators: OFF unless activated
                return 1 if active_activators > 0 else 0

            # Both activators and inhibitors: compare counts
            if active_activators > active_inhibitors:
                return 1
            elif active_inhibitors > active_activators:
                return 0
            else:
                # Tie: activators win (gene is expressed)
                return 1 if active_activators > 0 else 0

        elif rule == "majority":
            # No regulators: maintain current state
            if active_activators == 0 and active_inhibitors == 0:
                return state.get(gene, 0)
            # Majority rule
            if active_activators > active_inhibitors:
                return 1
            elif active_inhibitors > active_activators:
                return 0
            else:
                # Tie: maintain current state
                return state.get(gene, 0)

        elif rule == "activation_wins":
            # Gene is ON if any activator is ON and no inhibitor is ON
            if active_inhibitors > 0:
                return 0
            elif active_activators > 0:
                return 1
            else:
                return state.get(gene, 0)

        elif rule == "inhibition_wins":
            # Gene is ON only if activator ON and all inhibitors OFF
            if active_inhibitors > 0:
                return 0
            elif active_activators > 0:
                return 1
            else:
                return 0

        else:
            raise ValueError(f"Unknown rule: {rule}")

    def synchronous_update(
        self,
        state: dict[str, int],
        rule: str = "majority",
    ) -> dict[str, int]:
        """
        Perform synchronous update of all genes.

        Args:
            state: Current state dictionary
            rule: Update rule to use

        Returns:
            Next state dictionary
        """
        next_state = {}
        for gene in self.genes:
            next_state[gene] = self.update_gene(gene, state, rule)
        return next_state

    def random_state(self, rng: Optional[np.random.Generator] = None) -> dict[str, int]:
        """Generate random initial state."""
        if rng is None:
            rng = np.random.default_rng()
        return {gene: int(rng.integers(0, 2)) for gene in self.genes}

    def state_to_tuple(self, state: dict[str, int]) -> tuple:
        """Convert state dict to hashable tuple."""
        return tuple(state[g] for g in self.genes)

    def tuple_to_state(self, state_tuple: tuple) -> dict[str, int]:
        """Convert tuple back to state dict."""
        return {g: s for g, s in zip(self.genes, state_tuple)}

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "genes": self.genes,
            "activators": self.activators,
            "inhibitors": self.inhibitors,
        }

    @property
    def num_genes(self) -> int:
        return len(self.genes)

    @property
    def num_interactions(self) -> int:
        total = 0
        for gene in self.genes:
            total += len(self.activators.get(gene, []))
            total += len(self.inhibitors.get(gene, []))
        return total

    def __repr__(self) -> str:
        return f"BooleanNetwork(genes={self.genes}, interactions={self.num_interactions})"
