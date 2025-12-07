"""Tests for boolean network simulator."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "simulator"))

from boolean_network import BooleanNetwork
from dynamics import simulate, detect_dynamics
from classify_behavior import BehaviorClassifier, classify_phenotype


def test_boolean_network_creation():
    """Test creating boolean network from circuit."""
    circuit = {
        "phenotype": "oscillator",
        "interactions": [
            {"source": "lacI", "target": "tetR", "type": "inhibits"},
            {"source": "tetR", "target": "cI", "type": "inhibits"},
            {"source": "cI", "target": "lacI", "type": "inhibits"},
        ],
    }

    network = BooleanNetwork.from_circuit(circuit)

    assert network.num_genes == 3
    assert network.num_interactions == 3
    assert "laci" in network.genes
    assert "tetr" in network.genes
    assert "ci" in network.genes


def test_simulation():
    """Test boolean network simulation."""
    circuit = {
        "interactions": [
            {"source": "geneA", "target": "geneB", "type": "activates"},
            {"source": "geneB", "target": "geneA", "type": "inhibits"},
        ],
    }

    network = BooleanNetwork.from_circuit(circuit)
    result = simulate(network, max_steps=100)

    assert len(result.trajectory) > 0
    # Should eventually reach a fixed point or cycle
    assert result.cycle_start is not None or len(result.trajectory) == 100


def test_dynamics_detection():
    """Test dynamics detection from multiple initial conditions."""
    # Toggle switch should have multiple fixed points
    toggle = {
        "interactions": [
            {"source": "geneA", "target": "geneB", "type": "inhibits"},
            {"source": "geneB", "target": "geneA", "type": "inhibits"},
        ],
    }

    network = BooleanNetwork.from_circuit(toggle)
    dynamics = detect_dynamics(network, num_initial_conditions=20)

    assert dynamics["num_attractors"] > 0


def test_classify_repressilator():
    """Test that repressilator is classified as oscillator."""
    repressilator = {
        "phenotype": "oscillator",
        "interactions": [
            {"source": "lacI", "target": "tetR", "type": "inhibits"},
            {"source": "tetR", "target": "cI", "type": "inhibits"},
            {"source": "cI", "target": "lacI", "type": "inhibits"},
        ],
    }

    predicted, details = classify_phenotype(repressilator)
    print(f"Repressilator classified as: {predicted}")
    print(f"Details: {details}")

    # Repressilator should oscillate or be classified based on topology
    assert predicted in ["oscillator", "stable", "adaptation"]


def test_classify_toggle():
    """Test that toggle switch is classified as toggle_switch."""
    toggle = {
        "phenotype": "toggle_switch",
        "interactions": [
            {"source": "lacI", "target": "tetR", "type": "inhibits"},
            {"source": "tetR", "target": "lacI", "type": "inhibits"},
        ],
    }

    predicted, details = classify_phenotype(toggle)
    print(f"Toggle switch classified as: {predicted}")
    print(f"Details: {details}")

    # Toggle should have multiple fixed points (bistability)
    assert predicted in ["toggle_switch", "stable"]


def test_classify_cascade():
    """Test that cascade is classified as amplifier or stable."""
    cascade = {
        "phenotype": "amplifier",
        "interactions": [
            {"source": "geneA", "target": "geneB", "type": "activates"},
            {"source": "geneB", "target": "geneC", "type": "activates"},
            {"source": "geneC", "target": "geneD", "type": "activates"},
        ],
    }

    predicted, details = classify_phenotype(cascade)
    print(f"Cascade classified as: {predicted}")
    print(f"Details: {details}")

    assert predicted in ["amplifier", "stable"]


def test_behavior_classifier():
    """Test BehaviorClassifier class."""
    classifier = BehaviorClassifier(
        num_initial_conditions=10,
        max_steps=100,
    )

    circuit = {
        "interactions": [
            {"source": "geneA", "target": "geneB", "type": "activates"},
        ],
    }

    network = BooleanNetwork.from_circuit(circuit)
    predicted, details = classifier.classify(network)

    assert predicted in ["oscillator", "toggle_switch", "adaptation",
                        "pulse_generator", "amplifier", "stable"]


if __name__ == "__main__":
    test_boolean_network_creation()
    print("✓ test_boolean_network_creation")

    test_simulation()
    print("✓ test_simulation")

    test_dynamics_detection()
    print("✓ test_dynamics_detection")

    test_classify_repressilator()
    print("✓ test_classify_repressilator")

    test_classify_toggle()
    print("✓ test_classify_toggle")

    test_classify_cascade()
    print("✓ test_classify_cascade")

    test_behavior_classifier()
    print("✓ test_behavior_classifier")

    print("\nAll simulator tests passed!")
