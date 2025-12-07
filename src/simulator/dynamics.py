"""Dynamics simulation for boolean networks."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .boolean_network import BooleanNetwork
except ImportError:
    from boolean_network import BooleanNetwork


@dataclass
class SimulationResult:
    """Result of a boolean network simulation."""

    trajectory: list[tuple]  # List of state tuples
    cycle_start: Optional[int]  # Index where cycle begins (None if no cycle found)
    cycle_length: Optional[int]  # Length of cycle (None if no cycle)
    attractor: Optional[list[tuple]]  # States in the attractor
    is_fixed_point: bool  # Whether attractor is a single fixed point


def simulate(
    network: BooleanNetwork,
    initial_state: Optional[dict[str, int]] = None,
    max_steps: int = 200,
    rule: str = "majority",
    rng: Optional[np.random.Generator] = None,
) -> SimulationResult:
    """
    Simulate boolean network dynamics.

    Args:
        network: BooleanNetwork to simulate
        initial_state: Initial state (random if None)
        max_steps: Maximum simulation steps
        rule: Update rule to use
        rng: Random number generator

    Returns:
        SimulationResult with trajectory and attractor info
    """
    if initial_state is None:
        initial_state = network.random_state(rng)

    trajectory = []
    state_to_idx: dict[tuple, int] = {}

    current_state = initial_state
    cycle_start = None
    cycle_length = None

    for step in range(max_steps):
        state_tuple = network.state_to_tuple(current_state)
        trajectory.append(state_tuple)

        # Check if we've seen this state before (cycle detection)
        if state_tuple in state_to_idx:
            cycle_start = state_to_idx[state_tuple]
            cycle_length = step - cycle_start
            break

        state_to_idx[state_tuple] = step

        # Update
        current_state = network.synchronous_update(current_state, rule)

    # Extract attractor
    attractor = None
    is_fixed_point = False

    if cycle_start is not None:
        attractor = trajectory[cycle_start:]
        is_fixed_point = len(attractor) == 1

    return SimulationResult(
        trajectory=trajectory,
        cycle_start=cycle_start,
        cycle_length=cycle_length,
        attractor=attractor,
        is_fixed_point=is_fixed_point,
    )


def detect_dynamics(
    network: BooleanNetwork,
    num_initial_conditions: int = 20,
    max_steps: int = 200,
    rule: str = "majority",
    seed: int = 42,
) -> dict:
    """
    Analyze dynamics from multiple initial conditions.

    Args:
        network: BooleanNetwork to analyze
        num_initial_conditions: Number of random initial states
        max_steps: Maximum steps per simulation
        rule: Update rule
        seed: Random seed

    Returns:
        Dictionary with dynamics analysis
    """
    rng = np.random.default_rng(seed)

    attractors: dict[tuple, int] = {}  # attractor -> count
    fixed_points: set[tuple] = set()
    oscillations: list[int] = []  # cycle lengths

    for _ in range(num_initial_conditions):
        result = simulate(network, max_steps=max_steps, rule=rule, rng=rng)

        if result.attractor is not None:
            # Use first state as attractor identifier
            attractor_key = tuple(result.attractor)
            attractors[attractor_key] = attractors.get(attractor_key, 0) + 1

            if result.is_fixed_point:
                fixed_points.add(result.attractor[0])
            else:
                oscillations.append(result.cycle_length)

    return {
        "num_attractors": len(attractors),
        "num_fixed_points": len(fixed_points),
        "fixed_points": list(fixed_points),
        "num_oscillations": len(oscillations),
        "oscillation_periods": oscillations,
        "attractor_counts": attractors,
    }


def test_adaptation(
    network: BooleanNetwork,
    perturbation_gene: Optional[str] = None,
    stabilization_steps: int = 50,
    recovery_steps: int = 50,
    rule: str = "majority",
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Test for adaptation behavior.

    Adaptation: system returns to baseline after perturbation.

    Args:
        network: BooleanNetwork to test
        perturbation_gene: Gene to perturb (random if None)
        stabilization_steps: Steps to reach steady state
        recovery_steps: Steps after perturbation to check recovery
        rule: Update rule
        rng: Random number generator

    Returns:
        Dictionary with adaptation analysis
    """
    if rng is None:
        rng = np.random.default_rng()

    # Run to steady state
    result = simulate(
        network, max_steps=stabilization_steps, rule=rule, rng=rng
    )

    if result.attractor is None or not result.is_fixed_point:
        # Not at fixed point, can't test classical adaptation
        return {
            "is_adaptive": False,
            "reason": "no_fixed_point",
            "baseline": None,
            "perturbed": None,
            "recovered": None,
        }

    baseline_state = network.tuple_to_state(result.attractor[0])

    # Choose gene to perturb
    if perturbation_gene is None:
        perturbation_gene = rng.choice(network.genes)

    # Perturb: flip the gene state
    perturbed_state = baseline_state.copy()
    perturbed_state[perturbation_gene] = 1 - perturbed_state[perturbation_gene]

    # Simulate from perturbed state
    recovery_result = simulate(
        network,
        initial_state=perturbed_state,
        max_steps=recovery_steps,
        rule=rule,
        rng=rng,
    )

    # Check if we return to baseline
    final_state = network.tuple_to_state(recovery_result.trajectory[-1])
    returned_to_baseline = baseline_state == final_state

    return {
        "is_adaptive": returned_to_baseline,
        "reason": "returned_to_baseline" if returned_to_baseline else "did_not_return",
        "baseline": baseline_state,
        "perturbed_gene": perturbation_gene,
        "recovered": final_state,
        "steps_to_recover": (
            len(recovery_result.trajectory) if returned_to_baseline else None
        ),
    }


def compute_trajectory_autocorrelation(
    trajectory: list[tuple],
    max_lag: int = 50,
) -> list[float]:
    """
    Compute autocorrelation of trajectory for oscillation detection.

    Args:
        trajectory: List of state tuples
        max_lag: Maximum lag to compute

    Returns:
        List of autocorrelation values for lags 0 to max_lag
    """
    if len(trajectory) < 3:
        return [1.0]

    # Convert to numpy array
    arr = np.array([list(t) for t in trajectory], dtype=float)

    # Flatten to 1D for autocorrelation
    flat = arr.flatten()

    # Normalize
    flat = flat - flat.mean()
    if flat.std() < 1e-8:
        return [1.0] + [0.0] * min(max_lag, len(trajectory) - 1)

    flat = flat / flat.std()

    # Compute autocorrelation
    autocorr = []
    n = len(flat)
    for lag in range(min(max_lag + 1, n)):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = np.correlate(flat[:-lag], flat[lag:])[0] / (n - lag)
            autocorr.append(float(corr))

    return autocorr
