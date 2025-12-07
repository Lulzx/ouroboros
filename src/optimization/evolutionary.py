"""
Classical Evolutionary Optimization for GRN Circuit Design.

This implements model-based search methods that represent the traditional
approach to synthetic biology circuit design:

1. Random initialization of circuit topologies
2. Fitness evaluation via simulation (boolean network oracle)
3. Selection, mutation, and crossover
4. Iterate until target phenotype is achieved

This serves as a baseline to compare against Ouroboros's amortized generation.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from ..simulator.boolean_network import BooleanNetwork
from ..simulator.classify_behavior import BehaviorClassifier


@dataclass
class Circuit:
    """A circuit representation for evolutionary optimization."""

    genes: list[str]
    edges: list[tuple[int, int, int]]  # (src_idx, tgt_idx, type: 1=activate, -1=inhibit)
    fitness: float = 0.0
    phenotype: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to standard circuit dictionary format."""
        interactions = []
        for src_idx, tgt_idx, edge_type in self.edges:
            interactions.append({
                "source": self.genes[src_idx],
                "target": self.genes[tgt_idx],
                "type": "activates" if edge_type == 1 else "inhibits"
            })
        return {"interactions": interactions, "genes": self.genes}

    def to_network(self) -> BooleanNetwork:
        """Convert to BooleanNetwork for simulation."""
        return BooleanNetwork.from_circuit(self.to_dict())

    def copy(self) -> "Circuit":
        """Create a deep copy."""
        return Circuit(
            genes=self.genes.copy(),
            edges=self.edges.copy(),
            fitness=self.fitness,
            phenotype=self.phenotype
        )


@dataclass
class OptimizationResult:
    """Result from an optimization run."""

    success: bool
    best_circuit: Optional[Circuit]
    generations: int
    evaluations: int
    wall_time: float
    fitness_history: list[float] = field(default_factory=list)
    phenotype_history: list[str] = field(default_factory=list)


class EvolutionaryOptimizer:
    """
    Evolutionary optimizer for GRN circuit design.

    This is a (μ + λ) evolutionary strategy with:
    - μ parents selected from population
    - λ offspring created via mutation/crossover
    - Tournament selection
    - Adaptive mutation rates
    """

    def __init__(
        self,
        gene_pool: list[str],
        num_genes: int = 3,
        population_size: int = 50,
        offspring_size: int = 50,
        tournament_size: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        max_edges: int = 9,
        seed: int = 42,
    ):
        """
        Initialize optimizer.

        Args:
            gene_pool: Available gene names
            num_genes: Number of genes per circuit
            population_size: μ (parent population size)
            offspring_size: λ (offspring per generation)
            tournament_size: Size of tournament selection
            mutation_rate: Probability of mutation per edge
            crossover_rate: Probability of crossover
            max_edges: Maximum edges in a circuit
            seed: Random seed
        """
        self.gene_pool = gene_pool
        self.num_genes = num_genes
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_edges = max_edges
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Classifier for fitness evaluation
        self.classifier = BehaviorClassifier(
            num_initial_conditions=20,
            max_steps=200,
            rule="constitutive"
        )

        # Statistics
        self.evaluation_count = 0

    def random_circuit(self) -> Circuit:
        """Generate a random circuit."""
        genes = self.rng.sample(self.gene_pool, min(self.num_genes, len(self.gene_pool)))

        # Random number of edges (at least 1)
        num_edges = self.rng.randint(1, min(self.max_edges, self.num_genes ** 2))

        edges = []
        for _ in range(num_edges):
            src = self.rng.randint(0, len(genes) - 1)
            tgt = self.rng.randint(0, len(genes) - 1)
            edge_type = self.rng.choice([1, -1])
            edges.append((src, tgt, edge_type))

        # Remove duplicates
        edges = list(set(edges))

        return Circuit(genes=genes, edges=edges)

    def evaluate(self, circuit: Circuit, target_phenotype: str) -> float:
        """
        Evaluate circuit fitness for target phenotype.

        Returns:
            Fitness score in [0, 1], where 1 = exact match
        """
        self.evaluation_count += 1

        if not circuit.edges:
            return 0.0

        network = circuit.to_network()
        reward, details = self.classifier.compute_shaped_reward(
            network, target_phenotype
        )

        circuit.fitness = reward
        circuit.phenotype = details.get("predicted", "unknown")

        return reward

    def mutate(self, circuit: Circuit) -> Circuit:
        """
        Mutate a circuit.

        Mutations:
        1. Add an edge
        2. Remove an edge
        3. Change edge type (activate <-> inhibit)
        4. Change edge source/target
        """
        child = circuit.copy()

        if not child.edges:
            # Add an edge if empty
            src = self.rng.randint(0, len(child.genes) - 1)
            tgt = self.rng.randint(0, len(child.genes) - 1)
            edge_type = self.rng.choice([1, -1])
            child.edges.append((src, tgt, edge_type))
            return child

        mutation_type = self.rng.choice(["add", "remove", "flip", "rewire"])

        if mutation_type == "add" and len(child.edges) < self.max_edges:
            src = self.rng.randint(0, len(child.genes) - 1)
            tgt = self.rng.randint(0, len(child.genes) - 1)
            edge_type = self.rng.choice([1, -1])
            new_edge = (src, tgt, edge_type)
            if new_edge not in child.edges:
                child.edges.append(new_edge)

        elif mutation_type == "remove" and len(child.edges) > 1:
            idx = self.rng.randint(0, len(child.edges) - 1)
            child.edges.pop(idx)

        elif mutation_type == "flip":
            idx = self.rng.randint(0, len(child.edges) - 1)
            src, tgt, etype = child.edges[idx]
            child.edges[idx] = (src, tgt, -etype)

        elif mutation_type == "rewire":
            idx = self.rng.randint(0, len(child.edges) - 1)
            src, tgt, etype = child.edges[idx]
            if self.rng.random() < 0.5:
                src = self.rng.randint(0, len(child.genes) - 1)
            else:
                tgt = self.rng.randint(0, len(child.genes) - 1)
            child.edges[idx] = (src, tgt, etype)

        # Remove duplicates
        child.edges = list(set(child.edges))

        return child

    def crossover(self, parent1: Circuit, parent2: Circuit) -> Circuit:
        """
        Crossover two circuits.

        Strategy: uniform crossover of edges (with same genes).
        """
        # Use genes from parent1
        child = Circuit(genes=parent1.genes.copy(), edges=[])

        all_edges = list(set(parent1.edges + parent2.edges))
        for edge in all_edges:
            src, tgt, _ = edge
            if src < len(child.genes) and tgt < len(child.genes):
                if self.rng.random() < 0.5:
                    child.edges.append(edge)

        # Ensure at least one edge
        if not child.edges and parent1.edges:
            child.edges.append(self.rng.choice(parent1.edges))

        return child

    def tournament_select(self, population: list[Circuit]) -> Circuit:
        """Select a parent via tournament selection."""
        competitors = self.rng.sample(population, min(self.tournament_size, len(population)))
        return max(competitors, key=lambda c: c.fitness)

    def optimize(
        self,
        target_phenotype: str,
        max_generations: int = 100,
        max_evaluations: int = 10000,
        early_stop_fitness: float = 1.0,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Run evolutionary optimization to find circuit with target phenotype.

        Args:
            target_phenotype: Target phenotype to achieve
            max_generations: Maximum generations
            max_evaluations: Maximum fitness evaluations
            early_stop_fitness: Stop when fitness >= this value
            verbose: Print progress

        Returns:
            OptimizationResult with best circuit and statistics
        """
        start_time = time.time()
        self.evaluation_count = 0

        # Initialize population
        population = [self.random_circuit() for _ in range(self.population_size)]

        # Evaluate initial population
        for circuit in population:
            self.evaluate(circuit, target_phenotype)

        fitness_history = []
        phenotype_history = []
        best_circuit = max(population, key=lambda c: c.fitness)

        for gen in range(max_generations):
            # Check termination
            if best_circuit.fitness >= early_stop_fitness:
                if verbose:
                    print(f"Gen {gen}: Found solution with fitness {best_circuit.fitness:.3f}")
                break

            if self.evaluation_count >= max_evaluations:
                if verbose:
                    print(f"Gen {gen}: Max evaluations reached")
                break

            # Create offspring
            offspring = []
            while len(offspring) < self.offspring_size:
                parent1 = self.tournament_select(population)

                if self.rng.random() < self.crossover_rate:
                    parent2 = self.tournament_select(population)
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                if self.rng.random() < self.mutation_rate:
                    child = self.mutate(child)

                offspring.append(child)

            # Evaluate offspring
            for circuit in offspring:
                self.evaluate(circuit, target_phenotype)

            # (μ + λ) selection: keep best from parents + offspring
            combined = population + offspring
            combined.sort(key=lambda c: c.fitness, reverse=True)
            population = combined[:self.population_size]

            # Track best
            gen_best = population[0]
            if gen_best.fitness > best_circuit.fitness:
                best_circuit = gen_best

            fitness_history.append(best_circuit.fitness)
            phenotype_history.append(best_circuit.phenotype)

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: best fitness = {best_circuit.fitness:.3f}, "
                      f"phenotype = {best_circuit.phenotype}, "
                      f"evals = {self.evaluation_count}")

        wall_time = time.time() - start_time
        success = best_circuit.fitness >= early_stop_fitness

        return OptimizationResult(
            success=success,
            best_circuit=best_circuit if success else None,
            generations=gen + 1,
            evaluations=self.evaluation_count,
            wall_time=wall_time,
            fitness_history=fitness_history,
            phenotype_history=phenotype_history,
        )


class RandomSearchOptimizer:
    """
    Random search baseline: repeatedly generate random circuits until success.

    This represents the simplest possible baseline.
    """

    def __init__(
        self,
        gene_pool: list[str],
        num_genes: int = 3,
        max_edges: int = 9,
        seed: int = 42,
    ):
        self.gene_pool = gene_pool
        self.num_genes = num_genes
        self.max_edges = max_edges
        self.rng = random.Random(seed)

        self.classifier = BehaviorClassifier(
            num_initial_conditions=20,
            max_steps=200,
            rule="constitutive"
        )
        self.evaluation_count = 0

    def random_circuit(self) -> Circuit:
        """Generate a random circuit."""
        genes = self.rng.sample(self.gene_pool, min(self.num_genes, len(self.gene_pool)))
        num_edges = self.rng.randint(1, min(self.max_edges, self.num_genes ** 2))

        edges = []
        for _ in range(num_edges):
            src = self.rng.randint(0, len(genes) - 1)
            tgt = self.rng.randint(0, len(genes) - 1)
            edge_type = self.rng.choice([1, -1])
            edges.append((src, tgt, edge_type))

        edges = list(set(edges))
        return Circuit(genes=genes, edges=edges)

    def optimize(
        self,
        target_phenotype: str,
        max_evaluations: int = 10000,
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run random search until finding target phenotype."""
        start_time = time.time()
        self.evaluation_count = 0

        best_circuit = None
        best_fitness = 0.0

        for i in range(max_evaluations):
            circuit = self.random_circuit()
            network = circuit.to_network()

            predicted, _ = self.classifier.classify(network)
            self.evaluation_count += 1

            fitness = 1.0 if predicted == target_phenotype else 0.0
            circuit.fitness = fitness
            circuit.phenotype = predicted

            if fitness > best_fitness:
                best_fitness = fitness
                best_circuit = circuit

            if fitness >= 1.0:
                if verbose:
                    print(f"Found solution after {i+1} evaluations")
                break

        wall_time = time.time() - start_time
        success = best_fitness >= 1.0

        return OptimizationResult(
            success=success,
            best_circuit=best_circuit if success else None,
            generations=1,
            evaluations=self.evaluation_count,
            wall_time=wall_time,
        )


class HillClimbingOptimizer:
    """
    Hill climbing with restarts.

    Start from random circuit, mutate if improvement, restart if stuck.
    """

    def __init__(
        self,
        gene_pool: list[str],
        num_genes: int = 3,
        max_edges: int = 9,
        restarts: int = 10,
        steps_per_restart: int = 100,
        seed: int = 42,
    ):
        self.gene_pool = gene_pool
        self.num_genes = num_genes
        self.max_edges = max_edges
        self.restarts = restarts
        self.steps_per_restart = steps_per_restart
        self.rng = random.Random(seed)

        self.classifier = BehaviorClassifier(
            num_initial_conditions=20,
            max_steps=200,
            rule="constitutive"
        )
        self.evaluation_count = 0

    def random_circuit(self) -> Circuit:
        """Generate a random circuit."""
        genes = self.rng.sample(self.gene_pool, min(self.num_genes, len(self.gene_pool)))
        num_edges = self.rng.randint(1, min(self.max_edges, self.num_genes ** 2))

        edges = []
        for _ in range(num_edges):
            src = self.rng.randint(0, len(genes) - 1)
            tgt = self.rng.randint(0, len(genes) - 1)
            edge_type = self.rng.choice([1, -1])
            edges.append((src, tgt, edge_type))

        edges = list(set(edges))
        return Circuit(genes=genes, edges=edges)

    def mutate(self, circuit: Circuit) -> Circuit:
        """Single mutation."""
        child = circuit.copy()

        if not child.edges:
            src = self.rng.randint(0, len(child.genes) - 1)
            tgt = self.rng.randint(0, len(child.genes) - 1)
            child.edges.append((src, tgt, self.rng.choice([1, -1])))
            return child

        mutation_type = self.rng.choice(["add", "remove", "flip", "rewire"])

        if mutation_type == "add" and len(child.edges) < self.max_edges:
            src = self.rng.randint(0, len(child.genes) - 1)
            tgt = self.rng.randint(0, len(child.genes) - 1)
            edge_type = self.rng.choice([1, -1])
            new_edge = (src, tgt, edge_type)
            if new_edge not in child.edges:
                child.edges.append(new_edge)
        elif mutation_type == "remove" and len(child.edges) > 1:
            idx = self.rng.randint(0, len(child.edges) - 1)
            child.edges.pop(idx)
        elif mutation_type == "flip":
            idx = self.rng.randint(0, len(child.edges) - 1)
            src, tgt, etype = child.edges[idx]
            child.edges[idx] = (src, tgt, -etype)
        elif mutation_type == "rewire":
            idx = self.rng.randint(0, len(child.edges) - 1)
            src, tgt, etype = child.edges[idx]
            if self.rng.random() < 0.5:
                src = self.rng.randint(0, len(child.genes) - 1)
            else:
                tgt = self.rng.randint(0, len(child.genes) - 1)
            child.edges[idx] = (src, tgt, etype)

        child.edges = list(set(child.edges))
        return child

    def evaluate(self, circuit: Circuit, target_phenotype: str) -> float:
        """Evaluate circuit fitness."""
        self.evaluation_count += 1
        network = circuit.to_network()
        reward, details = self.classifier.compute_shaped_reward(network, target_phenotype)
        circuit.fitness = reward
        circuit.phenotype = details.get("predicted", "unknown")
        return reward

    def optimize(
        self,
        target_phenotype: str,
        max_evaluations: int = 10000,
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run hill climbing with restarts."""
        start_time = time.time()
        self.evaluation_count = 0

        best_circuit = None
        best_fitness = 0.0
        fitness_history = []

        for restart in range(self.restarts):
            if self.evaluation_count >= max_evaluations:
                break

            current = self.random_circuit()
            self.evaluate(current, target_phenotype)

            if current.fitness > best_fitness:
                best_fitness = current.fitness
                best_circuit = current.copy()

            if current.fitness >= 1.0:
                if verbose:
                    print(f"Found solution at restart {restart}")
                break

            for step in range(self.steps_per_restart):
                if self.evaluation_count >= max_evaluations:
                    break

                candidate = self.mutate(current)
                self.evaluate(candidate, target_phenotype)

                if candidate.fitness >= current.fitness:
                    current = candidate

                if current.fitness > best_fitness:
                    best_fitness = current.fitness
                    best_circuit = current.copy()

                if current.fitness >= 1.0:
                    break

            fitness_history.append(best_fitness)

            if best_fitness >= 1.0:
                break

        wall_time = time.time() - start_time
        success = best_fitness >= 1.0

        return OptimizationResult(
            success=success,
            best_circuit=best_circuit if success else None,
            generations=len(fitness_history),
            evaluations=self.evaluation_count,
            wall_time=wall_time,
            fitness_history=fitness_history,
        )
