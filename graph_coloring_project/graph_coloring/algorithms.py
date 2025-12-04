import time
import random
from typing import Dict, Tuple, Any

import networkx as nx


ColoringResult = Tuple[Dict[Any, int], int, float]  # (colors, num_colors, time_seconds)


def greedy_coloring(graph: nx.Graph) -> ColoringResult:
    """
    Basic greedy coloring - by node iteration order.
    """
    start_time = time.perf_counter()
    colors: Dict[Any, int] = {}

    for node in graph.nodes():
        neighbor_colors = {colors[n] for n in graph.neighbors(node) if n in colors}
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color

    end_time = time.perf_counter()
    num_colors = len(set(colors.values()))
    return colors, num_colors, end_time - start_time


def dsatur_coloring(graph: nx.Graph) -> ColoringResult:
    """
    DSatur algorithm - saturation degree heuristic.
    """
    start_time = time.perf_counter()
    colors: Dict[Any, int] = {}
    saturation = {node: 0 for node in graph.nodes()}
    degree = {node: graph.degree(node) for node in graph.nodes()}
    uncolored = set(graph.nodes())

    while uncolored:
        # choose node with max saturation; tie-breaker: max degree
        selected = max(uncolored, key=lambda n: (saturation[n], degree[n]))
        neighbor_colors = {colors[n] for n in graph.neighbors(selected) if n in colors}
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[selected] = color
        uncolored.remove(selected)

        # update saturation of neighbors
        for neighbor in graph.neighbors(selected):
            if neighbor in uncolored:
                neighbor_neighbor_colors = {colors[nn] for nn in graph.neighbors(neighbor) if nn in colors}
                saturation[neighbor] = len(neighbor_neighbor_colors)

    end_time = time.perf_counter()
    num_colors = len(set(colors.values()))
    return colors, num_colors, end_time - start_time


def largest_degree_first(graph: nx.Graph) -> ColoringResult:
    """
    Largest Degree First greedy algorithm.
    """
    start_time = time.perf_counter()
    colors: Dict[Any, int] = {}

    nodes_by_degree = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
    for node in nodes_by_degree:
        neighbor_colors = {colors[n] for n in graph.neighbors(node) if n in colors}
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color

    end_time = time.perf_counter()
    num_colors = len(set(colors.values()))
    return colors, num_colors, end_time - start_time


def welsh_powell_coloring(graph: nx.Graph) -> ColoringResult:
    """
    Welsh–Powell graph coloring algorithm.
    """
    start_time = time.perf_counter()
    colors: Dict[Any, int] = {}

    sorted_nodes = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
    uncolored = set(sorted_nodes)
    current_color = 0

    while uncolored:
        available = set(uncolored)
        for node in sorted_nodes:
            if node in available:
                colors[node] = current_color
                uncolored.remove(node)
                available.remove(node)
                for neighbor in graph.neighbors(node):
                    if neighbor in available:
                        available.remove(neighbor)
        current_color += 1

    end_time = time.perf_counter()
    num_colors = len(set(colors.values()))
    return colors, num_colors, end_time - start_time


def random_greedy_coloring(graph: nx.Graph, iterations: int = 20) -> ColoringResult:
    """
    Randomized greedy: shuffle node ordering multiple times, keep the best.
    """
    start_time = time.perf_counter()
    nodes = list(graph.nodes())
    best_colors: Dict[Any, int] = {}
    best_num_colors = float("inf")

    for _ in range(iterations):
        random.shuffle(nodes)
        colors: Dict[Any, int] = {}
        for node in nodes:
            neighbor_colors = {colors[n] for n in graph.neighbors(node) if n in colors}
            color = 0
            while color in neighbor_colors:
                color += 1
            colors[node] = color

        num_colors = len(set(colors.values()))
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_colors = colors.copy()

    end_time = time.perf_counter()
    return best_colors, best_num_colors, end_time - start_time


def genetic_algorithm_coloring(
    graph: nx.Graph,
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    elite_ratio: float = 0.1,
) -> ColoringResult:
    """
    Genetic Algorithm for graph coloring.

    Representation:
        - Individual: list[int], len = number of nodes, each gene is a color index.
    Fitness:
        - First minimize number of conflicts, then number of distinct colors.
    """

    start_time = time.perf_counter()
    nodes = list(graph.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj_list = [[node_to_idx[neighbor] for neighbor in graph.neighbors(node)] for node in nodes]

    # use greedy as an upper bound for number of colors
    _, greedy_num_colors, _ = greedy_coloring(graph)
    max_colors = greedy_num_colors + 2

    def create_individual() -> list[int]:
        return [random.randint(0, max_colors - 1) for _ in range(n)]

    def fitness(individual: list[int]) -> tuple[int, int]:
        conflicts = 0
        for i in range(n):
            for j in adj_list[i]:
                if i < j and individual[i] == individual[j]:
                    conflicts += 1
        colors_used = len(set(individual))
        return conflicts, colors_used

    def repair(individual: list[int]) -> list[int]:
        """
        Fix conflicts node-by-node if a node has same color as a neighbor.
        """
        ind = individual.copy()
        for i in range(n):
            neighbor_colors = {ind[j] for j in adj_list[i]}
            if ind[i] in neighbor_colors:
                # find a color not used by neighbors (allow expansion if needed)
                c = 0
                while c in neighbor_colors:
                    c += 1
                ind[i] = c
        return ind

    def crossover(p1: list[int], p2: list[int]) -> tuple[list[int], list[int]]:
        if n == 1:
            return p1[:], p2[:]
        point = random.randint(1, n - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    def mutate(individual: list[int]) -> list[int]:
        ind = individual.copy()
        for i in range(n):
            if random.random() < mutation_rate:
                neighbor_colors = {ind[j] for j in adj_list[i]}
                # try to find a color not used by neighbors
                for c in range(max_colors):
                    if c not in neighbor_colors:
                        ind[i] = c
                        break
        return ind

    def select(population: list[list[int]], fitnesses: list[tuple[int, int]]) -> list[list[int]]:
        """
        Tournament selection (k=3).
        """
        selected: list[list[int]] = []
        pop_fit = list(zip(population, fitnesses))
        for _ in range(2):
            candidates = random.sample(pop_fit, min(3, len(pop_fit)))
            winner = min(candidates, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    # initial population
    population = [repair(create_individual()) for _ in range(population_size)]
    elite_size = max(1, int(population_size * elite_ratio))

    best_individual: list[int] | None = None
    best_fitness = (float("inf"), float("inf"))

    for _ in range(generations):
        fitnesses = [fitness(ind) for ind in population]

        for ind, fit in zip(population, fitnesses):
            if fit < best_fitness:
                best_fitness = fit
                best_individual = ind.copy()

        # elitism
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1])
        elites = [ind.copy() for ind, _ in sorted_pop[:elite_size]]
        new_population: list[list[int]] = elites.copy()

        # offspring
        while len(new_population) < population_size:
            parents = select(population, fitnesses)
            c1, c2 = crossover(parents[0], parents[1])
            new_population.append(repair(mutate(c1)))
            if len(new_population) < population_size:
                new_population.append(repair(mutate(c2)))

        population = new_population

    if best_individual is None:
        # fallback – should not really happen
        best_individual = population[0]

    best_individual = repair(best_individual)

    # compress colors to 0..k-1
    unique_colors = sorted(set(best_individual))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    best_individual = [color_map[c] for c in best_individual]

    colors: Dict[Any, int] = {nodes[i]: best_individual[i] for i in range(n)}
    end_time = time.perf_counter()
    num_colors = len(set(colors.values()))
    return colors, num_colors, end_time - start_time
