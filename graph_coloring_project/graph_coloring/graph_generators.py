import random
import networkx as nx


def generate_random_graph(n: int, p: float, seed: int | None = None) -> nx.Graph:
    """
    Erdős–Rényi G(n, p) random graph.
    """
    return nx.erdos_renyi_graph(n, p, seed=seed)


def generate_crown_graph(n: int) -> nx.Graph:
    """
    Crown graph on 2n vertices.
    Two partitions A={a0..a_{n-1}}, B={b0..b_{n-1}},
    edges between ai and bj if i != j.
    """
    G = nx.Graph()
    for i in range(n):
        G.add_node(f"a{i}")
        G.add_node(f"b{i}")

    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(f"a{i}", f"b{j}")

    return G


def generate_mycielski_graph(k: int) -> nx.Graph:
    """
    Mycielski graph of size k (networkx built-in).
    """
    return nx.mycielski_graph(k)


def generate_adversarial_for_greedy(n: int) -> nx.Graph:
    """
    Adversarial graph for simple greedy coloring.
    Split vertices into two halves, with different densities.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    half = n // 2

    # dense connections in the first half
    for i in range(half):
        for j in range(i + 1, half):
            if random.random() < 0.6:
                G.add_edge(i, j)

    # dense-ish connections between halves
    for i in range(half):
        for j in range(half, n):
            if random.random() < 0.7:
                G.add_edge(i, j)

    # sparse connections in the second half
    for i in range(half, n):
        for j in range(i + 1, n):
            if random.random() < 0.2:
                G.add_edge(i, j)

    return G


def generate_bipartite_like_graph(n_left: int, n_right: int, p: float = 0.5) -> nx.Graph:
    """
    Almost-bipartite random graph:
    - Start from random bipartite between two partitions.
    - Add a few edges inside each partition.
    """
    G = nx.Graph()
    left = [f"L{i}" for i in range(n_left)]
    right = [f"R{i}" for i in range(n_right)]
    G.add_nodes_from(left + right)

    # bipartite edges
    for u in left:
        for v in right:
            if random.random() < p:
                G.add_edge(u, v)

    # small noise inside each side
    for i in range(n_left):
        for j in range(i + 1, n_left):
            if random.random() < 0.1:
                G.add_edge(left[i], left[j])

    for i in range(n_right):
        for j in range(i + 1, n_right):
            if random.random() < 0.1:
                G.add_edge(right[i], right[j])

    return G
