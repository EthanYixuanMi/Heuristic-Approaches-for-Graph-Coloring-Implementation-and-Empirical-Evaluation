from typing import List

import numpy as np
import matplotlib.pyplot as plt

from .graph_generators import (
    generate_random_graph,
    generate_mycielski_graph,
    generate_crown_graph,
    generate_adversarial_for_greedy,
    generate_bipartite_like_graph,
)
from .evaluation import run_all_algorithms
from .visualization import (
    visualize_colored_graph,
    plot_comparison_bar,
    plot_time_comparison,
    save_and_show,
)


def experiment_random_graph_comparison() -> None:
    """
    Experiment 1: single random graph, compare all algorithms.
    """
    print("=== Experiment 1: Random Graph Comparison ===")
    G = generate_random_graph(80, 0.35, seed=42)
    results = run_all_algorithms(
        G,
        use_genetic=True,
        genetic_params={"population_size": 60, "generations": 80},
    )

    for name, r in results.items():
        print(
            f"{name:<14} colors={r['num_colors']:<3}  "
            f"time={r['time'] * 1000:>7.2f} ms  valid={r['valid']}"
        )

    alg_order = list(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    # top row: graph visualizations (limit to first 4 algorithms)
    for i, alg in enumerate(alg_order[:4]):
        visualize_colored_graph(
            G,
            results[alg]["colors"],
            title=f"{alg}",
            ax=axes[i],
        )

    # bottom row: bar charts
    plot_comparison_bar(results, ax=axes[4], title="Colors on Random Graph (n=80,p=0.35)")
    plot_time_comparison(results, ax=axes[5], title="Time on Random Graph (n=80,p=0.35)")

    fig.suptitle("Experiment 1: Random Graph Comparison", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp1_random_comparison.png")


def experiment_challenging_graph_structures() -> None:
    """
    Experiment 2: Mycielski, Crown, Adversarial, Bipartite-like.
    """
    print("\n=== Experiment 2: Challenging Graph Structures ===")

    graphs = {
        "Mycielski-5": (generate_mycielski_graph(5), 5),
        "Crown-8": (generate_crown_graph(8), 2),
        "Adversarial-30": (generate_adversarial_for_greedy(30), None),
        "Bipartite-like": (generate_bipartite_like_graph(15, 15, 0.4), None),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    for idx, (name, (G, optimal)) in enumerate(graphs.items()):
        print(f"\n>>> {name}")
        print(f"Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}, Optimal={optimal}")
        results = run_all_algorithms(
            G,
            use_genetic=True,
            genetic_params={"population_size": 40, "generations": 60},
        )

        for alg, r in results.items():
            print(
                f"{alg:<14} colors={r['num_colors']:<3} "
                f"time={r['time'] * 1000:>7.2f} ms  valid={r['valid']}"
            )

        # top row: graph drawing
        visualize_colored_graph(G, results["DSatur"]["colors"], f"{name} (DSatur)", axes[idx])

        # bottom row: color count bar
        ax_bar = axes[idx + 4]
        plot_comparison_bar(results, ax=ax_bar, title=f"{name} - colors")

    fig.suptitle("Experiment 2: Challenging Graph Structures", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp2_challenging_graphs.png")


def experiment_scalability() -> None:
    """
    Experiment 3: Scalability with graph size (n).
    """
    print("\n=== Experiment 3: Scalability Test (vary n) ===")
    sizes = [30, 50, 80, 100, 150, 200]
    alg_names = ["Greedy", "DSatur", "LDF", "Welsh-Powell", "Random Greedy", "Genetic"]

    color_stats = {alg: [] for alg in alg_names}
    time_stats = {alg: [] for alg in alg_names}

    for n in sizes:
        G = generate_random_graph(n, 0.3, seed=42 + n)
        gen_params = {"population_size": max(20, 60 - n // 5), "generations": max(30, 80 - n // 3)}
        results = run_all_algorithms(G, use_genetic=True, genetic_params=gen_params)

        print(f"\n-- n = {n} --")
        for alg in alg_names:
            r = results[alg]
            color_stats[alg].append(r["num_colors"])
            time_stats[alg].append(r["time"] * 1000)
            print(
                f"{alg:<14} colors={r['num_colors']:<3} "
                f"time={r['time'] * 1000:>7.2f} ms  valid={r['valid']}"
            )

    # plot colors vs n, time vs n
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for alg, series in color_stats.items():
        ax1.plot(sizes, series, marker="o", label=alg)
    ax1.set_xlabel("Number of Vertices (n)")
    ax1.set_ylabel("Number of Colors")
    ax1.set_title("Colors vs Graph Size (p=0.3)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    for alg, series in time_stats.items():
        ax2.plot(sizes, series, marker="o", label=alg)
    ax2.set_xlabel("Number of Vertices (n)")
    ax2.set_ylabel("Time (ms, log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Runtime vs Graph Size (p=0.3)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.suptitle("Experiment 3: Scalability", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp3_scalability.png")


def experiment_density() -> None:
    """
    Experiment 4: impact of edge density p.
    """
    print("\n=== Experiment 4: Density Analysis (vary p) ===")
    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n_nodes = 60
    alg_names = ["Greedy", "DSatur", "LDF", "Welsh-Powell", "Random Greedy", "Genetic"]

    color_stats = {alg: [] for alg in alg_names}
    time_stats = {alg: [] for alg in alg_names}

    for p in densities:
        G = generate_random_graph(n_nodes, p, seed=int(p * 1000))
        results = run_all_algorithms(
            G,
            use_genetic=True,
            genetic_params={"population_size": 40, "generations": 50},
        )

        print(f"\n-- p = {p:.2f} --")
        for alg in alg_names:
            r = results[alg]
            color_stats[alg].append(r["num_colors"])
            time_stats[alg].append(r["time"] * 1000)
            print(
                f"{alg:<14} colors={r['num_colors']:<3} "
                f"time={r['time'] * 1000:>7.2f} ms  valid={r['valid']}"
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for alg, series in color_stats.items():
        ax1.plot(densities, series, marker="o", label=alg)
    ax1.set_xlabel("Edge Probability p")
    ax1.set_ylabel("Number of Colors")
    ax1.set_title(f"Colors vs Density (n={n_nodes})")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    for alg, series in time_stats.items():
        ax2.plot(densities, series, marker="o", label=alg)
    ax2.set_xlabel("Edge Probability p")
    ax2.set_ylabel("Time (ms, log scale)")
    ax2.set_yscale("log")
    ax2.set_title(f"Runtime vs Density (n={n_nodes})")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.suptitle("Experiment 4: Density Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp4_density.png")


def experiment_crown_analysis() -> None:
    """
    Experiment 5: Crown graphs (adversarial for greedy).
    """
    print("\n=== Experiment 5: Crown Graph Analysis ===")
    crown_sizes = [4, 6, 8, 10]
    alg_names = ["Greedy", "DSatur", "Genetic"]

    # for plotting
    greedy_colors: List[int] = []
    dsatur_colors: List[int] = []
    genetic_colors: List[int] = []

    print(f"{'n':<5} {'Optimal':<8} {'Greedy':<8} {'DSatur':<8} {'Genetic':<8}")
    for n in crown_sizes:
        G = generate_crown_graph(n)
        results = run_all_algorithms(
            G,
            use_genetic=True,
            genetic_params={"population_size": 50, "generations": 80},
        )
        optimal = 2
        g = results["Greedy"]["num_colors"]
        d = results["DSatur"]["num_colors"]
        ga = results["Genetic"]["num_colors"]

        greedy_colors.append(g)
        dsatur_colors.append(d)
        genetic_colors.append(ga)

        print(f"{n:<5} {optimal:<8} {g:<8} {d:<8} {ga:<8}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(crown_sizes, greedy_colors, marker="o", label="Greedy")
    ax.plot(crown_sizes, dsatur_colors, marker="o", label="DSatur")
    ax.plot(crown_sizes, genetic_colors, marker="o", label="Genetic")
    ax.axhline(2, color="gray", linestyle="--", label="Optimal")

    ax.set_xlabel("n in Crown-n")
    ax.set_ylabel("Number of Colors")
    ax.set_title("Experiment 5: Crown Graphs (Adversarial for Greedy)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    save_and_show(fig, "exp5_crown_analysis.png")


def experiment_statistics(num_trials: int = 30) -> None:
    """
    Experiment 6: Statistical analysis on random graphs.
    """
    print("\n=== Experiment 6: Statistical Analysis ===")
    n_nodes, p_edges = 60, 0.35
    alg_names = ["Greedy", "DSatur", "LDF", "Welsh-Powell", "Random Greedy", "Genetic"]

    stats = {alg: {"colors": [], "times": []} for alg in alg_names}

    for trial in range(num_trials):
        G = generate_random_graph(n_nodes, p_edges, seed=2000 + trial)
        results = run_all_algorithms(
            G,
            use_genetic=True,
            genetic_params={"population_size": 30, "generations": 40},
        )
        for alg in alg_names:
            r = results[alg]
            stats[alg]["colors"].append(r["num_colors"])
            stats[alg]["times"].append(r["time"] * 1000)

    print(f"{'Algorithm':<15} {'Mean':<7} {'Min':<5} {'Max':<5} {'Std':<7} {'Time(ms)':<10}")
    for alg in alg_names:
        colors = np.array(stats[alg]["colors"])
        times = np.array(stats[alg]["times"])
        print(
            f"{alg:<15} {colors.mean():<7.2f} {colors.min():<5d} {colors.max():<5d} "
            f"{colors.std():<7.2f} {times.mean():<10.2f}"
        )

    # boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.boxplot([stats[alg]["colors"] for alg in alg_names], labels=alg_names)
    ax1.set_ylabel("Number of Colors")
    ax1.set_title("Distribution of Color Counts")

    ax2.boxplot([stats[alg]["times"] for alg in alg_names], labels=alg_names)
    ax2.set_yscale("log")
    ax2.set_ylabel("Time (ms, log scale)")
    ax2.set_title("Distribution of Runtimes")

    fig.suptitle("Experiment 6: Statistical Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_and_show(fig, "exp6_statistics.png")


def run_all_experiments() -> None:
    """
    Convenience function to run all experiments in sequence.
    """
    experiment_random_graph_comparison()
    experiment_challenging_graph_structures()
    experiment_scalability()
    experiment_density()
    experiment_crown_analysis()
    experiment_statistics()
    print("\nAll experiments completed. Figures saved to 'img/' folder.")
