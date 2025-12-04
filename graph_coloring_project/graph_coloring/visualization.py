import os
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ---- Global plot style ----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.dpi"] = 150

COLORS = {
    "text": "#222222",
    "grid": "#DDDDDD",
    "palette": [
        "#4C72B0",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#CCB974",
        "#64B5CD",
    ],
}


def ensure_img_dir() -> str:
    """
    Create 'img' directory if not exists, and return its path.
    """
    out_dir = "img"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def visualize_colored_graph(
    graph: nx.Graph,
    colors: Dict[Any, int],
    title: str,
    ax: plt.Axes,
) -> None:
    """
    Draw a colored graph using spring layout.
    """
    n = graph.number_of_nodes()
    if n == 0:
        ax.set_title(title)
        return

    pos = nx.spring_layout(graph, seed=42, k=2 / np.sqrt(n))
    num_colors = max(colors.values()) + 1 if colors else 1
    cmap = plt.cm.Pastel1(np.linspace(0.0, 0.9, max(num_colors, 9)))
    node_colors = [cmap[colors[node] % 9] for node in graph.nodes()]

    nx.draw(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        with_labels=True,
        node_size=450,
        font_size=8,
        font_weight="bold",
        font_family="Times New Roman",
        edge_color="#CCCCCC",
        width=1.0,
        alpha=0.95,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"])


def plot_comparison_bar(
    results: Dict[str, dict],
    ax: plt.Axes,
    title: str = "Number of Colors Used",
) -> None:
    """
    Bar chart comparing number of colors among algorithms.
    """
    algorithms: List[str] = list(results.keys())
    num_colors = [results[alg]["num_colors"] for alg in algorithms]
    x = np.arange(len(algorithms))

    bars = ax.bar(
        x,
        num_colors,
        color=COLORS["palette"],
        edgecolor="#666666",
        linewidth=1.2,
    )
    for rect, value in zip(bars, num_colors):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.1,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=20)
    ax.set_ylabel("Number of Colors")
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def plot_time_comparison(
    results: Dict[str, dict],
    ax: plt.Axes,
    title: str = "Execution Time (ms)",
) -> None:
    """
    Bar chart comparing runtime among algorithms (ms).
    Uses log-scale if spread is huge.
    """
    algorithms: List[str] = list(results.keys())
    times_ms = [results[alg]["time"] * 1000.0 for alg in algorithms]
    x = np.arange(len(algorithms))

    bars = ax.bar(
        x,
        times_ms,
        color=COLORS["palette"],
        edgecolor="#666666",
        linewidth=1.2,
        width=0.6,
    )
    for rect, value in zip(bars, times_ms):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() * 1.01 + 1e-6,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )

    if max(times_ms) / (min(times_ms) + 1e-9) > 100:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=20)
    ax.set_ylabel("Time (ms)")
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"])
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def save_and_show(fig: plt.Figure, filename: str) -> None:
    """
    Save figure into img/ directory and show it.
    """
    out_dir = ensure_img_dir()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"[INFO] Figure saved to {path}")
