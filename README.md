# Improved-GA-for-Graph-Coloring  
A collection of classical heuristics and an improved Genetic Algorithm for solving the graph coloring problem, with full experimental evaluation and visualization.  


## Project Overview

This project implements six graph coloring algorithms — ranging from simple greedy heuristics
to a customized, improved Genetic Algorithm (GA). The goal is to show how heuristic design
strongly affects solution quality, runtime, and robustness across different graph structures.

All experiments are fully reproducible, and all figures are generated automatically.  


## Key Contributions

### 1. A suite of 6 graph coloring algorithms
- Greedy  
- Largest Degree First (LDF)  
- Welsh–Powell  
- DSatur  
- Random Greedy (multi-start)  
- **Improved Genetic Algorithm (main innovation)**  

### 2. Improved GA tailored for graph coloring
Our GA is not a standard textbook GA.  
It includes several **graph-coloring–specific enhancements**:

- **Greedy-based color upper bound** to reduce search space  
- **Lexicographic fitness** prioritizing conflict elimination  
- **Repair operator** to fix invalid individuals  
- **Guided mutation** selecting colors that avoid neighbor conflicts  
- **Tournament selection + elitism** for stable convergence  

### 3. Five categories of input graphs
Random graphs, Mycielski graphs, Crown graphs, adversarial graphs, bipartite-like graphs.

### 4. Six comprehensive experiments with visualization
All figures saved under `img/`.


## Project Structure
```
graph_coloring_project/
│
├─ main.py                      # Entry point: runs all experiments
├─ requirements.txt             # Python dependencies
│
└─ graph_coloring/
   ├─ __init__.py
   ├─ algorithms.py             # Implementations of all 6 algorithms
   ├─ graph_generators.py       # Random, crown, mycielski, adversarial...
   ├─ evaluation.py             # Unified algorithm runner + validity check
   ├─ visualization.py          # Plot functions (bar charts, graph layouts)
   └─ experiments.py            # All six experiments (exp1–exp6)
│
└─ img/                         # Auto-created: all experiment figures saved here
```  


## Experimental Setup

We evaluate all algorithms across six experiments:

1. Random graph comparison  
2. Challenging graph structures  
3. Scalability with graph size  
4. Density analysis  
5. Crown graph adversarial test  
6. Statistical analysis over 30 trials  

All figures are automatically generated and saved in the `img/` directory.


## Experiment Results & Figures  

### Experiment 1 — Random Graph Comparison

This experiment compares all six algorithms on a random G(n,p) graph with n=80, p=0.35.
We observe:

- Greedy uses the most colors (14)
- DSatur performs best with 11 colors
- Our Genetic Algorithm achieves 12 colors
- Runtime difference is dramatic: GA is the slowest, greedy the fastest

#### Figure: Algorithm Outputs + Performance Comparison  
![Experiment 1](graph_coloring_project/img/exp1_random_comparison.png)  

### Experiment 2 — Challenging Graph Structures

We evaluate on four structured graphs:
- Mycielski-5 (chromatic number = 5)
- Crown-8 (adversarial for greedy)
- Adversarial-30 graph
- Bipartite-like noisy graph

Observations:
- All algorithms reach optimal 5 colors on Mycielski-5
- Greedy completely fails on Crown-8 (uses 8 colors instead of 2)
- GA and DSatur both achieve optimal solutions for Crown graphs

#### Figure: Algorithm behavior on structured graphs (Mycielski, Crown, Adversarial, Bipartite-like)  
![Experiment 2](graph_coloring_project/img/exp2_challenging_graphs.png)

### Experiment 3 — Scalability

Tested n ∈ {30, 50, 80, 100, 150, 200} on random graphs with p=0.3.

Results:
- DSatur consistently uses fewer colors as n grows
- GA performance is good but computationally expensive
- Greedy is fastest but quality degrades with graph size

#### Figure: Scalability of color usage and runtime as graph size increases   
![Experiment 3](graph_coloring_project/img/exp3_scalability.png)

### Experiment 4 — Density Analysis

We vary the density p from 0.1 to 0.7.
As density increases:
- All algorithms require more colors
- DSatur remains the most robust
- GA stays competitive but slow

#### Figure: Impact of edge density on color requirements and runtime   
![Experiment 4](graph_coloring_project/img/exp4_density.png)

### Experiment 5 — Crown Graph Analysis (Adversarial for Greedy)

Crown graphs are bipartite and 2-colorable, but greedy performs extremely poorly:
- Greedy uses n colors on Crown-n
- DSatur and GA always find optimal 2 colors

This experiment highlights the importance of using intelligent heuristics.

#### Figure: Performance on Crown graphs (worst-case for greedy)  
![Experiment 5](graph_coloring_project/img/exp5_crown_analysis.png)

### Experiment 6 — Statistical Analysis (30 Trials)

We run 30 independent trials on random graphs (n=60, p=0.35).

Results:
- DSatur shows the most stable behavior (lowest standard deviation)
- GA performs well but with high runtime variance
- Greedy fluctuates but remains fast

#### Figure: Distribution of color counts and runtimes over 30 trials  
![Experiment 6](graph_coloring_project/img/exp6_statistics.png)



