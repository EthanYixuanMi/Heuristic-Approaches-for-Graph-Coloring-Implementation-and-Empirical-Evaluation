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

## Experiment 1 — Random Graph Comparison

This experiment compares all six algorithms on a random G(n,p) graph with n=80, p=0.35.
We observe:

- Greedy uses the most colors (14)
- DSatur performs best with 11 colors
- Our Genetic Algorithm achieves 12 colors
- Runtime difference is dramatic: GA is the slowest, greedy the fastest

### Figure: Algorithm Outputs + Performance Comparison
![Experiment 1](graph_coloring_project/img/exp1_random_comparison.png)  




