import random
import numpy as np

from graph_coloring.experiments import run_all_experiments


def main() -> None:
    # fix seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    run_all_experiments()


if __name__ == "__main__":
    main()
