import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    src = "/home/vishu/repos/hbmep-paper/reports/experiments/bo/learn_posterior_iterative/times.npy"
    times = np.load(src)

    sns.histplot(times, binwidth=5)
    plt.savefig("times.png", dpi=300, bbox_inches="tight")

    src = "/home/vishu/repos/hbmep-paper/reports/experiments/bo/learn_posterior_iterative/loo_scores.npy"
    loo_scores = np.load(src)
    print(loo_scores.sum())

    return


if __name__ == "__main__":
    main()