import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep.nn import functional as F


def main():
    params = (4, 1, 1, 1, 10, 1)
    L = params[3]
    H = params[-1]
    x = np.linspace(0, 10, 1000)
    y = F.rectified_logistic(x, *params)
    y = np.array(y)
    s50 = F.rectified_logistic_s50(*params)
    s50 = np.array(s50)
    print(type(s50))
    nrows, ncols = 1, 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5),
        constrained_layout=True, squeeze=False
    )
    ax = axes[0, 0]
    sns.lineplot(x=x.reshape(-1,), y=y.reshape(-1,), ax=ax)
    sns.scatterplot(x=[s50.item()], y=[L + (H / 2)], s=20, color="red", zorder=10)
    ax.axhline(L + (H / 2), alpha=.4)
    dest = "/home/vishu/testing/s50.png"
    fig.savefig(dest)
    print(f"Saved to {dest}")


if __name__ == "__main__":
    main()