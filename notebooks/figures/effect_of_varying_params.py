import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from hbmep.nn import functional as F
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from constants import BUILD_DIR

logger = logging.getLogger(__name__)

colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.3, 1, 3))
lineplot_kwargs = {
    "linewidth": 1
}


def main():
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7, 1.7 * nrows),
        constrained_layout=True, squeeze=False
    )
    x = np.linspace(0, 10, 1000)
    named_params = [site.a, site.L, site.H, site.b, site.ell]
    check_values = [(1, 2, 3), (.5, .8, 1), (2, 3, 4), (.25, .5, 1), (.01, .1, 1)]
    yticks = [[1, 2], [.5, .8, 1, 1.5, 1.8, 2], [1, 3, 4, 5], [1, 2], [1, 2]]
    default_values = {named_param: 1 for named_param in named_params}
    for i, named_param in enumerate(named_params):
        ax = axes[i // ncols, i % ncols]
        for j, value in enumerate(check_values[i]):
            param = default_values.copy()
            param[named_param] = value
            y = F.rectified_logistic(
                x, param[site.a], param[site.b], param[site.L], param[site.ell], param[site.H]
            )
            sns.lineplot(x=x, y=y, color=colors[j], ax=ax, label=f"{named_param}={value}", **lineplot_kwargs)
        ax.set_xticks([0, 1, 10])
        ax.set_yticks(yticks[i])
        ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 2]
    values = [(.3, .05), (.4, .05), (.5, .05)]
    for i, (b, ell) in enumerate(values):
        y = F.rectified_logistic(x, 1, b, 1, ell, 1)
        sns.lineplot(x=x, y=y, color=colors[i], ax=ax, label=f"{site.b}={b}, {site.ell}={ell}", **lineplot_kwargs)
    ax.set_xticks([0, 1, 10])
    ax.set_yticks([1, 2])
    ax.legend(loc="upper left", fontsize=8)
    # if ax.get_legend(): ax.get_legend().remove()

    ax = axes[0, 0]
    ax.set_xticks([0, 1, 2, 3, 10])
    ax.text(-.4, 1.8, "$\mathcal{F}(x)$", fontsize=10)
    ax.text(5, .8, "$x$")

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            sides = ["top", "right"]
            for side in sides:
                ax.spines[side].set_visible(False)
            ax.tick_params(
                axis='both',
                which='both',
                left=True,
                bottom=True,
                right=False,
                top=False,
                labelleft=True,
                labelbottom=True,
                labelright=False,
                labeltop=False,
                labelrotation=15,
                labelsize=8
            )

    fig.suptitle("All other parameters are 1", fontsize=10)
    dest = os.path.join(BUILD_DIR, "effect_of_varying_params.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    os.makedirs(BUILD_DIR, exist_ok=True)
    setup_logging(BUILD_DIR, os.path.basename(__file__))
    main()
