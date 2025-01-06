import os
import logging
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from hbmep_paper.utils import setup_logging
from bootstrap__models import (
    HierarchicalBayesianModel,
    DefaultHierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    LeastSquares
)
from constants import (
    BOOTSTRAP_DIR,
    BOOTSTRAP_EXPERIMENTS_DIR,
    BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR
)

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"

TIMES_SEM = 1
FIG_SIZE_CONST = 1.25

ROTATION = 0
TICK_SIZE = 10
AXIS_LABEL_SIZE = 12
INSIDE_TEXT_SIZE = 8
MARKER_SIZE = 2.5
LINE_WIDTH = 1
LINE_STYLE = "--"
GRID_LINE_STYLE = "--"
GRID_ALPHA = .2

MODEL_NAMES_DICT = {
    HierarchicalBayesianModel.NAME: "Hierarchical Bayesian (HB)",
    NonHierarchicalBayesianModel.NAME: "Non-hierarchical Bayesian (nHB)",
    MaximumLikelihoodModel.NAME: "Maximum likelihood (ML)",
    LeastSquares.NAME: "Least squares method (LSM)"
}

COLORS = sns.color_palette("colorblind")
COLORS2 = [
    (128, 128, 128),
    (128,0,128),
    (128, 128, 0),
    (0, 206, 209),
]
COLORS2 = [(r / 255, g / 255, b / 255) for r, g, b in COLORS2]

MODEL_COLORS = {
    HierarchicalBayesianModel.NAME: COLORS2[3],
    NonHierarchicalBayesianModel.NAME: COLORS2[1],
    MaximumLikelihoodModel.NAME: COLORS[8],
    LeastSquares.NAME: COLORS[-3],
    DefaultHierarchicalBayesianModel.NAME: COLORS2[3],
}
MODEL_PLOT_KWARGS = {}
for model_name in MODEL_COLORS:
    MODEL_PLOT_KWARGS[model_name] = {
        "color": MODEL_COLORS[model_name],
        "linestyle": LINE_STYLE,
        "marker": "o",
    }

RCOLOR = "r"
LEVEL_ALPHA = .5
JITTER = .2

MODEL_PLOT_KWARGS[HierarchicalBayesianModel.NAME]["linestyle"] = "-"
BLOCK_SIZE = 100

MODEL_NAMES_DICT = {
    HierarchicalBayesianModel.NAME: "HBe",
    DefaultHierarchicalBayesianModel.NAME: "HB",
    NonHierarchicalBayesianModel.NAME: "nHB",
    MaximumLikelihoodModel.NAME: "ML",
    LeastSquares.NAME: "LSM"
}


def main():
    src = "/home/vishu/repos/hbmep-paper/reports/intraoperative/hierarchical_bayesian_model_20000W_20000S_4C_20T_20D_0.95A/inference.pkl"
    with open(src, "rb") as f:
        model, mcmc, posterior_samples_ = pickle.load(f)

    src = os.path.join(BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR, "results.pkl")
    with open(src, "rb") as f:
        (
            arr,
            reject,
            correct_reject,
            model_names,
            n_subjects_space,
        ) = pickle.load(f)

    logger.info(arr.shape)
    NUM_BLOCKS = arr.shape[0] // BLOCK_SIZE
    arr = arr[:NUM_BLOCKS * BLOCK_SIZE, ...]
    arr = arr.reshape(NUM_BLOCKS, BLOCK_SIZE, *arr.shape[1:])
    logger.info(arr.shape)

    nr, nc = 1, 5
    fig, axes = plt.subplots(
        nr, nc, squeeze=False, constrained_layout=True, figsize=(12, 3.2), sharex=True, sharey="row"
    )

    for muscle_ind in range(model.n_response):
        for model_ind, model_name in enumerate(model_names):
            ax = axes[0, muscle_ind]
            x = [JITTER * (model_ind - len(model_names) // 2) + n for n in n_subjects_space]
            yme = arr.mean(axis=(0, 1))[model_ind, :, muscle_ind]
            yerr = TIMES_SEM * arr.mean(axis=1).std(axis=0, ddof=1)[model_ind, :, muscle_ind]
            ax.errorbar(
                x=x,
                y=yme,
                yerr=yerr,
                # marker="o",
                label=MODEL_NAMES_DICT[model_name],
                # linestyle=LINE_STYLE,
                ms=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                # color=MODEL_COLORS[model_name]
                **MODEL_PLOT_KWARGS[model_name]
            )

            ax = axes[0, -1]
            yme = arr.any(axis=-1).mean(axis=(0, 1))[model_ind, :]
            yerr = TIMES_SEM * arr.any(axis=-1).mean(axis=1).std(axis=0, ddof=1)[model_ind, :]
            ax.errorbar(
                x=x,
                y=yme,
                yerr=yerr,
                # marker="o",
                label=MODEL_NAMES_DICT[model_name],
                # linestyle=LINE_STYLE,
                ms=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                # color=MODEL_COLORS[model_name]
                **MODEL_PLOT_KWARGS[model_name]
            )

    for j in range(nc):
        ax = axes[0, j]
        ax.set_xlabel("")
        ax.set_ylabel("")
        sides = ["top", "right"]
        for side in sides: ax.spines[side].set_visible(False)
        ax.tick_params(
            axis='both',
            which='both',
            left=True,
            bottom=True,
            right=False,
            top=False,
            labelleft=True if not j else False,
            labelbottom=True,
            labelright=False,
            labeltop=False,
            labelrotation=ROTATION,
            labelsize=TICK_SIZE
        )
        ax.grid(axis="y", linestyle=GRID_LINE_STYLE, alpha=GRID_ALPHA)
        ax.axhline(y=.05, linestyle="--", color=RCOLOR, linewidth=LINE_WIDTH, alpha=LEVEL_ALPHA, xmin=0.01, xmax=1)
        if j < 4: ax.set_title(model.response[j], size=AXIS_LABEL_SIZE - 2)
        else: ax.set_title("Overall", size=AXIS_LABEL_SIZE - 2)

    ax = axes[0, 0]
    ax.set_xticks(n_subjects_space)
    ax.set_yticks([.02 * i for i in range(6)] + [.05])
    ax.legend(loc="upper right", fontsize=INSIDE_TEXT_SIZE, labelspacing=.8, reverse=True, ncol=2, frameon=False)
    ax.set_ylabel("False positive rate", fontsize=AXIS_LABEL_SIZE - 2)
    ax.set_xlabel("Number of participants", fontsize=AXIS_LABEL_SIZE - 2)
    ax.text(1.8, .051, "5% significance level", va="bottom", ha="left", fontsize=INSIDE_TEXT_SIZE)

    ax = axes[0, -1]
    ax.set_ylabel("Family-wise error rate", fontsize=AXIS_LABEL_SIZE - 2)

    fig.align_xlabels()
    fig.align_ylabels()

    dest = os.path.join(BOOTSTRAP_DIR, "within_FWER.svg")
    fig.savefig(dest, dpi=600); logger.info(f"Saved to {dest}")
    dest = os.path.join(BOOTSTRAP_DIR, "within_FWER.png")
    fig.savefig(dest, dpi=600); logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=BOOTSTRAP_DIR,
        fname=os.path.basename(__file__)
    )
    main()
