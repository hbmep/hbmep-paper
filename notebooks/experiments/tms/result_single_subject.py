import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import jax

from hbmep.model.utils import Site as site
from models import Simulator, HBModel, NHBModel

from simulate_data import TOTAL_SUBJECTS
from subjects_exp import fix_rng
from subjects_exp import EXPERIMENT_NAME, N_DRAWS, N_REPEATS

plt.rcParams['svg.fonttype'] = 'none'

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
dest = "/home/vishu/logs/subjects-experiment-results.log"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(dest, mode="w"),
        logging.StreamHandler()
    ],
    force=True
)


def main():
    """ Load simulated data """
    dir ="/home/vishu/repos/hbmep-paper/reports/experiments/subjects/simulate-data/a_random_mean_-2.5_a_random_scale_1.5"
    src = os.path.join(dir, "simulation_ppd.pkl")
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    ppd_a = simulation_ppd[site.a]

    src = os.path.join(dir, "filter.npy")
    filter = np.load(src)

    """ Fix rng """
    rng_key = simulator.rng_key
    max_draws = ppd_a.shape[0]
    max_seeds = N_REPEATS * 100
    draws_space, seeds_for_generating_subjects = fix_rng(
        rng_key, 4000, max_seeds
    )

    """ Results """
    n_subjects_space = [1, 4, 8, 12, 16]
    models = [HBModel, NHBModel]
    # n_subjects_space = [1, 4, 8, 16]
    # models = [HBModel]

    # j = 0
    # n_subjects_space = n_subjects_space[:2]
    draws_space = draws_space[:5]
    # seeds_for_generating_subjects = seeds_for_generating_subjects[:1]
    # draws_space = draws_space[10:12]
    # seeds_for_generating_subjects = seeds_for_generating_subjects[:5]
    logger.info(draws_space)
    logger.info(seeds_for_generating_subjects)

    mae = []
    mse = []
    prob = []
    for n_subjects in n_subjects_space:
        for draw in draws_space:
            for seed in seeds_for_generating_subjects:
                valid_subjects = \
                    np.arange(0, ppd_a.shape[1], 1)[filter[draw, ...]]
                subjects = \
                    jax.random.choice(
                        key=jax.random.PRNGKey(seed),
                        a=valid_subjects,
                        shape=(n_subjects,),
                        replace=False
                    ) \
                    .tolist()
                common_subject = subjects[0]
                argsort = np.array(subjects).argsort().argsort()
                common_subject = argsort[0]
                for m in models:
                    n_subjects_dir, draw_dir, seed_dir = f"n{n_subjects}", f"d{draw}", f"s{seed}"
                    dir = os.path.join(simulator.build_dir, EXPERIMENT_NAME, draw_dir, n_subjects_dir, seed_dir, m.NAME)
                    # dir = "/home/vishu/repos/hbmep-paper/reports/experiments/subjects/simulate-data/a_random_mean_-2.5_a_random_scale_1.5/archived/subjects"
                    # dir = os.path.join(dir, m.NAME, draw_dir, n_subjects_dir, seed_dir)
                    a_true = np.load(os.path.join(dir, "a_true.npy"))
                    a_pred = np.load(os.path.join(dir, "a_pred.npy"))

                    # logger.info(f"a_true before: {a_true.shape}")
                    # logger.info(f"a_pred before: {a_pred.shape}")

                    # Take the common subject only
                    a_true = a_true[common_subject, ...]
                    a_pred = a_pred[:, common_subject, ...]

                    # logger.info(f"a_true: {a_true.shape}")
                    # logger.info(f"a_pred: {a_pred.shape}")

                    a_pred = a_pred.mean(axis=0).reshape(-1,)
                    a_true = a_true.reshape(-1,)

                    curr_mae = np.abs(a_true - a_pred).mean()
                    curr_mse = np.square(a_true - a_pred).mean()
                    mae.append(curr_mae)
                    mse.append(curr_mse)

                    if m.NAME == "hbm":
                        a_random_mean = np.load(os.path.join(dir, "a_random_mean.npy")).reshape(-1,)
                        # a_random_scale = np.load(os.path.join(dir, "a_random_scale.npy")).reshape(-1,)
                        curr_prob = (a_random_mean < 0).mean()
                        prob.append(curr_prob)
                    else:
                        prob.append(0)

    mae = np.array(mae).reshape(len(n_subjects_space), len(draws_space), len(seeds_for_generating_subjects), len(models))
    mse = np.array(mse).reshape(len(n_subjects_space), len(draws_space), len(seeds_for_generating_subjects), len(models))
    prob = np.array(prob).reshape(len(n_subjects_space), len(draws_space), len(seeds_for_generating_subjects), len(models))

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False, constrained_layout=True, sharex="col", sharey=True)
    colors = ["blue", "orange"]
    for model_ind, model in enumerate(models):
        ax = axes[model_ind, 0]
        sns.kdeplot(mae[..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
        ax = axes[-1, 0]
        sns.kdeplot(mae[..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
    axes[0, 0].legend(loc="upper right")
    for model_ind, model in enumerate(models):
        ax = axes[model_ind, 1]
        sns.kdeplot(mae[:-1, ..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
        ax = axes[-1, 1]
        sns.kdeplot(mae[:-1, ..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
    for model_ind, model in enumerate(models):
        ax = axes[model_ind, 2]
        sns.kdeplot(mae[-1, ..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
        ax = axes[-1, 2]
        sns.kdeplot(mae[-1, ..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
    axes[0, 0].set_title(f"n = [1, 4, 8, 16]")
    axes[0, 1].set_title(f"n = [1, 4, 8]")
    axes[0, 2].set_title(f"n = [16]")
    axes[0, 0].legend(loc="upper right")
    axes[1, 0].legend(loc="upper right")
    for i in range(nrows):
        for j in range(nrows):
            ax = axes[i, j]
            ax.tick_params(axis="both", bottom=True, labelbottom=True)
    dest = os.path.join(simulator.build_dir, "error-dist-single-subject.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False, constrained_layout=True)
    colors = ["blue", "orange"]
    for model_ind, model in enumerate(models):
        ax = axes[model_ind, 0]
        sns.boxplot(mae[..., model_ind].reshape(-1,), ax=ax, color=colors[model_ind])
        # ax = axes[-1, 0]
        # sns.kdeplot(mae[..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
    axes[0, 0].legend(loc="upper right")
    for model_ind, model in enumerate(models):
        ax = axes[model_ind, 1]
        sns.boxplot(mae[:-1, ..., model_ind].reshape(-1,), ax=ax, color=colors[model_ind])
        # ax = axes[-1, 1]
        # sns.kdeplot(mae[:-1, ..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
    for model_ind, model in enumerate(models):
        ax = axes[model_ind, 2]
        sns.boxplot(mae[-1, ..., model_ind].reshape(-1,), ax=ax, color=colors[model_ind])
        # ax = axes[-1, 2]
        # sns.kdeplot(mae[-1, ..., model_ind].reshape(-1,), ax=ax, label=f"{model.NAME}", color=colors[model_ind])
    axes[0, 0].set_title(f"n = [1, 4, 8, 16]")
    axes[0, 1].set_title(f"n = [1, 4, 8]")
    axes[0, 2].set_title(f"n = [16]")
    axes[0, 0].legend(loc="upper right")
    axes[1, 0].legend(loc="upper right")
    # for i in range(nrows):
    #     for j in range(nrows):
    #         ax = axes[i, j]
    #         ax.tick_params(axis="both", bottom=True, labelbottom=True)
    dest = os.path.join(simulator.build_dir, "error-dist-single-subject-box.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")
    logger.info(f"PROB: {prob.shape}")

    mae = mae.reshape(mae.shape[0], -1, len(models))
    mse = mse.reshape(mse.shape[0], -1, len(models))
    # medae = np.median(mae, axis=-2)
    # mae = mae.mean(axis=-2)
    # mse = mse.mean(axis=-2)
    # logger.info(f"MEDAE: {medae.shape}")
    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")

    prob = prob > .95
    prob = prob.mean(axis=-2)
    # prob = prob.reshape(prob.shape[0], -1, len(models))
    logger.info(f"PROB: {prob.shape}")
    # prob = (prob > .95)
    # prob = prob.mean(axis=-2)

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False, constrained_layout=True)

    ax = axes[0, 0]
    for model_ind, model in enumerate(models):
        x = n_subjects_space
        y = mae[..., model_ind]
        # y = medae[..., model_ind]
        logger.info(f"MAE: {y.shape}")
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(x=x, y=yme, yerr=ysem, marker="o", label=f"{model.NAME}", linestyle="--", ms=4)
        ax.set_xticks(x)
        ax.legend(loc="upper right")

    ax = axes[0, 1]
    for model_ind, model in enumerate(models):
        x = n_subjects_space
        y = prob[..., model_ind]
        logger.info(f"PROB: {y.shape}")
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ysd = y.std(axis=-1)
        ax.errorbar(x=x, y=yme, yerr=ysd, marker="o", label=f"{model.NAME}", linestyle="--", ms=4)
        ax.set_xticks(x)
        ax.legend(loc="upper right")

    fig.align_xlabels()
    fig.align_ylabels()
    dest = os.path.join(simulator.build_dir, "single-single-subject.svg")
    fig.savefig(dest, dpi=600)
    dest = os.path.join(simulator.build_dir, "subjects-single-subject.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()
