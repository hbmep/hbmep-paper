import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    dir ="/home/vishu/repos/hbmep-paper/reports/experiments/subjects/simulate-data/a_random_mean_-2.5_a_random_scale_1.5/tms_200"
    src = os.path.join(dir, "simulation_ppd.pkl")
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    ppd_a = simulation_ppd[site.a]

    """ Fix rng """
    rng_key = simulator.rng_key
    max_draws = ppd_a.shape[0]
    max_seeds = N_REPEATS * 100
    draws_space, seeds_for_generating_subjects = fix_rng(
        rng_key, max_draws, max_seeds
    )

    """ Results """
    n_subjects_space = [1, 4, 8, 16]
    models = [HBModel, NHBModel]
    # models = [HBModel]

    # j = 0
    # n_subjects_space = n_subjects_space[:2]
    draws_space = draws_space[:17]
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
                for m in models:
                    n_subjects_dir, draw_dir, seed_dir = f"n{n_subjects}", f"d{draw}", f"s{seed}"
                    dir = os.path.join(simulator.build_dir, "tms_200", EXPERIMENT_NAME, draw_dir, n_subjects_dir, seed_dir, m.NAME)
                    # dir = "/home/vishu/repos/hbmep-paper/reports/experiments/subjects/simulate-data/a_random_mean_-2.5_a_random_scale_1.5/archived/subjects"
                    # dir = os.path.join(dir, m.NAME, draw_dir, n_subjects_dir, seed_dir)
                    a_true = np.load(os.path.join(dir, "a_true.npy"))
                    a_pred = np.load(os.path.join(dir, "a_pred.npy"))

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

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")
    logger.info(f"PROB: {prob.shape}")

    # mae = mae.reshape(mae.shape[0], -1, len(models))
    # mse = mse.reshape(mse.shape[0], -1, len(models))
    mae = mae.mean(axis=-2)
    mse = mse.mean(axis=-2)
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
    dest = os.path.join(simulator.build_dir, "tms_200", "subjects-exp.svg")
    fig.savefig(dest, dpi=600)
    dest = os.path.join(simulator.build_dir, "tms_200", "subjects-exp.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()