import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import jax

from hbmep.model.utils import Site as site

from models import Simulator, HBModel, NHBModel
from core_subjects import EXPERIMENT_NAME, N_PULSES, N_REPS
from utils import fix_draws_and_seeds
from constants import TOTAL_SUBJECTS, N_DRAWS, N_SEEDS
from hbmep_paper.utils import setup_logging

plt.rcParams['svg.fonttype'] = 'none'
logger = logging.getLogger(__name__)

SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_-3.0_a_random_scale_1.5"
SIMULATION_PARAMS_PATH = os.path.join(SIMULATION_DIR, "simulation_params.pkl")
MASK_PATH = os.path.join(SIMULATION_DIR, "mask.npy")
EXPERIMENTS_DIR = os.path.join(SIMULATION_DIR, "experiments")


def main():
    """ Load simulated data """
    src = SIMULATION_PARAMS_PATH
    with open(src, "rb") as g:
        simulator, simulation_params = pickle.load(g)

    setup_logging(
        dir=simulator.build_dir,
        fname=os.path.basename(__file__)
    )

    ppd_a = simulation_params[site.a]

    src = MASK_PATH
    mask = np.load(src)

    """ Fix rng """
    rng_key = simulator.rng_key
    max_draws = ppd_a.shape[0]
    max_seeds = N_SEEDS * 100
    draws_space, seeds_for_generating_subjects = fix_draws_and_seeds(
        rng_key, max_draws, max_seeds
    )
    n_subjects_space = [1, 4, 8, 16]

    draws_space = draws_space[:19]
    models = [HBModel, NHBModel]
    # models = [HBModel]

    n_reps, n_pulses = N_REPS, N_PULSES

    """ Results """
    mae = []
    mse = []
    prob = []
    for n_subjects in n_subjects_space:
        for draw in draws_space:
            for seed in seeds_for_generating_subjects:
                for M in models:
                    n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
                    draw_dir, seed_dir = f"d{draw}", f"s{seed}"

                    if M.NAME in ["hbm"]:
                        dir = os.path.join(
                            EXPERIMENTS_DIR,
                            EXPERIMENT_NAME,
                            draw_dir,
                            n_subjects_dir,
                            n_reps_dir,
                            n_pulses_dir,
                            seed_dir,
                            M.NAME
                        )
                        a_true = np.load(os.path.join(dir, "a_true.npy"))
                        a_pred = np.load(os.path.join(dir, "a_pred.npy"))

                        a_pred = a_pred.mean(axis=0).reshape(-1,)
                        a_true = a_true.reshape(-1,)

                        a_random_mean = np.load(os.path.join(dir, "a_random_mean.npy")).reshape(-1,)
                        curr_prob = (a_random_mean < 0).mean()

                    elif M.NAME in ["nhbm"]:
                        n_subjects_dir = f"n{n_subjects_space[-1]}"
                        valid_subjects = \
                            np.arange(0, ppd_a.shape[1], 1)[mask[draw, ...]]
                        subjects = \
                            jax.random.choice(
                                key=jax.random.PRNGKey(seed),
                                a=valid_subjects,
                                shape=(n_subjects,),
                                replace=False
                            ) \
                            .tolist()

                        a_true, a_pred, differences = [], [], []
                        for subject in subjects:
                            sub_dir = f"subject{subject}"
                            dir = os.path.join(
                                EXPERIMENTS_DIR,
                                EXPERIMENT_NAME,
                                draw_dir,
                                n_subjects_dir,
                                n_reps_dir,
                                n_pulses_dir,
                                seed_dir,
                                M.NAME,
                                sub_dir
                            )
                            a_true_sub = np.load(os.path.join(dir, "a_true.npy"))
                            a_pred_sub = np.load(os.path.join(dir, "a_pred.npy"))

                            a_pred_sub_map = a_pred_sub.mean(axis=0)
                            a_true_sub = a_true_sub

                            differences += (a_pred_sub_map[0, 1, 0] - a_pred_sub_map[0, 0, 0]).reshape(-1,).tolist()

                            a_true += a_pred_sub_map.reshape(-1,).tolist()
                            a_pred += a_true_sub.reshape(-1,).tolist()

                        a_true = np.array(a_true)
                        a_pred = np.array(a_pred)

                        differences = np.array(differences)
                        if n_subjects != 1:
                            curr_prob = stats.ttest_1samp(
                                a=differences, popmean=0, alternative="less"
                            ).pvalue.item()
                        else:
                            curr_prob = 1

                    else:
                        raise ValueError

                    curr_mae = np.abs(a_true - a_pred).mean()
                    curr_mse = np.square(a_true - a_pred).mean()
                    mae.append(curr_mae)
                    mse.append(curr_mse)
                    prob.append(curr_prob)

    mae = np.array(mae).reshape(len(n_subjects_space), len(draws_space), len(seeds_for_generating_subjects), len(models))
    mse = np.array(mse).reshape(len(n_subjects_space), len(draws_space), len(seeds_for_generating_subjects), len(models))
    prob = np.array(prob).reshape(len(n_subjects_space), len(draws_space), len(seeds_for_generating_subjects), len(models))

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")
    logger.info(f"PROB: {prob.shape}")

    mae = mae.mean(axis=-2)
    mse = mse.mean(axis=-2)
    prob[..., 0] = prob[..., 0] > .95
    if prob.shape[-1] > 1:
        prob[..., 1] = prob[..., 1] < .05
    prob = prob.mean(axis=-2)
    logger.info(f"MAE after reps dim removed: {mae.shape}")
    logger.info(f"MSE after reps dim removed: {mse.shape}")
    logger.info(f"PROB after reps dim removed: {prob.shape}")

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

    fig.align_xlabels()
    fig.align_ylabels()
    dest = os.path.join(simulator.build_dir, "number_of_subjects.svg")
    fig.savefig(dest, dpi=600)
    dest = os.path.join(simulator.build_dir, "number_of_subjects.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()
