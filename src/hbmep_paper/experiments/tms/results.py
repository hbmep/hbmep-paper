import os
import pickle
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, median_absolute_error

import jax
import jax.numpy as jnp

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.experiments.tms.sparse_subjects_power import HierarchicalBayesianSimulator

logger = logging.getLogger(__name__)


def _process_hb_posterior():
    model_link = "hierarchical_bayesian"
    mu_a_delta = None
    p_value = None
    a_error = None

    for N in tqdm(N_space[:N_LIM], desc="N"):
        curr_N_mu_delta = None
        curr_N_p_value = None
        curr_N_a_error = None

        N_dir = f"N_{N}"

        for draw_ind in tqdm(draws_space[:DRAW_LIM], desc="Draw"):
            curr_draw_mu_delta = None
            curr_draw_p_value = None
            curr_draw_a_error = None

            draw_dir = f"draw_{draw_ind}"

            for seed in repeats_space[:SEED_LIM]:
                seed_dir = f"seed_{seed}"

                """ Posterior samples """
                src = os.path.join(CONFIG.BUILD_DIR, simulation_prefix, experiment_prefix, model_link, draw_dir, N_dir, seed_dir, "inference.pkl")
                with open(src, "rb") as g:
                    posterior_samples, = pickle.load(g)

                """ mu_a_delta """
                mu_delta_temp = posterior_samples["mu_a_delta"]

                mu_delta_temp = mu_delta_temp[None, ...]
                if curr_draw_mu_delta is None:
                    curr_draw_mu_delta = mu_delta_temp
                else:
                    curr_draw_mu_delta = np.concatenate([curr_draw_mu_delta, mu_delta_temp], axis=0)

                """ True threshold """
                subjects_ind = \
                    jax.random.choice(
                        key=jax.random.PRNGKey(seed),
                        a=np.arange(0, TOTAL_SUBJECTS, 1),
                        shape=(N,),
                        replace=False
                    ) \
                    .tolist()
                a_true = a[draw_ind, ...]
                a_true = a_true[sorted(subjects_ind), ...]

                """ Estimated threshold """
                a_temp = np.array(posterior_samples[site.a].mean(axis=0))

                """ Error """
                mae_temp = np.abs(a_true - a_temp)
                mae_temp = mae_temp.reshape(-1,).mean().item()

                mse_temp = (a_true - a_temp) ** 2
                mse_temp = mse_temp.reshape(-1,).mean().item()

                median_error = median_absolute_error(y_true=a_true.reshape(-1,), y_pred=a_temp.reshape(-1,))

                error_temp = np.array([mae_temp, mse_temp, median_error])

                error_temp = error_temp[None, ...]
                if curr_draw_a_error is None:
                    curr_draw_a_error = error_temp
                else:
                    curr_draw_a_error = np.concatenate([curr_draw_a_error, error_temp], axis=0)

                """ p-value """
                diff = a_temp[..., 1, 0] - a_temp[..., 0, 0]
                diff = diff.reshape(-1,)

                ttest = stats.ttest_1samp(a=diff, popmean=0, alternative="less").pvalue
                ranktest = stats.wilcoxon(x=diff, alternative="less").pvalue

                p_value_temp = np.array([ttest, ranktest])

                p_value_temp = p_value_temp[None, ...]
                if curr_draw_p_value is None:
                    curr_draw_p_value = p_value_temp
                else:
                    curr_draw_p_value = np.concatenate([curr_draw_p_value, p_value_temp], axis=0)

            curr_draw_mu_delta = curr_draw_mu_delta[None, ...]
            if curr_N_mu_delta is None:
                curr_N_mu_delta = curr_draw_mu_delta
            else:
                curr_N_mu_delta = np.concatenate([curr_N_mu_delta, curr_draw_mu_delta], axis=0)

            curr_draw_a_error = curr_draw_a_error[None, ...]
            if curr_N_a_error is None:
                curr_N_a_error = curr_draw_a_error
            else:
                curr_N_a_error = np.concatenate([curr_N_a_error, curr_draw_a_error], axis=0)

            curr_draw_p_value = curr_draw_p_value[None, ...]
            if curr_N_p_value is None:
                curr_N_p_value = curr_draw_p_value
            else:
                curr_N_p_value = np.concatenate([curr_N_p_value, curr_draw_p_value], axis=0)

        curr_N_mu_delta = curr_N_mu_delta[None, ...]
        if mu_a_delta is None:
            mu_a_delta = curr_N_mu_delta
        else:
            mu_a_delta = np.concatenate([mu_a_delta, curr_N_mu_delta], axis=0)

        curr_N_a_error = curr_N_a_error[None, ...]
        if a_error is None:
            a_error = curr_N_a_error
        else:
            a_error = np.concatenate([a_error, curr_N_a_error], axis=0)

        curr_N_p_value = curr_N_p_value[None, ...]
        if p_value is None:
            p_value = curr_N_p_value
        else:
            p_value = np.concatenate([p_value, curr_N_p_value], axis=0)

    return mu_a_delta.copy(), a_error.copy(), p_value.copy()


def _process_nhb_posterior():
    model_link = "non_hierarchical_bayesian"
    p_value = None
    a_error = None

    for N in tqdm(N_space[:N_LIM], desc="N"):
        curr_N_p_value = None
        curr_N_a_error = None

        N_dir = f"N_{N}"

        for draw_ind in tqdm(draws_space[:DRAW_LIM], desc="Draw"):
            curr_draw_p_value = None
            curr_draw_a_error = None

            draw_dir = f"draw_{draw_ind}"

            for seed in repeats_space[:SEED_LIM]:
                seed_dir = f"seed_{seed}"

                """ True threshold """
                subjects_ind = \
                    jax.random.choice(
                        key=jax.random.PRNGKey(seed),
                        a=np.arange(0, TOTAL_SUBJECTS, 1),
                        shape=(N,),
                        replace=False
                    ) \
                    .tolist()
                a_true = a[draw_ind, ...]
                a_true = a_true[sorted(subjects_ind), ...]

                a_pred = None
                for subject in sorted(subjects_ind):
                    subject_dir = f"subject_{subject}"

                    """ Posterior samples """
                    src = os.path.join(CONFIG.BUILD_DIR, simulation_prefix, experiment_prefix, model_link, draw_dir, N_dir, seed_dir, subject_dir, "inference.pkl")
                    with open(src, "rb") as g:
                        posterior_samples, = pickle.load(g)

                    """ Estimated threshold """
                    a_temp = np.array(posterior_samples[site.a].mean(axis=0))

                    if a_pred is None:
                        a_pred = a_temp
                    else:
                        a_pred = np.concatenate([a_pred, a_temp], axis=0)

                """ Error """
                mae_temp = np.abs(a_true - a_pred)
                mae_temp = mae_temp.reshape(-1,).mean().item()

                mse_temp = (a_true - a_pred) ** 2
                mse_temp = mse_temp.reshape(-1,).mean().item()

                median_error = median_absolute_error(y_true=a_true.reshape(-1,), y_pred=a_pred.reshape(-1,))

                error_temp = np.array([mae_temp, mse_temp, median_error])

                error_temp = error_temp[None, ...]
                if curr_draw_a_error is None:
                    curr_draw_a_error = error_temp
                else:
                    curr_draw_a_error = np.concatenate([curr_draw_a_error, error_temp], axis=0)

                """ p-value """
                diff = a_pred[..., 1, 0] - a_pred[..., 0, 0]
                diff = diff.reshape(-1,)

                ttest = stats.ttest_1samp(a=diff, popmean=0, alternative="less").pvalue
                ranktest = stats.wilcoxon(x=diff, alternative="less").pvalue

                p_value_temp = np.array([ttest, ranktest])

                p_value_temp = p_value_temp[None, ...]
                if curr_draw_p_value is None:
                    curr_draw_p_value = p_value_temp
                else:
                    curr_draw_p_value = np.concatenate([curr_draw_p_value, p_value_temp], axis=0)

            curr_draw_a_error = curr_draw_a_error[None, ...]
            if curr_N_a_error is None:
                curr_N_a_error = curr_draw_a_error
            else:
                curr_N_a_error = np.concatenate([curr_N_a_error, curr_draw_a_error], axis=0)

            curr_draw_p_value = curr_draw_p_value[None, ...]
            if curr_N_p_value is None:
                curr_N_p_value = curr_draw_p_value
            else:
                curr_N_p_value = np.concatenate([curr_N_p_value, curr_draw_p_value], axis=0)

        curr_N_a_error = curr_N_a_error[None, ...]
        if a_error is None:
            a_error = curr_N_a_error
        else:
            a_error = np.concatenate([a_error, curr_N_a_error], axis=0)

        curr_N_p_value = curr_N_p_value[None, ...]
        if p_value is None:
            p_value = curr_N_p_value
        else:
            p_value = np.concatenate([p_value, curr_N_p_value], axis=0)

    return a_error.copy(), p_value.copy()


def _process(arr):
    me = arr.mean(axis=-1)
    sem = stats.sem(arr, axis=-1)
    std = arr.std(axis=-1)
    return me, sem, std


def _process_bp(arr):
    arr = ((arr < 0).mean(axis=(-1, -2, -3)) > .95).mean(axis=-1)
    return _process(arr)


def _process_fp(arr):
    arr = (arr < .05).mean(axis=-1)
    return _process(arr)


if __name__=="__main__":
    """ Initialize simulator  """
    toml_path = "/home/vishu/repos/hbmep-paper/configs/paper/tms/mixed-effects/simulator/hierarchical_bayesian_simulator.toml"
    mu_a_delta, sigma_a_delta = -1.5, 1
    simulation_prefix = f"mu_a_delta_{mu_a_delta}__sigma_a_delta_{sigma_a_delta}"

    CONFIG = Config(toml_path=toml_path)
    CONFIG.BUILD_DIR = os.path.join(CONFIG.BUILD_DIR, simulation_prefix)

    SIMULATOR = HierarchicalBayesianSimulator(config=CONFIG, mu_a_delta=mu_a_delta, sigma_a_delta=sigma_a_delta)

    """ Simulation dataframe """
    TOTAL_SUBJECTS = 1000

    PREDICTION_DF = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[SIMULATOR.subject]) \
        .merge(
            pd.DataFrame(np.arange(0, 2, 1), columns=SIMULATOR.features),
            how="cross"
        ) \
        .merge(
            pd.DataFrame([0, 100], columns=[SIMULATOR.intensity]),
            how="cross"
        )
    PREDICTION_DF = SIMULATOR.make_prediction_dataset(df=PREDICTION_DF, num_points=60)

    """ Load learned posterior """
    src = os.path.join(SIMULATOR.build_dir, "POSTERIOR_PREDICTIVE.pkl")
    with open(src, "rb") as g:
        POSTERIOR_PREDICTIVE, = pickle.load(g)

    OBS = np.array(POSTERIOR_PREDICTIVE[site.obs])
    a = np.array(POSTERIOR_PREDICTIVE[site.a])

    """ Experiment space """
    experiment_prefix = "sprase_subjects_power"

    TOTAL_DRAWS = OBS.shape[0]
    N_space = [1, 2, 4, 6, 8, 12, 16, 20]
    n_draws = 50
    n_repeats = 50

    keys = jax.random.split(SIMULATOR.rng_key, num=2)
    draws_space = \
        jax.random.choice(
            key=keys[0],
            a=np.arange(0, TOTAL_DRAWS, 1),
            shape=(n_draws,),
            replace=False
        ) \
        .tolist()
    repeats_space = \
        jax.random.choice(
            key=keys[1],
            a=np.arange(0, n_repeats * 100, 1),
            shape=(n_repeats,),
            replace=False
        ) \
        .tolist()

    N_LIM, DRAW_LIM, SEED_LIM = len(N_space), 5, n_repeats
    fname = f"N_LIM_{N_LIM}__DRAW_LIM_{DRAW_LIM}__SEED_LIM_{SEED_LIM}"
    extension = ".pkl"

    hbm_mu_delta, hbm_error, hbm_p_value = _process_hb_posterior()
    nhbm_error, nhbm_p_value = _process_nhb_posterior()

    dest = os.path.join(SIMULATOR.build_dir, fname + extension)
    with open(dest, "wb") as f:
        pickle.dump((hbm_mu_delta, hbm_error, hbm_p_value, nhbm_error, nhbm_p_value), f)
    # src = os.path.join(SIMULATOR.build_dir, fname + extension)
    # with open(src, "rb") as f:
    #     hbm_mu_delta, hbm_error, hbm_p_value, nhbm_error, nhbm_p_value, = pickle.load(f)

    """ Plot """
    lw = 1.4
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False, constrained_layout=True)

    ax = axes[0, 0]
    me, sem, std = _process(hbm_error[..., 0].reshape(N_LIM, -1))
    ax.errorbar(x=N_space[:N_LIM], y=me, yerr=sem, marker="o", label="Hieararchical Bayesian", linestyle="--", linewidth=lw, ms=4)

    me, sem, std = _process(nhbm_error[..., 0].reshape(N_LIM, -1))
    ax.errorbar(x=np.array(N_space[:N_LIM]) + .1, y=me, yerr=sem, marker="o", label="Non-Hieararchical Bayesian", linestyle="--", linewidth=lw, ms=4)

    ax.set_xticks(N_space)
    ax.grid()
    ax.legend()

    ax = axes[0, 1]
    me, sem, std = _process_bp(hbm_mu_delta)
    ax.errorbar(x=N_space[:N_LIM], y=me, yerr=sem, marker="o", label="Hiearchical Bayesian", linestyle="--", linewidth=lw, ms=4)

    me, sem, std = _process_fp(nhbm_p_value[..., 1])
    ax.errorbar(x=np.array(N_space[:N_LIM]), y=me, yerr=sem, marker="o", label="Non-Hieararchical Bayesian", linestyle="--", linewidth=lw, ms=4)

    ax.set_xticks(N_space)
    ax.set_yticks(np.arange(0, 1.1, .1))
    ax.grid()
    ax.legend()

    extension = ".png"
    dest = os.path.join(SIMULATOR.build_dir, fname + extension)
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")
