import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np

from hbmep.config import Config
from hbmep.model.utils import Site as site

from models import RectifiedLogistic
from learn_posterior import TOML_PATH, DATA_PATH

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf

from models import RectifiedLogistic
from utils import run_inference

from scipy.stats import gaussian_kde
from scipy.integrate import nquad
from joblib import Parallel, delayed

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Change this to indicate path to inference.pkl from learn_posterior.py
POSTERIOR_PATH = "/home/mcintosh/Local/temp/test_hbmep/hbmep_sim/build/learn_posterior_2/inference.pkl"

# Change this to indicate toml path
TOML_PATH = "/home/mcintosh/Local/gitprojects/hbmep-paper/configs/experiments/basic_setup.toml"

def cumulative_gaussian(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def integrand(x, y, kde):
    pdf_val = kde.evaluate([x, y])[0]
    return -pdf_val * np.log(pdf_val) if pdf_val > 0 else 0


def fit_lookahead_wrapper(simulation_df_future, cand_y_at_x, config):
    for ix_muscle in range(len(config.RESPONSE)):
        simulation_df_future.iloc[-1, simulation_df_future.columns.get_loc(config.RESPONSE[ix_muscle])] = cand_y_at_x[0][ix_muscle]
    model_fut, mcmc_fut, posterior_samples_fut = fit_new_model(0, simulation_df_future,
                                                               do_save=False, make_figures=False)
    entropy = calculate_entropy(posterior_samples_fut, config)
    return entropy


def calculate_entropy(posterior_samples_fut, config):
    entropy = []
    for ix_muscle in range(len(config.RESPONSE)):
        posterior_samples_a = posterior_samples_fut['a'][:, 0, ix_muscle]
        posterior_samples_b = posterior_samples_fut['H'][:, 0, ix_muscle]

        joint_samples = np.column_stack((posterior_samples_a, posterior_samples_b))
        kde = gaussian_kde(joint_samples.T)

        # the fact that the bw is not set is pretty dodgy I think for an entropy calc.
        # could introduce a lot of bias/error
        bounds_a = (np.min(posterior_samples_a) * 0.9, np.max(posterior_samples_a) * 1.1)
        bounds_b = (np.min(posterior_samples_b) * 0.9, np.max(posterior_samples_b) * 1.1)

        entropy_muscle, _ = nquad(integrand, [bounds_a, bounds_b], args=(kde,))
        entropy.append(entropy_muscle)
    return np.mean(entropy)


def negative_log_likelihood(params, intensities, responses):
    mu, sigma = params
    responses = np.array(responses)  # Convert responses to a numpy array
    probabilities = cumulative_gaussian(intensities, mu, sigma)
    # Ensuring probabilities are within a range to avoid log(0)
    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
    return -np.sum(responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities))


def choose_next_intensity(mu, sigma, range_min, range_max):
    return np.clip(mu + np.random.randn() * sigma, range_min, range_max)


def fit_new_model(ix, df, do_save=True, make_figures=True):
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, f"learn_posterior_{ix}")

    model = RectifiedLogistic(config=config)
    model._make_dir(model.build_dir)

    """ Set up logging in build directory """
    # dest = os.path.join(model.build_dir, "learn_posterior.log")
    # logging.basicConfig(
    #     format=FORMAT,
    #     level=logging.INFO,
    #     handlers=[
    #         logging.FileHandler(dest, mode="w"),
    #         logging.StreamHandler()
    #     ],
    #     force=True
    # )
    # logger.info(f"Logging to {dest}")
    #
    # # Change this to indicate dataset path
    # src = DATA_PATH
    # df = pd.read_csv(src)
    #
    # """ Filter dataset as required """
    # logger.info(df["participant_condition"].unique().tolist())
    # ind = df["participant_condition"].isin(["Uninjured"])
    # df = df[ind].reset_index(drop=True).copy()

    """ Run inference """
    model, mcmc, posterior_samples = run_inference(config, model, df, do_save=do_save, make_figures=make_figures)
    return model, mcmc, posterior_samples


def main():
    random_seed_start = 100
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "simulate_data")

    simulator = RectifiedLogistic(config=config)
    simulator._make_dir(simulator.build_dir)

    """ Set up logging in build directory """
    dest = os.path.join(simulator.build_dir, "simulate_data.log")
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")

    """ Load learnt posterior """
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        model, mcmc, posterior_samples = pickle.load(g)

    logger.info(
        f"Logging all sites learnt posterior with their shapes ..."
    )   # Useful for freezing priors
    for k, v in posterior_samples.items():
        logger.info(f"{k}: {v.shape}")

    """
    Simulate new participants
    at equi-spaced intensities.
    We need to freeze the priors.
    """
    # Simulate TOTAL_SUBJECTS subjects
    TOTAL_PULSES = 1
    TOTAL_SUBJECTS = 1
    # Create template dataframe for simulation
    simulation_df = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame([0, 90], columns=[simulator.intensity]),
            how="cross"
        )
    simulation_df = simulator.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=0,
        num=TOTAL_PULSES
    )
    logger.info(
        f"Simulation (new participants) dataframe: {simulation_df.shape}"
    )

    sites_to_exclude = {
        site.a, site.b, site.v,
        site.L, site.ell, site.H,
        site.c_1, site.c_2,
        site.mu, site.beta,
        site.obs
    }
    posterior_samples = {
        k: v for k, v in posterior_samples.items() if k not in sites_to_exclude
    }

    simulation_df.loc[0, 'TMSInt'] = 0
    simulation_ppd = \
        simulator.predict(df=simulation_df, posterior_samples=posterior_samples)

    vec_ = list(simulation_ppd.keys())
    vec_ = [item for item in vec_ if item != site.obs]
    vec_ = [item for item in vec_ if item != site.mu]
    vec_ = [item for item in vec_ if item != site.beta]
    for k in vec_:
        posterior_samples[k] = simulation_ppd[k]

    for k in posterior_samples.keys():
        posterior_samples[k] = posterior_samples[k][-5:-4, ...]

    ix_gen_seed = 48
    range_min, range_max = 0, 100

    try_al = True
    if try_al:
        np.random.seed(ix_gen_seed)

        # Initial guess
        intensities = [range_min]
        responses = []
        N_max = 30

        for ix in range(N_max):
            # Simulate response
            simulation_df.loc[0, 'TMSInt'] = intensities[-1]
            simulation_ppd = \
                simulator.predict(df=simulation_df, posterior_samples=posterior_samples, random_seed=random_seed_start + ix)
            mep_size = simulation_ppd['obs'][0][0]
            response = mep_size
            responses.append(response)

            # TODO: this needs to be generalised:
            simulation_df_happened = pd.DataFrame({'TMSInt': intensities,
                                                   config.RESPONSE[0]: np.array(responses)[:, 0],
                                                   config.RESPONSE[1]: np.array(responses)[:, 1],
                                                   })
            simulation_df_happened['participant___participant_condition'] = 0
            simulation_df_happened['participant'] = '0'
            simulation_df_happened['participant_condition'] = 'Uninjured'
            # Choose next intensity
            model_hap, mcmc_hap, posterior_samples_hap = fit_new_model(ix, simulation_df_happened)
            entropy_base = calculate_entropy(posterior_samples_hap, config)

            if ix < 1:
                next_intensity = range_max
            else:
                list_candidate_intensities = range(range_min, range_max)
                vec_entropy = np.zeros((len(list_candidate_intensities)))
                for ix_future in range(len(list_candidate_intensities)):
                    N = 12  # this is what has to be very large
                    candidate_int = list_candidate_intensities[ix_future]
                    print(f'Testing intensity: {candidate_int}')
                    simulation_df_future = pd.DataFrame({'TMSInt': [candidate_int]})
                    simulation_df_future['participant___participant_condition'] = 0
                    simulation_df_future['participant'] = '0'
                    simulation_df_future['participant_condition'] = 'Uninjured'
                    # TODO: turn off mixture here if it is in the model
                    posterior_predictive = model.predict(df=simulation_df_future,
                                                         posterior_samples=posterior_samples_hap)
                    ix_from_chain = np.random.choice(range(posterior_predictive['obs'].shape[0]), N)

                    candidate_y_at_this_x = posterior_predictive['obs'][ix_from_chain]

                    simulation_df_future = simulation_df_happened.copy()
                    # TODO: this needs to be generalised:
                    new_values = {'TMSInt': candidate_int, config.RESPONSE[0]: 0, config.RESPONSE[1]: 0}
                    new_row = simulation_df_future.iloc[-1].copy()
                    new_row.update(new_values)
                    new_row = pd.DataFrame([new_row])
                    simulation_df_future = pd.concat([simulation_df_future, new_row], ignore_index=True)

                    fit_lookahead_wrapper(simulation_df_future, candidate_y_at_this_x[0], config)
                    with Parallel(n_jobs=-1) as parallel:
                        entropy_list = parallel(
                            delayed(fit_lookahead_wrapper)(simulation_df_future, candidate_y_at_this_x[ix_sample], config)
                            for ix_sample in range(N)
                        )

                    vec_entropy[ix_future] = np.mean(entropy_list)
                ix_min_entropy = np.argmin(vec_entropy - entropy_base)
                next_intensity = list_candidate_intensities[ix_min_entropy]

            intensities.append(next_intensity)

        intensity_final = intensities.pop()

    return


if __name__ == "__main__":
    main()
