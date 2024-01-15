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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Change this to indicate path to inference.pkl from learn_posterior.py
POSTERIOR_PATH = "/home/mcintosh/Local/temp/test_hbmep/hbmep_sim/build/learn_posterior/inference.pkl"

# Change this to indicate toml path
TOML_PATH = "/home/mcintosh/Local/gitprojects/hbmep-paper/configs/experiments/basic_setup.toml"

def cumulative_gaussian(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def integrand(x, y, kde):
    pdf_val = kde.evaluate([x, y])[0]
    return -pdf_val * np.log(pdf_val) if pdf_val > 0 else 0


def calculate_entropy(posterior_samples_fut):
    posterior_samples_a = posterior_samples_fut['a'][:, 0, 0]
    posterior_samples_b = posterior_samples_fut['H'][:, 0, 0]

    joint_samples = np.column_stack((posterior_samples_a, posterior_samples_b))
    kde = gaussian_kde(joint_samples.T)

    bounds_a = (np.min(posterior_samples_a), np.max(posterior_samples_a))
    bounds_b = (np.min(posterior_samples_b), np.max(posterior_samples_b))

    entropy, _ = nquad(integrand, [bounds_a, bounds_b], args=(kde,))
    return entropy


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

    logger.info(f"Simulating new participants ...")
    x, y, var = [], [], []
    for ix in range(0, 100, 1):
        x_ = ix + np.random.random()
        simulation_df.loc[0, 'TMSInt'] = x_
        simulation_ppd = \
            simulator.predict(df=simulation_df, posterior_samples=posterior_samples, random_seed=ix)
        y_ = simulation_ppd['obs'][0][0][0]
        x.append(x_)
        y.append(y_)
        var_ = (simulation_ppd[site.mu] / (posterior_samples[site.c_1] + (posterior_samples[site.c_2]/simulation_ppd[site.mu]))).item()
        var.append(var_)


    # plt.subplot(1,2, 1)
    # plt.plot(x, y, '.')
    # plt.ylim([0, 1])
    # plt.subplot(1, 2, 2)
    # sns.lineplot(x=x, y=var)
    # plt.show()
    ix_gen_seed = 47
    range_min, range_max = 10, 90

    try_mtat = True
    if try_mtat:
        # Parameters
        np.random.seed(ix_gen_seed)

        # Initial guess
        intensities = [np.random.uniform(range_min, range_max)]
        responses = []
        th = 50e-3

        for ix in range(30):
            # Simulate response
            simulation_df.loc[0, 'TMSInt'] = intensities[-1]
            simulation_ppd = \
                simulator.predict(df=simulation_df, posterior_samples=posterior_samples, random_seed=ix)
            mep_size = simulation_ppd['obs'][0][0][0]
            response = mep_size > th
            responses.append(response)

            # Estimate threshold
            result = minimize(negative_log_likelihood, [np.mean(intensities), 1], args=(intensities, responses),
                              bounds=[(range_min, range_max), (0.1, 30)])
            estimated_mu, estimated_sigma = result.x
            if len(responses) < 5:  # Wait until we have enough data
                estimated_sigma = 20

            # Choose next intensity
            next_intensity = choose_next_intensity(estimated_mu, estimated_sigma, range_min, range_max)
            intensities.append(next_intensity)
            # intensities[-1] = ix/5
        intensity_final = intensities.pop()

        simulation_df_local = pd.DataFrame({'TMSInt': range(101)})
        simulation_df_local['participant___participant_condition'] = 0

        simulation_ppd = \
            simulator.predict(df=simulation_df_local, posterior_samples=posterior_samples)

        arr = simulation_ppd[site.mu][0, :, 0] > th
        first_true_index = np.where(arr == True)[0][0]
        true_mu_at_th = simulation_df_local["TMSInt"][first_true_index]

        print(f"Estimated Threshold (mu): {estimated_mu:.2f}")
        print(f"Estimated Sigma: {estimated_sigma:.2f}")
        plt.plot(intensities, 'o')
        plt.plot([0, len(intensities)], posterior_samples[site.a][0][0][0] * np.ones((2)))
        plt.plot([0, len(intensities)], true_mu_at_th * np.ones((2)))
        plt.show()

        # plt.plot(simulation_df_local['TMSInt'], simulation_ppd[site.mu][0, :, 0])
        # plt.show()

    try_bo = False
    if try_bo:
        np.random.seed(ix_gen_seed)
        # Parameters

        # Initial guess
        intensities = [np.random.uniform(range_min, range_max)]
        intensities = [range_min]
        responses = []

        for ix in range(30):
            # Simulate response
            simulation_df.loc[0, 'TMSInt'] = intensities[-1]
            simulation_ppd = \
                simulator.predict(df=simulation_df, posterior_samples=posterior_samples, random_seed=ix)
            mep_size = simulation_ppd['obs'][0][0][0]
            response = mep_size
            responses.append(response)

            simulation_df_happened = pd.DataFrame({'TMSInt': intensities, 'PKPK_APB': responses})
            simulation_df_happened['participant___participant_condition'] = 0
            simulation_df_happened['participant'] = '0'
            simulation_df_happened['participant_condition'] = 'Uninjured'
            # Choose next intensity
            model_hap, mcmc_hap, posterior_samples_hap = fit_new_model(ix, simulation_df_happened)
            simulation_df_happened_pred = model.make_prediction_dataset(df=simulation_df_happened)
            # TODO: need to turn off the outliers here!
            posterior_predictive = model.predict(df=simulation_df_happened_pred, posterior_samples=posterior_samples_hap)
            x = np.percentile(posterior_predictive['obs'], [5, 95], 0)
            x = x[1, :] - x[0, :]
            ix_max = np.argmax(x)

            if ix < 1:
                next_intensity = range_max
            elif ix % 2 == 0:
                next_intensity = simulation_df_happened_pred.loc[ix_max, "TMSInt"]
            else:
                # ix_rand_a = np.random.choice(posterior_samples_hap['a'].shape[0])
                # next_intensity = posterior_samples_hap['a'][ix_rand_a, 0, 0]
                next_intensity = np.mean(posterior_samples_hap['a'], 0)[0][0]

                next_intensity = next_intensity * (1 + 0.2 * (np.random.rand() - 0.5) * 2)

            # next_intensity = np.random.rand() * 100 ## REPLACE
            intensities.append(next_intensity)
            # intensities[-1] = ix/5
        intensity_final = intensities.pop()

    try_ab = True
    if try_ab:
        np.random.seed(ix_gen_seed)
        # Parameters

        # Initial guess
        intensities = [np.random.uniform(range_min, range_max)]
        intensities = [range_min]
        responses = []
        N_max = 30

        for ix in range(N_max):
            # Simulate response
            simulation_df.loc[0, 'TMSInt'] = intensities[-1]
            simulation_ppd = \
                simulator.predict(df=simulation_df, posterior_samples=posterior_samples, random_seed=ix)
            mep_size = simulation_ppd['obs'][0][0][0]
            response = mep_size
            responses.append(response)

            simulation_df_happened = pd.DataFrame({'TMSInt': intensities, 'PKPK_APB': responses})
            simulation_df_happened['participant___participant_condition'] = 0
            simulation_df_happened['participant'] = '0'
            simulation_df_happened['participant_condition'] = 'Uninjured'
            # Choose next intensity
            model_hap, mcmc_hap, posterior_samples_hap = fit_new_model(ix, simulation_df_happened)
            entropy_base = calculate_entropy(posterior_samples_hap)

            if ix < 1:
                next_intensity = range_max
            else:
                list_candidate_intensities = range(range_min, range_max)
                vec_entropy = np.zeros((len(list_candidate_intensities)))
                for ix_future in range(len(list_candidate_intensities)):
                    N = 1  # this is what has to be very large
                    candidate_int = list_candidate_intensities[ix_future]
                    print(f'Testing intensity: {candidate_int}')
                    simulation_df_future = pd.DataFrame({'TMSInt': [candidate_int]})
                    simulation_df_future['participant___participant_condition'] = 0
                    simulation_df_future['participant'] = '0'
                    simulation_df_future['participant_condition'] = 'Uninjured'
                    # TODO: need to turn off the outliers here!
                    posterior_predictive = model.predict(df=simulation_df_future,
                                                         posterior_samples=posterior_samples_hap)
                    candidate_y_at_this_x = np.random.choice(posterior_predictive['obs'][:, 0, 0], N)

                    simulation_df_future = simulation_df_happened.copy()
                    new_values = {'TMSInt': candidate_int, 'PKPK_APB': 0}
                    new_row = simulation_df_future.iloc[-1].copy()
                    new_row.update(new_values)
                    new_row = pd.DataFrame([new_row])
                    simulation_df_future = pd.concat([simulation_df_future, new_row], ignore_index=True)

                    entropy_list = []
                    for ix_sample in range(N):
                        simulation_df_future.iloc[-1, simulation_df_future.columns.get_loc("PKPK_APB")] = candidate_y_at_this_x[ix_sample]
                        model_fut, mcmc_fut, posterior_samples_fut = fit_new_model(0, simulation_df_future, do_save=False, make_figures=False)
                        entropy = calculate_entropy(posterior_samples_fut)

                        entropy_list.append(entropy)

                    vec_entropy[ix_future] = np.mean(entropy_list)
                ix_min_entropy = np.argmin(vec_entropy - entropy_base)
                next_intensity = list_candidate_intensities[ix_min_entropy]

            # next_intensity = np.random.rand() * 100 ## REPLACE
            intensities.append(next_intensity)
            # intensities[-1] = ix/5
        intensity_final = intensities.pop()

    return


if __name__ == "__main__":
    main()
