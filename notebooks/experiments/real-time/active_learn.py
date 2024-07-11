import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np
import jax

from hbmep.config import Config
from hbmep.model.utils import Site as site

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf

from learn_posterior import TOML_PATH
from hbmep.model import RectifiedLogistic
from utils import run_inference

from scipy.stats import gaussian_kde
from scipy.integrate import nquad
from joblib import Parallel, delayed
from pathlib import Path
from copy import deepcopy
import sys
import inspect


def output_info_about_variable(X):
    # just for some serious debugging...
    print("Type:", type(X))
    print("Value:", X)
    print("String Representation:", str(X))
    print("Repr Representation:", repr(X))
    print("Attributes and Methods:", dir(X))
    print("Documentation String:", X.__doc__)
    print("Size (in bytes):", sys.getsizeof(X))

    if hasattr(X, '__len__'):
        print("Length:", len(X))

    if hasattr(X, 'shape'):
        print("Shape:", X.shape)

    if hasattr(X, 'dtypes'):
        print("Data Types:", X.dtypes)

    if inspect.isfunction(X) or inspect.isclass(X):
        print("Signature:", inspect.signature(X))


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def create_max_diff_sequence(min_range=0, max_range=100, N=10):
    # courtesey of GPT4... make a vector where each new point is as far as possible from previous points
    # Create a vector spaced by N
    vector = np.linspace(min_range, max_range, N)

    # Initialize the sequence with the first element
    sequence = [vector[0]]

    # Remove the first element from the vector
    vector = np.delete(vector, 0)

    # Function to find the next element with the maximum minimum distance
    def find_max_min_distance_element(sequence, vector):
        max_min_distance = -1
        next_element = None

        for element in vector:
            min_distance = min([abs(element - x) for x in sequence])
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_element = element

        return next_element

    # Construct the sequence
    while len(vector) > 0:
        next_element = find_max_min_distance_element(sequence, vector)
        sequence.append(next_element)
        vector = np.delete(vector, np.where(vector == next_element))

    return sequence


def integrand(*args):
    kde = args[-1]
    points = args[:-1]
    pdf_val = kde.evaluate(points)[0]
    return -pdf_val * np.log(pdf_val) if pdf_val > 0 else 0


def fit_lookahead_wrapper(simulation_df_future, candidate_int, cand_y_at_x, config, opt_param):
    new_values = {**{config.RESPONSE[ix]: 0 for ix in range(len(config.RESPONSE))}, **{'TMSInt': candidate_int[0]}}
    new_row = simulation_df_future.iloc[-1].copy()
    new_row.update(new_values)
    new_row = pd.DataFrame([new_row])
    simulation_df_future = pd.concat([simulation_df_future, new_row], ignore_index=True)

    for ix_muscle in range(len(config.RESPONSE)):
        simulation_df_future.iloc[-1, simulation_df_future.columns.get_loc(config.RESPONSE[ix_muscle])] = cand_y_at_x[ix_muscle]
    config_local = deepcopy(config)
    config_local.BUILD_DIR = Path(config.BUILD_DIR) / f"ignore"
    model_fut, mcmc_fut, posterior_samples_fut = fit_new_model(config_local, simulation_df_future,
                                                               do_save=False, make_figures=False)

    entropy = calculate_entropy(posterior_samples_fut, config_local, opt_param)
    return entropy


def calculate_entropy(posterior_samples_fut, config, opt_param=('a', 'H')):
    # this is actually quite slow as well...
    entropy = []
    for ix_muscle in range(len(config.RESPONSE)):
        posterior_samples = []
        bounds = []
        for ix_opt_param in range(len(opt_param)):
            str_param = opt_param[ix_opt_param]
            posterior_samples_ = posterior_samples_fut[str_param][:, 0, ix_muscle]
            posterior_samples.append(posterior_samples_)
            extend_x = (np.max(posterior_samples_) - np.min(posterior_samples_)) * 0.2
            bounds_ = [np.min(posterior_samples_) - extend_x, np.max(posterior_samples_) + extend_x]
            bounds.append(bounds_)

        joint_samples = np.column_stack(posterior_samples)
        try:
            kde = gaussian_kde(joint_samples.T, bw_method='silverman')
            entropy_muscle, _ = nquad(integrand, bounds, args=(kde,))
            entropy.append(entropy_muscle)
        except Exception as e:
            print("An error occurred:", e)
            # print('Info about joint_samples.T:')
            # output_info_about_variable(joint_samples.T)
            # print('Sum axis=1:')
            # print(np.sum(joint_samples.T, axis=1))
            # print(joint_samples.T)
            # print('Info about bounds:')
            # output_info_about_variable(bounds)
            # print(bounds)
            # if this is rare maybe just set entropy_muscle to np.nan? and carry on?
            entropy.append(np.nan)
            # raise

    return entropy


def fit_new_model(config, df, do_save=True, make_figures=True):
    # toml_path = TOML_PATH
    # config = Config(toml_path=toml_path)
    # config.BUILD_DIR = os.path.join(config.BUILD_DIR, f"learn_posterior_{ix}")

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
    # before running this 1) learn_posterior 2) simulate_data
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    root_dir = Path(config.BUILD_DIR)
    config.BUILD_DIR = root_dir / 'simulate_data'
    config_fast = deepcopy(config)
    config_fast.MCMC_PARAMS['num_chains'] = 1
    config_fast.MCMC_PARAMS['num_warmup'] = 500
    config_fast.MCMC_PARAMS['num_samples'] = 1000
    seed = dict()
    seed['ix_gen_seed'] = 10
    seed['ix_participant'] = 62
    opt_param = [site.a]  # [site.a]
    choose_interp = False
    make_figures_per_sample = False  # True eventually crashes some X-sessions
    N_max = 40
    N_reps = 1  # if N_max = 50, then good choices are 1, 2, 5, 10
    N_obs = 15  # this is how many entropy calcs to do per every y drawn from x... larger is better
    range_min, range_max = 0, 100
    do_parallel = True  # easier to debug non-parallel
    assert N_obs % 2 != 0, "Better if N_obs is odd."
    if choose_interp:
        vec_intensity_lin = create_max_diff_sequence(min_range=range_min, max_range=range_max, N=int(np.ceil(N_max/N_reps) + 1))
        if N_reps > 1:
            if not (N_max / N_reps) % 1 == 0:
                print(f'N_max / N_reps is not an integer: {N_max / N_reps}. Better if it is.')
            vec_intensity_lin = np.tile(np.array(vec_intensity_lin), (N_reps, 1)).transpose().flatten()
            vec_intensity_lin = list(vec_intensity_lin[:N_max])
        intensities = [vec_intensity_lin[0]]
    else:
        assert N_reps == 1, "N_reps has to be = 1 if active learning."
        vec_intensity_lin = []
        intensities = [range_min]

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
    src = root_dir / "learn_posterior" / "inference.pkl"
    with open(src, "rb") as g:
        model, mcmc, posterior_samples, df = pickle.load(g)
    df_data = df.copy()

    rng_key = model.rng_key
    seed['predict'] = list(jax.random.split(rng_key, num=N_max))

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
    df = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame([0, 90], columns=[simulator.intensity]),
            how="cross"
        )
    df = simulator.make_prediction_dataset(
        df=df, min_intensity=0, max_intensity=0, num_points=TOTAL_PULSES)

    logger.info(
        f"Simulation (new participants) dataframe: {df.shape}"
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

    df.loc[0, 'TMSInt'] = 0
    # return_sites = ['']
    simulation_ppd = \
        simulator.predict(df=df, posterior_samples=posterior_samples)

    # if 'gradient' in simulation_ppd.keys():
    #     simulation_ppd['max_grad'] = np.zeros(simulation_ppd[site.H].shape)
    #     for ix_p in range(len(participants)):
    #         for ix_muscle in range(n_muscles):
    #             for ix_cond in range(len(conditions)):
    #                 Y = pp[ix_p][ix_cond]['gradient'][:, :, ix_muscle]  # not sure why this index is flipped...
    #                 _posterior_samples['max_grad'][:, ix_cond, ix_p, ix_muscle] = np.max(Y, axis=1)

    vec_ = list(simulation_ppd.keys())
    vec_ = [item for item in vec_ if item != site.obs]
    vec_ = [item for item in vec_ if item != site.mu]
    vec_ = [item for item in vec_ if item != site.beta]
    vec_ = [item for item in vec_ if item != 'gradient']
    # simulation_ppd only has sites not present in posterior samples, so add them back in
    for k in vec_:
        posterior_samples[k] = simulation_ppd[k]

    posterior_samples_individual = posterior_samples.copy()
    for k in posterior_samples.keys():
        posterior_samples_individual[k] = posterior_samples[k][seed['ix_participant']:seed['ix_participant']+1, ...]

    posterior_samples_individual['a'][0][0][0] = posterior_samples_individual['a'][0][0][0] + 4
    if posterior_samples_individual['a'].shape[-1] == 2:
        posterior_samples_individual['a'][0][0][1] = posterior_samples_individual['a'][0][0][1] - 4

    """ Save individual participant info"""
    d_participant = root_dir / 'participant'
    if not os.path.exists(d_participant):
        os.makedirs(d_participant)
    dest = d_participant / "inference.pkl"
    if dest.exists():
        with open(dest, "rb") as g:
            model, mcmc, posterior_samples_individual, df, seed = pickle.load(g)
    else:
        with open(dest, "wb") as f:
            pickle.dump((model, mcmc, posterior_samples_individual, df, seed), f)

    # SANITY CHECK
    # simulation_df_test = simulator.make_prediction_dataset(
    #     df=df, min_intensity=0, max_intensity=100, num=500)
    # simulation_ppd = \
    #     simulator.predict(df=simulation_df_test, posterior_samples=posterior_samples, random_seed=random_seed_start + 0)
    # plt.plot(simulation_df_test.loc[:, 'TMSInt'], simulation_ppd['obs'][0, :, 0])
    # plt.plot(simulation_df_test.loc[:, 'TMSInt'], simulation_ppd['obs'][0, :, 1])
    # plt.show()
    assert N_max == len(seed['predict']), "N_max issue"
    np.random.seed(seed['ix_gen_seed'])

    # Initial guess
    responses = []

    for ix in range(N_max):
        # Simulate response
        seed_local = seed['predict'][ix]
        df.loc[0, 'TMSInt'] = intensities[-1]
        simulation_ppd = \
            simulator.predict(df=df, posterior_samples=posterior_samples_individual, rng_key=seed_local)
        mep_size = simulation_ppd['obs'][0][0]
        response = mep_size
        responses.append(response)

        simulation_df_happened = {**{config.RESPONSE[ix]: np.array(responses)[:, ix] for ix in range(len(config.RESPONSE))}, **{'TMSInt': intensities}}
        simulation_df_happened = pd.DataFrame(simulation_df_happened)
        simulation_df_happened['participant___participant_condition'] = 0
        simulation_df_happened['participant'] = '0'
        simulation_df_happened['participant_condition'] = 'Uninjured'
        simulation_df_happened = simulation_df_happened.astype({"TMSInt": 'float64'})
        # Choose next intensity
        config.BUILD_DIR = root_dir / f"learn_posterior_rt{ix:03}"
        config_fast.BUILD_DIR = root_dir / f"learn_posterior_rt{ix:03}"
        model_hap, mcmc_hap, posterior_samples_hap = fit_new_model(config, simulation_df_happened, make_figures=make_figures_per_sample)

        if choose_interp:
            next_intensity = vec_intensity_lin[ix+1]
        elif ix < 1:
            next_intensity = range_max
        else:
            entropy_base = calculate_entropy(posterior_samples_hap, config, opt_param)

            list_candidate_intensities = range(range_min, range_max)
            # ix_start = ix % 2  # This is just subsampling the x to make things a bit faster...
            vec_candidate_int = np.array(list_candidate_intensities)

            simulation_df_future = pd.DataFrame({'TMSInt': vec_candidate_int})
            simulation_df_future['participant___participant_condition'] = 0
            simulation_df_future['participant'] = '0'
            simulation_df_future['participant_condition'] = 'Uninjured'
            simulation_df_future = simulation_df_future.astype({"TMSInt": 'float64'})
            # TODO: turn off mixture here if it is in the model
            posterior_predictive = model.predict(df=simulation_df_future,
                                                 posterior_samples=posterior_samples_hap)
            n_muscles = posterior_predictive['obs'].shape[-1]

            # pct_of_chain = np.linspace(0, 100, N_obs + 2)[1:-1]
            # candidate_y_at_this_int = np.percentile(posterior_predictive['obs'], pct_of_chain, axis=0)
            candidate_y_at_this_int = np.zeros((N_obs, len(vec_candidate_int), n_muscles))
            int_at_this_int = np.zeros((N_obs, len(vec_candidate_int), n_muscles))
            for ix_intensity in range(len(vec_candidate_int)):
                for ix_muscle in range(n_muscles):
                    samples = posterior_predictive['obs'][:, ix_intensity, ix_muscle]
                    # don't use min samples, to max because then you really need a very large N_obs
                    # perhaps the way this linspace is handled could be improved...
                    # or you could use percentile spacing, then correct for it in the integration (but was not obvious)
                    y_grid = np.linspace(np.percentile(samples, 2.5), np.percentile(samples, 97.5), N_obs)
                    candidate_y_at_this_int[:, ix_intensity, ix_muscle] = y_grid
                    int_at_this_int[:, ix_intensity, ix_muscle] = vec_candidate_int[ix_intensity]  # just so it can use the same reshape code next

            # vec_candidate_int_flattened = np.tile(vec_candidate_int[None, :, None], [N_obs, 1, 1]).reshape(-1, 1)
            vec_candidate_int_flattened = int_at_this_int.reshape(-1, n_muscles)
            candidate_y_at_this_int_flattened = candidate_y_at_this_int.reshape(-1, n_muscles)

            simulation_df_future = simulation_df_happened.copy()  # just an empty template
            # would be great to init the MCMC chains with the previous full fit (or fit of previous parallel op)
            if do_parallel:
                with Parallel(n_jobs=-1) as parallel:
                    entropy_list_flattened = parallel(
                        delayed(fit_lookahead_wrapper)(simulation_df_future,
                                                       vec_candidate_int_flattened[ix_sample],
                                                       candidate_y_at_this_int_flattened[ix_sample],
                                                       config_fast, opt_param)
                        for ix_sample in range(vec_candidate_int_flattened.shape[0])
                    )
            else:
                entropy_list_flattened = []
                for ix_sample in range(vec_candidate_int_flattened.shape[0]):
                    entropy_list_flattened.append(fit_lookahead_wrapper(simulation_df_future,
                                                   vec_candidate_int_flattened[ix_sample],
                                                   candidate_y_at_this_int_flattened[ix_sample],
                                                   config_fast, opt_param))

            op_shape = list(candidate_y_at_this_int.shape)
            # op_shape.append(n_muscles)
            entropy_list = np.array(entropy_list_flattened).reshape(op_shape)
            # Estimate the expected entropy E(H(x)) = SUM(H(y|x)p(y|x))dy
            mat_entropy = np.full((len(vec_candidate_int), n_muscles), np.nan)
            for ix_intensity in range(len(vec_candidate_int)):
                if np.any(np.isnan(entropy_list[:, ix_intensity, :])):
                    # think this is rare... but should keep an eye on this.
                    print(f'We have a NAN entropy at intenisty {ix_intensity}. Not considering it.')
                    continue
                for ix_muscle in range(n_muscles):
                    H_y = entropy_list[:, ix_intensity, ix_muscle]
                    y_grid = candidate_y_at_this_int[:, ix_intensity, ix_muscle]
                    samples = posterior_predictive['obs'][:, ix_intensity, ix_muscle]
                    kde = gaussian_kde(samples, bw_method='silverman')
                    p_y = kde(y_grid)
                    dy = np.median(np.diff(y_grid))
                    H_exp = np.sum(H_y * p_y) * dy
                    mat_entropy[ix_intensity, ix_muscle] = H_exp

            mat_entropy_diff = mat_entropy - entropy_base
            # collapse over muscles
            vec_entropy_diff = mat_entropy_diff.mean(axis=-1)

            ix_min_entropy = np.nanargmin(vec_entropy_diff)
            next_intensity = vec_candidate_int[ix_min_entropy]

            # save some stuff for later debugging
            if not choose_interp:
                with open(config.BUILD_DIR / 'entropy.pkl', "wb") as f:
                    pickle.dump((mat_entropy, entropy_base, next_intensity, vec_candidate_int, N_obs), f)

        intensities.append(next_intensity)

    intensity_final = intensities.pop()

    return


if __name__ == "__main__":
    main()
