import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import (
    Simulator,
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel
)
from constants import (
    TOML_PATH,
    REP,
    SIMULATION_DF,
    INFERENCE_FILE,
    TOTAL_REPS,
    TOTAL_PULSES,
    SIMULATE_DATA_DIR,
    SIMULATE_DATA_NO_EFFECT_DIR,
    EXPERIMENTS_DIR,
    EXPERIMENTS_NO_EFFECT_DIR
)

logger = logging.getLogger(__name__)

N_REPS = TOTAL_REPS
N_PULSES = TOTAL_PULSES


import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from hbmep.model import GammaModel
from hbmep.nn import functional as F


class TestModel(GammaModel):
    def __init__(self, config: Config, a_random_mean_scale, a_random_scale_scale):
        super(TestModel, self).__init__(config=config)
        self.a_random_mean_scale = a_random_mean_scale
        self.a_random_scale_scale = a_random_scale_scale

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_random = n_features[1] - 1

        # Fixed Effects (Baseline)
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample(
                    "a_fixed_mean", dist.TruncatedNormal(50., 20., low=0)
                )
                a_fixed_scale = numpyro.sample(
                    "a_fixed_scale", dist.HalfNormal(30.)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_mean, a_fixed_scale, low=0
                        )
                    )

        # Random Effects (Delta)
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_random", n_random):
                a_random_mean = numpyro.sample(
                    "a_random_mean", dist.Normal(0., self.a_random_mean_scale)
                )
                a_random_scale = numpyro.sample(
                    "a_random_scale", dist.HalfNormal(self.a_random_scale_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # a_random = numpyro.sample(
                    #     "a_random", dist.StudentT(2, a_random_mean, a_random_scale)
                    # )
                    a_random = numpyro.sample(
                        "a_random", dist.Normal(a_random_mean, a_random_scale)
                    )

                    # Penalty for negative a
                    penalty_for_negative_a = (
                        jnp.fabs(a_fixed + a_random) - (a_fixed + a_random)
                    )
                    numpyro.factor(
                        "penalty_for_negative_a", -penalty_for_negative_a
                    )

        with numpyro.plate(site.n_response, self.n_response):
            # Global Priors
            b_scale_global_scale = numpyro.sample(
                "b_scale_global_scale", dist.HalfNormal(5.)
            )

            L_scale_global_scale = numpyro.sample(
                "L_scale_global_scale", dist.HalfNormal(.5)
            )
            ell_scale_global_scale = numpyro.sample(
                "ell_scale_global_scale", dist.HalfNormal(10.)
            )
            H_scale_global_scale = numpyro.sample(
                "H_scale_global_scale", dist.HalfNormal(5.)
            )

            c_1_scale_global_scale = numpyro.sample(
                "c_1_scale_global_scale", dist.HalfNormal(5.)
            )
            c_2_scale_global_scale = numpyro.sample(
                "c_2_scale_global_scale", dist.HalfNormal(5.)
            )

            with numpyro.plate(site.n_features[1], n_features[1]):
                # Hyper-priors
                b_scale = numpyro.sample(
                    "b_scale", dist.HalfNormal(b_scale_global_scale)
                )

                L_scale = numpyro.sample(
                    "L_scale", dist.HalfNormal(L_scale_global_scale)
                )
                ell_scale = numpyro.sample(
                    "ell_scale", dist.HalfNormal(ell_scale_global_scale)
                )
                H_scale = numpyro.sample(
                    "H_scale", dist.HalfNormal(H_scale_global_scale)
                )

                c_1_scale = numpyro.sample(
                    "c_1_scale", dist.HalfNormal(c_1_scale_global_scale)
                )
                c_2_scale = numpyro.sample(
                    "c_2_scale", dist.HalfNormal(c_2_scale_global_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=1)
                    )

                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    self.rate(
                        mu,
                        c_1[feature0, feature1],
                        c_2[feature0, feature1]
                    )
                )
                alpha = numpyro.deterministic(
                    site.alpha,
                    self.concentration(mu, beta)
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


@timing
def main(simulation_data_dir, build_dir):
    # Load simulated dataframe
    src = os.path.join(simulation_data_dir, SIMULATION_DF)
    simulation_df = pd.read_csv(src)

    # Load simulation ppd
    src = os.path.join(simulation_data_dir, INFERENCE_FILE)
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    ppd_a = simulation_ppd[site.a]
    ppd_obs = simulation_ppd[site.obs]
    simulation_ppd = None
    del simulation_ppd
    gc.collect()

    # Set up logging
    simulator._make_dir(build_dir)
    setup_logging(
        dir=build_dir,
        fname=os.path.basename(__file__)
    )


    # Define experiment
    def run_experiment(
        draw,
        n_subjects,
        M,
        a_random_mean_scale=20.,
        a_random_scale_scale=30.
    ):
        n_reps = 1

        # Required for build directory
        draw_dir = f"d{draw}"
        n_subjects_dir = f"n{n_subjects}"
        a_random_dir = f"ar__{a_random_mean_scale}__{a_random_scale_scale}"

        # Load data
        ind = (
            (simulation_df[simulator.features[0]] < n_subjects) &
            (simulation_df[REP] < n_reps)
        )
        df = simulation_df[ind].reset_index(drop=True).copy()
        df[simulator.response[0]] = ppd_obs[draw, ind, 0]

        ind = df[simulator.response[0]] > 0
        df = df[ind].reset_index(drop=True).copy()

        # Build model
        config = Config(toml_path=TOML_PATH)
        config.BUILD_DIR = os.path.join(
            build_dir,
            "pooling_strength",
            M.NAME,
            draw_dir,
            n_subjects_dir,
            a_random_dir
        )
        model = M(config=config, a_random_mean_scale=a_random_mean_scale, a_random_scale_scale=a_random_scale_scale)

        # Set up logging
        model._make_dir(model.build_dir)
        setup_logging(
            dir=model.build_dir,
            fname="logs"
        )

        # Run inference
        df, encoder_dict = model.load(df=df)
        _, posterior_samples = model.run_inference(df=df)

        # Predictions and recruitment curves
        prediction_df = model.make_prediction_dataset(df=df)
        posterior_predictive = model.predict(
            df=prediction_df, posterior_samples=posterior_samples
        )
        model.render_recruitment_curves(
            df=df,
            encoder_dict=encoder_dict,
            posterior_samples=posterior_samples,
            prediction_df=prediction_df,
            posterior_predictive=posterior_predictive
        )

        # Compute error and save results
        a_true = ppd_a[draw, :n_subjects, ...]
        a_pred = posterior_samples[site.a]
        assert a_pred.mean(axis=0).shape == a_true.shape
        np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
        np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

        a_random_mean = posterior_samples["a_random_mean"]
        a_random_scale = posterior_samples["a_random_scale"]
        np.save(os.path.join(model.build_dir, "a_random_mean.npy"), a_random_mean)
        np.save(os.path.join(model.build_dir, "a_random_scale.npy"), a_random_scale)

        prob = (a_random_mean > 0).mean()
        logger.info(f"{1 - prob:.2f}")

        config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
        model, posterior_samples, posterior_predictive = None, None, None
        a_true, a_pred = None, None
        a_random_mean, a_random_scale = None, None
        del config, df, prediction_df, encoder_dict, _
        del model, posterior_samples, posterior_predictive
        del a_true, a_pred
        del a_random_mean, a_random_scale
        gc.collect()

        return


    # Experiment space
    draws_space = range(1)
    n_jobs = -1

    M = TestModel
    N_SUB = 4
    # ar_space = [(u, 30) for u in [1, 5, 10, 20, 30, 40, 50, 60]]
    ar_space = [(20, u) for u in [1, 5, 10, 20, 30, 40, 50, 60]]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(
                draw, N_SUB, M, ar[0], ar[1]
            )
            for draw in draws_space
            for ar in ar_space
        )

    # run_experiment(0, 4, TestModel)
    return


if __name__ == "__main__":
    # Run for simulation with effect
    main(SIMULATE_DATA_DIR, EXPERIMENTS_DIR)

    # # Run for simulation without effect
    # main(SIMULATE_DATA_NO_EFFECT_DIR, EXPERIMENTS_NO_EFFECT_DIR)
