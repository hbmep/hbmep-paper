import os
import pickle
import logging
import multiprocessing
from pathlib import Path

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import arviz as az
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class LearnPosterior(BaseModel):
    NAME = "learn_posterior"

    def __init__(self, config: Config):
        super(LearnPosterior, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)
        n_fixed = 1

        """ Fixed Effects (Baseline) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample("a_fixed_mean", dist.TruncatedNormal(50., 20., low=0))
                a_fixed_scale = numpyro.sample("a_fixed_scale", dist.HalfNormal(30.))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(a_fixed_mean, a_fixed_scale, low=0)
                    )

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(10))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5))

            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(ell_scale_global_scale))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.deterministic(
                        site.a, a_fixed
                    )

                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                    v = numpyro.sample(site.v, dist.HalfNormal(v_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        v=v[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    c_1[feature0, feature1] + jnp.true_divide(c_2[feature0, feature1], mu)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta),
                    obs=response_obs
                )


def main():
    toml_path = "/home/vishu/repos/hbmep-paper/configs/experiments/subjects.toml"
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "learn-posterior")

    model = LearnPosterior(config=config)
    model._make_dir(model.build_dir)
    dest = os.path.join(model.build_dir, "log.log")
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )

    # src = "/home/vishu/data/hbmep-processed/human/tms/data_pkpk_auc.csv"
    src = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"
    df = pd.read_csv(src)
    # df[model.response[0]] *= 1000
    df[model.features[1]] = 0
    df, encoder_dict = model.load(df=df)

    mcmc, posterior_samples = model.run_inference(df=df)

    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

    dest = os.path.join(model.build_dir, "inference.pkl")
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "inference.nc")
    az.to_netcdf(mcmc, dest)
    logger.info(dest)

    numpyro_data = az.from_numpyro(mcmc)
    """ Model evaluation """
    logger.info("Evaluating model ...")
    score = az.loo(numpyro_data, var_name=site.obs)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    score = az.waic(numpyro_data, var_name=site.obs)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")


if __name__ == "__main__":
    main()
