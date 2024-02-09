import time
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, pmap, vmap

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import (
    NUTS, MCMC, SVI, Trace_ELBO, Predictive
)
from numpyro.diagnostics import hpdi

from hbmep.config import Config
from hbmep.model import GammaModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site

logger = logging.getLogger(__name__)

import jax
# PLATFORM = "cuda"
# jax.config.update("jax_platforms", PLATFORM)
# numpyro.set_platform(PLATFORM)
jax.config.update("jax_enable_x64", False)

class ReLU(GammaModel):
    NAME = "relu"

    def __init__(self, config: Config):
        super(ReLU, self).__init__(config=config)
        self.n_steps = 10000

    def _collect_regressor(self, df):
        features = df[self.features].to_numpy().T
        intensity = df[self.intensity].to_numpy().reshape(-1,)
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))
        return features, intensity

    def _collect_response(self, df):
        response = df[self.response].to_numpy()
        return response,

    def _model(self, features, intensity, response_obs=None):
        # features, n_features = features
        # intensity, n_data = intensity
        # intensity = intensity.reshape(-1, 1)
        # intensity = np.tile(intensity, (1, self.n_response))
        n_features = np.max(features, axis=1) + 1
        n_data = features.shape[1]
        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Hyper-priors """
            a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Priors """
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                L = numpyro.sample(site.L, dist.HalfNormal(L_scale))

                c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.relu(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    self.rate(
                        mu,
                        c_1[feature0],
                        c_2[feature0]
                    )
                )
                alpha = numpyro.deterministic(
                    site.alpha,
                    self.concentration(mu, beta)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )

    def run_inference(self, df):
        """ Set up sampler """
        sampler = NUTS(self._model)
        mcmc = MCMC(sampler, **self.mcmc_params, progress_bar=False)

        """ Run MCMC inference """
        logger.info(f"Running inference with {self.NAME} ...")
        start = time.time()
        mcmc.run(self.rng_key, *self._collect_regressor(df=df), *self._collect_response(df=df))
        end = time.time()
        time_taken = end - start
        time_taken = np.array(time_taken)
        posterior_samples = mcmc.get_samples()
        posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
        return mcmc, posterior_samples, time_taken

    def run_svi(self, df):
        optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
        self._guide = numpyro.infer.autoguide.AutoNormal(self._model)
        svi = SVI(
            self._model,
            self._guide,
            optimizer,
            loss=Trace_ELBO()
        )
        start = time.time()
        svi_result = svi.run(
            self.rng_key,
            self.n_steps,
            *self._collect_regressor(df=df),
            *self._collect_response(df=df),
            progress_bar=False
        )
        end = time.time()
        time_taken = end - start
        time_taken = np.array(time_taken)
        predictive = Predictive(
            self._guide,
            params=svi_result.params,
            num_samples=4000
        )
        posterior_samples = predictive(self.rng_key, *self._collect_regressor(df=df))
        posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
        return svi_result, posterior_samples, time_taken

    def run_svi_jit(self, df):
        optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
        self._guide = numpyro.infer.autoguide.AutoNormal(self._model)
        svi = SVI(
            self._model,
            self._guide,
            optimizer,
            loss=Trace_ELBO()
        )

        features, intensity, = self._collect_regressor(df=df)
        response_obs, = self._collect_response(df=df)

        @jit
        def svi_step(svi_state):
            svi_state, loss = svi.update(svi_state, features, intensity, response_obs=response_obs)
            return svi_state, loss

        svi_state = svi.init(
            self.rng_key,
            features, intensity,
            response_obs=response_obs
        )
        svi_state, loss = svi_step(svi_state)

        losses = [loss]
        start = time.time()
        for step in range(self.n_steps):
            svi_state, loss = svi_step(svi_state)
            losses.append(loss)
        end = time.time()
        time_taken = end - start
        time_taken = np.array(time_taken)

        predictive = Predictive(
            self._guide,
            params=svi.get_params(svi_state),
            num_samples=4000
        )
        posterior_samples = predictive(self.rng_key, *self._collect_regressor(df=df))
        posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
        return losses, posterior_samples, time_taken


class ActiveReLU(GammaModel):
    NAME = "active_relu"

    def __init__(self, config: Config):
        super(ActiveReLU, self).__init__(config=config)
        self.n_steps = 2000
        self.n_grid = 100

    def _collect_regressor(self, df):
        intensity = df[self.intensity].to_numpy().reshape(-1,)
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))
        intensity = intensity[..., None]
        return intensity,

    def _collect_response(self, df):
        response = df[self.response].to_numpy()
        response = response[..., None]
        return response,

    def render_recruitment_curves(
        self,
        intensity,
        response,
        intensity_pred,
        a = None,
        mu = None,
        obs = None,
        destination_path = None
    ):
        logger.info(intensity[-1, 0, :])
        # intensity (n_data, n_response, n_regressions)
        # response (n_data, n_response, n_regressions)
        # intensity_pred (n_grid, n_response, n_regressions)
        a_map = a.mean(axis=0)
        mu_map = mu.mean(axis=0)
        obs_map = obs.mean(axis=0)
        a_hpdi = hpdi(a, prob=.95)
        mu_hpdi = hpdi(mu, prob=.95)
        obs_hpdi = hpdi(obs, prob=.95)
        logger.info(f"intensity: {intensity.shape}, response: {response.shape}")
        logger.info(f"intensity_pred: {intensity_pred.shape}, a: {a.shape}")
        logger.info(f"mu: {mu.shape}, obs: {obs.shape}")
        logger.info(f"a_map: {a_map.shape}, mu_map: {mu_map.shape}, obs_map: {obs_map.shape}")
        logger.info(f"a_hpdi: {a_hpdi.shape}, mu_hpdi: {mu_hpdi.shape}, obs_hpdi: {obs_hpdi.shape}")

        n_regressions = obs.shape[-1]
        n_response = obs.shape[-2]

        nrows, ncols = n_regressions, 3 * n_response
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                ncols * self.subplot_cell_width,
                nrows * self.subplot_cell_height
            ),
            constrained_layout=True,
            squeeze=False
        )
        for regression_ind in range(n_regressions):
            j = 0
            for response_ind in range(n_response):
                response_color = self.response_colors[response_ind]
                ax = axes[regression_ind, j]
                sns.scatterplot(
                    x=intensity[:, response_ind, regression_ind],
                    y=response[:, response_ind, regression_ind],
                    color=response_color,
                    ax=ax
                )
                ax.plot(
                    intensity[:, response_ind, regression_ind][-1],
                    response[:, response_ind,regression_ind][-1],
                    'ro'
                )
                sns.lineplot(
                    x=intensity_pred[:, response_ind, regression_ind],
                    y=mu_map[:, response_ind, regression_ind],
                    color=response_color,
                    ax=ax
                )
                sns.kdeplot(
                    a[:, response_ind, regression_ind],
                    color="green",
                    ax=ax
                )
                ax.sharex(axes[0, 0])
                j += 1

                ax = axes[regression_ind, j]
                sns.scatterplot(
                    x=intensity[:, response_ind, regression_ind],
                    y=response[:, response_ind, regression_ind],
                    color=response_color,
                    ax=ax
                )
                ax.plot(
                    intensity[:, response_ind,
                    regression_ind][-1],response[:, response_ind,regression_ind][-1],
                    'ro'
                )
                sns.lineplot(
                    x=intensity_pred[:, response_ind, regression_ind],
                    y=obs_map[:, response_ind, regression_ind],
                    color=response_color,
                    ax=ax
                )
                ax.fill_between(
                    intensity_pred[:, response_ind, regression_ind],
                    obs_hpdi[0, :, response_ind, regression_ind],
                    obs_hpdi[1, :, response_ind, regression_ind],
                    color="C0",
                    alpha=.4
                )
                ax.sharex(axes[0, 0])
                ax.sharey(axes[response_ind, j - 1])
                j += 1

                ax = axes[regression_ind, j]
                sns.kdeplot(
                    a[:, response_ind, regression_ind],
                    color="green",
                    ax=ax
                )
                ax.axvline(
                    a_map[response_ind, regression_ind],
                    color="k",
                    linestyle="--"
                )
                ax.axvline(
                    a_hpdi[0, response_ind, regression_ind],
                    color="green",
                    linestyle="--"
                )
                ax.axvline(
                    a_hpdi[1, response_ind, regression_ind],
                    color="green",
                    linestyle="--"
                )
                j += 1
            axes[regression_ind, 0].set_ylim(bottom=-.2)

        fig.savefig(destination_path)
        logger.info(f"Saved to {destination_path}")
        return

    def make_prediction_dataset(
        self, df: pd.DataFrame,
        num: int = 100,
        min_intensity: float | None = None,
        max_intensity: float | None = None
    ):
        pred_df = (
            pd.DataFrame(
                np.linspace(0, 100, num),
                columns=[self.intensity]
            )
        )
        return pred_df

    # def _model(self, intensity, response_obs=None):
    #     # intensity (n_data, n_response)
    #     # response_obs (n_data, n_response)
    #     n_data = intensity.shape[0]

    #     with numpyro.plate(site.n_response, self.n_response):
    #         """ Hyper-priors """
    #         a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
    #         a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

    #         b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
    #         L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))

    #         c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
    #         c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

    #         """ Priors """
    #         a = numpyro.sample(
    #             site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
    #         )

    #         b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
    #         L = numpyro.sample(site.L, dist.HalfNormal(L_scale))

    #         c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
    #         c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

    #     with numpyro.plate(site.n_response, self.n_response):
    #         with numpyro.plate(site.n_data, n_data):
    #             """ Model """
    #             mu = numpyro.deterministic(
    #                 site.mu,
    #                 F.relu(
    #                     x=intensity,
    #                     a=a,
    #                     b=b,
    #                     L=L,
    #                 )
    #             )
    #             beta = numpyro.deterministic(
    #                 site.beta,
    #                 self.rate(
    #                     mu,
    #                     c_1,
    #                     c_2,
    #                 )
    #             )
    #             alpha = numpyro.deterministic(
    #                 site.alpha,
    #                 self.concentration(mu, beta)
    #             )

    #             """ Observation """
    #             numpyro.sample(
    #                 site.obs,
    #                 dist.Gamma(concentration=alpha, rate=beta),
    #                 obs=response_obs
    #             )

    def _model(self, intensity, response_obs=None):
        # intensity (n_data, n_response, n_regressions)
        # response_obs (n_data, n_response, n_regressions)
        n_data = intensity.shape[0]
        n_regressions = intensity.shape[-1]

        with numpyro.plate("n_regressions", n_regressions):
            with numpyro.plate(site.n_response, self.n_response):
                """ Hyper-priors """
                a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

                b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
                L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

                """ Priors """
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                L = numpyro.sample(site.L, dist.HalfNormal(L_scale))

                c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate("n_regressions", n_regressions):
            with numpyro.plate(site.n_response, self.n_response):
                with numpyro.plate(site.n_data, n_data):
                    """ Model """
                    mu = numpyro.deterministic(
                        site.mu,
                        F.relu(
                            x=intensity,
                            a=a,
                            b=b,
                            L=L,
                        )
                    )
                    beta = numpyro.deterministic(
                        site.beta,
                        self.rate(
                            mu,
                            c_1,
                            c_2,
                        )
                    )
                    alpha = numpyro.deterministic(
                        site.alpha,
                        self.concentration(mu, beta)
                    )

                    """ Observation """
                    numpyro.sample(
                        site.obs,
                        dist.Gamma(concentration=alpha, rate=beta),
                        obs=response_obs
                    )

    def run_inference(self, intensity, response_obs):
        """ Set up sampler """
        sampler = NUTS(self._model)
        mcmc = MCMC(sampler, **self.mcmc_params, progress_bar=False)

        """ Run MCMC inference """
        logger.info(f"Running inference with {self.NAME} ...")
        start = time.time()
        mcmc.run(self.rng_key, intensity, response_obs)
        end = time.time()
        time_taken = end - start
        time_taken = np.array(time_taken)
        posterior_samples = mcmc.get_samples()
        posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
        return posterior_samples, time_taken

    def run_svi(self, intensity, response_obs):
        logger.info(f"Running SVI ...")
        logger.info(f"intensity: {intensity.shape}, response_obs: {response_obs.shape}")
        optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
        _guide = numpyro.infer.autoguide.AutoNormal(self._model)
        svi = SVI(
            self._model,
            _guide,
            optimizer,
            loss=Trace_ELBO()
        )

        @jit
        def svi_step(svi_state):
            svi_state, loss = svi.update(svi_state, intensity, response_obs=response_obs)
            return svi_state, loss

        svi_state = svi.init(
            self.rng_key,
            intensity,
            response_obs=response_obs
        )
        svi_state, _ = svi_step(svi_state)

        start = time.time()
        for step in range(self.n_steps):
            svi_state, _ = svi_step(svi_state)
        end = time.time()
        time_taken = end - start
        time_taken = np.array(time_taken)

        predictive = Predictive(
            _guide,
            params=svi.get_params(svi_state),
            num_samples=4000
        )
        posterior_samples = predictive(self.rng_key, intensity)
        posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
        return posterior_samples, time_taken


import os
import gc
import pickle
from constants import REP, TOML_PATH
from hbmep_paper.utils import setup_logging
from utils import generate_nested_pulses
SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-active-learn/simulate_data"
SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PPD_PATH = os.path.join(SIMULATION_DIR, "simulation_ppd.pkl")
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-active-learn/simulate_data/relu-active"


def run_inference(df, model):
    intensity, = model._collect_regressor(df=df)    # (n_data, n_response, 1)
    response_obs, = model._collect_response(df=df)  # (n_data, n_response, 1)
    posterior_samples, time_taken = model.run_svi(intensity, response_obs)
    for u, v in posterior_samples.items():
        logger.info(f"{u}: {v.shape}")
    logger.info(f"time_taken: {time_taken}")

    prediction_df = model.make_prediction_dataset(df=df, num=model.n_grid, min_intensity=0, max_intensity=99)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    for u, v in posterior_predictive.items():
        logger.info(f"{u}: {v.shape}")

    intensity, = model._collect_regressor(df=df)    # (n_data, n_response, 1)
    response, = model._collect_response(df=df)  # (n_data, n_response, 1)
    intensity_pred, = model._collect_regressor(df=prediction_df)     # (n_grid, n_response, 1)
    a = posterior_samples[site.a]        # (n_samples, n_data, n_response, 1)
    mu = posterior_predictive[site.mu]  # (n_samples, n_grid, n_response, 1)
    obs = posterior_predictive[site.obs]    # (n_samples, n_data, n_response, 1)
    logger.info(f"intensity: {intensity.shape}, response: {response.shape}")
    logger.info(f"intensity_pred: {intensity_pred.shape}, a: {a.shape}")
    logger.info(f"mu: {mu.shape}, obs: {obs.shape}")
    dest = os.path.join(model.build_dir, "recruitment_curves.png")
    model.render_recruitment_curves(
        intensity=intensity,
        response=response,
        intensity_pred=intensity_pred,
        a=a,
        mu=mu,
        obs=obs,
        destination_path=dest
    )

    intensity_new = intensity_pred.reshape(-1,)
    intensity_new = np.append(
        np.tile(intensity, (1, 1, model.n_grid)),
        intensity_new[None, None, :],
        axis=0
    )
    intensity_new = intensity_new[None, ...]
    intensity_new = np.tile(intensity_new, (obs.shape[0], 1, 1, 1))
    logger.info(f"intensity_new: {intensity_new.shape}")
    # logger.info(intensity_new[-1, 0, :])

    response_obs_new = response[None, ...]
    response_obs_new = np.tile(response_obs_new, (obs.shape[0], 1, 1, model.n_grid))
    response_obs_new = np.append(
        response_obs_new,
        np.swapaxes(obs, 1, -1),
        axis=1
    )
    logger.info(f"response_obs_new: {response_obs_new.shape}")

    times = []
    draw_ind = 0
    x = intensity_new[draw_ind, ...]
    y = response_obs_new[draw_ind, ...]
    logger.info(f"x: {x.shape}, y: {y.shape}")

    posterior_samples, time_taken = model.run_inference(x, y)
    for u, v in posterior_samples.items():
        logger.info(f"{u}: {v.shape}")
    logger.info(f"time_taken: {time_taken}")
    return

    posterior_samples, time_taken = model.run_svi(x, y)
    for u, v in posterior_samples.items():
        logger.info(f"{u}: {v.shape}")
    logger.info(f"time_taken: {time_taken}")



    prediction_df = model.make_prediction_dataset(df=df, num=model.n_grid, min_intensity=0, max_intensity=99)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    for u, v in posterior_predictive.items():
        logger.info(f"{u}: {v.shape}")

    n_reg_to_plot = 10
    a = posterior_samples[site.a]        # (n_samples, n_data, n_response, 1)
    mu = posterior_predictive[site.mu]  # (n_samples, n_grid, n_response, 1)
    obs = posterior_predictive[site.obs]    # (n_samples, n_data, n_response, 1)
    intensity = x[..., ::n_reg_to_plot]
    response = y[..., ::n_reg_to_plot]
    # logger.info(intensity[-1, 0, :])

    intensity_pred, = model._collect_regressor(df=prediction_df)
    intensity_pred = np.tile(intensity_pred, (1, 1, n_reg_to_plot))
    a = a[..., ::n_reg_to_plot]
    mu = mu[..., ::n_reg_to_plot]
    obs = obs[..., ::n_reg_to_plot]
    logger.info(f"intensity: {intensity.shape}, response: {response.shape}")
    logger.info(f"intensity_pred: {intensity_pred.shape}, a: {a.shape}")
    logger.info(f"mu: {mu.shape}, obs: {obs.shape}")
    dest = os.path.join(model.build_dir, f"draw_{draw_ind}.png")
    model.render_recruitment_curves(
        intensity=intensity,
        response=response,
        intensity_pred=intensity_pred,
        a=a,
        mu=mu,
        obs=obs,
        destination_path=dest
    )

    # prediction_df = model.make_prediction_dataset(df=df, num=model.n_grid)
    # posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    # for u, v in posterior_predictive.items():
    #     logger.info(f"{u}: {v.shape}")
    return

def main():
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    n_subjects, n_reps, n_pulses = 1, 1, 48
    pulses_map = generate_nested_pulses(simulator, simulation_df)
    ind = \
        (simulation_df[simulator.features[0]] < n_subjects) & \
        (simulation_df[REP] < n_reps) & \
        (simulation_df[simulator.intensity].isin(pulses_map[n_pulses][::10]))

    simulation_df = simulation_df[ind].reset_index(drop=True).copy()
    ppd_a = simulation_ppd[site.a][:, 0, :]
    ppd_obs = simulation_ppd[site.obs][:, ind, :]
    simulation_df[simulator.response[0]] = ppd_obs[0, ...]

    simulation_ppd = None
    del simulation_ppd
    gc.collect()

    os.makedirs(BUILD_DIR, exist_ok=True)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    logger.info(f"Simulation dataframe shape: {simulation_df.shape}")
    logger.info(f"Simulation ppd shape: {ppd_a.shape}, {ppd_obs.shape}")

    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR
    config.FEATURES = []
    model = ActiveReLU(config=config)
    run_inference(df=simulation_df, model=model)
    return


if __name__ == "__main__":
    # import jax
    # print(jax.device_count())
    # print(jax.devices())
    # n_devices = jax.local_device_count()
    # print(n_devices)
    main()
