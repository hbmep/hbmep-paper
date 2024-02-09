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
        self.n_grid = 10

    def _collect_regressor(self, df):
        intensity = df[self.intensity].to_numpy().reshape(-1,)
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))
        return intensity,

    def _collect_response(self, df):
        response = df[self.response].to_numpy()
        return response,

    def render_recruitment_curves(
        self,
        df,
        encoder_dict = None,
        posterior_samples = None,
        prediction_df = None,
        posterior_predictive = None,
        destination_path = None
    ):
        a = posterior_samples[site.a]
        mu = posterior_predictive[site.mu]
        obs = posterior_predictive[site.obs]
        a_map = a.mean(axis=0)
        mu_map = mu.mean(axis=0)
        obs_map = obs.mean(axis=0)
        a_hpdi = hpdi(a, prob=.95)
        mu_hpdi = hpdi(mu, prob=.95)
        obs_hpdi = hpdi(obs, prob=.95)

        nrows, ncols = self.n_response, 3
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
        for response_ind, response in enumerate(self.response):
            response_color = self.response_colors[response_ind]
            ax = axes[response_ind, 0]
            sns.scatterplot(
                data=df,
                x=self.intensity,
                y=response,
                color=response_color,
                ax=ax
            )
            sns.lineplot(
                data=prediction_df,
                x=self.intensity,
                y=mu_map[:, response_ind],
                color=response_color,
                ax=ax
            )
            sns.kdeplot(
                a[:, response_ind],
                color="green",
                ax=ax
            )
            ax.sharex(axes[0, 0])

            ax = axes[response_ind, 1]
            sns.scatterplot(
                data=df,
                x=self.intensity,
                y=response,
                color=response_color,
                ax=ax
            )
            sns.lineplot(
                data=prediction_df,
                x=self.intensity,
                y=obs_map[:, response_ind],
                color=response_color,
                ax=ax
            )
            ax.fill_between(
                prediction_df[self.intensity].values,
                obs_hpdi[0, :, response_ind],
                obs_hpdi[1, :, response_ind],
                color="C0",
                alpha=.4
            )
            ax.sharex(axes[0, 0])
            ax.sharey(axes[response_ind, 0])

            ax = axes[response_ind, 2]
            sns.kdeplot(
                a[:, response_ind],
                color="green",
                ax=ax
            )
            ax.axvline(
                a_map[response_ind],
                color="k",
                linestyle="--"
            )
            ax.axvline(
                a_hpdi[0, response_ind],
                color="green",
                linestyle="--"
            )
            ax.axvline(
                a_hpdi[1, response_ind],
                color="green",
                linestyle="--"
            )
        axes[0, 0].set_ylim(bottom=-.001)
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
        logger.info(pred_df.head())
        return pred_df

    def _model(self, intensity, response_obs=None):
        # intensity (n_data, n_response)
        # response_obs (n_data, n_response)
        n_data = intensity.shape[0]

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
            _guide,
            params=svi.get_params(svi_state),
            num_samples=4000
        )
        posterior_samples = predictive(self.rng_key, intensity)
        # posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
        return posterior_samples


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
    intensity, = model._collect_regressor(df=df)    # (n_data, n_response)
    response_obs, = model._collect_response(df=df)  # (n_data, n_response)
    posterior_samples = model.run_svi(intensity, response_obs)

    prediction_df = model.make_prediction_dataset(df=df, num=model.n_grid)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    for u, v in posterior_predictive.items():
        logger.info(f"{u}: {v.shape}")
    dest = os.path.join(model.build_dir, "recruitment_curves.png")
    model.render_recruitment_curves(
        df=df, posterior_samples=posterior_samples,
        prediction_df=prediction_df, posterior_predictive=posterior_predictive,
        destination_path=dest
    )

    obs = posterior_predictive[site.obs]    # (n_samples, n_data, n_response)
    intensity = intensity[None, None, ...]  # (1, 1, n_data, n_response)
    intensity = np.tile(intensity, (obs.shape[0], model.n_grid, 1, 1))  # (n_samples, n_grid, n_data, n_response)

    grid, = model._collect_regressor(df=prediction_df)  # (n_grid, n_response)
    grid = grid[None, :, None, :]   # (1, n_grid, 1, n_response)
    grid = np.tile(grid, (obs.shape[0], 1, 1, 1))  # (n_samples, n_grid, 1, n_response)
    logger.info(f"intensity: {intensity.shape}, grid: {grid.shape}")

    response_obs = response_obs[None, None, ...]  # (1, 1, n_data, n_response)
    response_obs = np.tile(response_obs, (obs.shape[0], model.n_grid, 1, 1))  # (n_samples, n_grid, n_data, n_response)

    obs = obs[..., None, :] # (n_samples, n_data, 1, n_response)

    intensity_concat = np.append(intensity, grid, axis=-2)  # (n_samples, n_grid, n_data + 1, n_response)
    response_obs_concat = np.append(response_obs, obs, axis=-2)  # (n_samples, n_grid, n_data + 1, n_response
    logger.info(f"intensity_concat: {intensity_concat.shape}")
    logger.info(f"response_obs_concat: {response_obs_concat.shape}")

    intensity_concat = intensity_concat[0, ...]
    response_obs_concat = response_obs_concat[0, ...]
    logger.info(f"intensity_concat: {intensity_concat.shape}")
    logger.info(f"response_obs_concat: {response_obs_concat.shape}")

    intensity_concat = intensity_concat[0:1, ...]
    response_obs_concat = response_obs_concat[0:1, ...]
    logger.info(f"intensity_concat: {intensity_concat.shape}")
    logger.info(f"response_obs_concat: {response_obs_concat.shape}")

    model.run_svi(intensity_concat, response_obs_concat)
    # losses, posterior_samples, time_taken = pmap(
    #     model.run_svi,
    #     in_axes=0,
    # )(intensity_concat, intensity_concat)
    # posterior_samples = pmap(
    #     model.run_svi,
    #     in_axes=0,
    # )(intensity_concat, intensity_concat)
    posterior_samples = vmap(
        model.run_svi,
        in_axes=0,
    )(intensity_concat, intensity_concat)
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
        (simulation_df[simulator.intensity].isin(pulses_map[n_pulses]))

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
    main()
