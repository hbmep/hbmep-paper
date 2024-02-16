import time
import logging

import numpy as np
import jax.numpy as jnp
from jax import jit, lax

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import (
    NUTS, MCMC, SVI, Trace_ELBO, Predictive
)

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


class OnlineReLU(GammaModel):
    NAME = "online_relu"

    def __init__(self, config: Config):
        super(OnlineReLU, self).__init__(config=config)
        self.n_steps = 2000
        self.n_grid = 100

    def _collect_regressor(self, df):
        features = df[self.features].to_numpy().T
        intensity = df[self.intensity].to_numpy().reshape(-1,)
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))
        return features, intensity

    def _collect_response(self, df):
        response = df[self.response].to_numpy()
        return response,

    def _model(self, intensity, response_obs=None):
        logger.info(intensity.shape)
        n_data = intensity.shape[0]
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

    # def run_inference(self, df):
    #     """ Set up sampler """
    #     sampler = NUTS(self._model)
    #     mcmc = MCMC(sampler, **self.mcmc_params, progress_bar=False)

    #     """ Run MCMC inference """
    #     logger.info(f"Running inference with {self.NAME} ...")
    #     start = time.time()
    #     mcmc.run(self.rng_key, *self._collect_regressor(df=df), *self._collect_response(df=df))
    #     end = time.time()
    #     time_taken = end - start
    #     time_taken = np.array(time_taken)
    #     posterior_samples = mcmc.get_samples()
    #     posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
    #     return mcmc, posterior_samples, time_taken

    # def run_svi(self, df):
    #     optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
    #     self._guide = numpyro.infer.autoguide.AutoNormal(self._model)
    #     svi = SVI(
    #         self._model,
    #         self._guide,
    #         optimizer,
    #         loss=Trace_ELBO()
    #     )
    #     start = time.time()
    #     svi_result = svi.run(
    #         self.rng_key,
    #         self.n_steps,
    #         *self._collect_regressor(df=df),
    #         *self._collect_response(df=df),
    #         progress_bar=False
    #     )
    #     end = time.time()
    #     time_taken = end - start
    #     time_taken = np.array(time_taken)
    #     predictive = Predictive(
    #         self._guide,
    #         params=svi_result.params,
    #         num_samples=4000
    #     )
    #     posterior_samples = predictive(self.rng_key, *self._collect_regressor(df=df))
    #     posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
    #     return svi_result, posterior_samples, time_taken

    def run_online(self, df):
        features, intensity, = self._collect_regressor(df=df)
        response_obs, = self._collect_response(df=df)

        # Fit on observed data

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
