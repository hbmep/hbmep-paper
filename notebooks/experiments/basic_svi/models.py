import logging

import numpy as np
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO, Predictive

from hbmep.config import Config
from hbmep.model import GammaModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site

logger = logging.getLogger(__name__)


class ReLU(GammaModel):
    NAME = "relu"

    def __init__(self, config: Config):
        super(ReLU, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            # Hyper Priors
            a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(150., 100., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[0], n_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                L = numpyro.sample(site.L, dist.HalfNormal(L_scale))

                c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
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

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


    def run_svi(self, df):
        # optimizer = numpyro.optim.Adam(step_size=0.0005)
        optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
        self._guide = numpyro.infer.autoguide.AutoNormal(self._model)
        svi = SVI(
            self._model,
            self._guide,
            optimizer,
            loss=Trace_ELBO()
        )
        svi_result = svi.run(
            self.rng_key,
            2000,
            *self._collect_regressor(df=df),
            *self._collect_response(df=df)
        )
        predictive = Predictive(self._guide, params=svi_result.params, num_samples=4000)
        posterior_samples = predictive(self.rng_key, *self._collect_regressor(df=df))
        posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
        return svi_result, posterior_samples
