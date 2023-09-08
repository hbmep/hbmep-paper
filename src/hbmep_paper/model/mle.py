import logging

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from hbmep.config import Config
from hbmep.model import Baseline
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils.constants import MLE

logger = logging.getLogger(__name__)


class MaximumLikelihood(Baseline):
    LINK = MLE

    def __init__(self, config: Config):
        super(MaximumLikelihood, self).__init__(config=config)

    def _model(self, subject, features, intensity, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_data = intensity.shape[0]
        n_subject = np.unique(subject).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate(site.n_subject, n_subject, dim=-2):
                with numpyro.plate("n_feature0", n_feature0, dim=-3):
                    """ Priors """
                    a = numpyro.sample(
                        site.a,
                        dist.TruncatedNormal(150, 500, low=0)
                    )
                    b = numpyro.sample(site.b, dist.HalfNormal(100))

                    L = numpyro.sample(site.L, dist.HalfNormal(0.05))
                    H = numpyro.sample(site.H, dist.HalfNormal(100))
                    v = numpyro.sample(site.v, dist.HalfNormal(200))

                    g_1 = numpyro.sample(site.g_1, dist.Exponential(0.01))
                    g_2 = numpyro.sample(site.g_2, dist.Exponential(0.01))

        """ Model """
        mu = numpyro.deterministic(
            site.mu,
            L[feature0, subject]
            + jnp.maximum(
                0,
                -1
                + (H[feature0, subject] + 1)
                / jnp.power(
                    1
                    + (jnp.power(1 + H[feature0, subject], v[feature0, subject]) - 1)
                    * jnp.exp(-b[feature0, subject] * (intensity - a[feature0, subject])),
                    1 / v[feature0, subject]
                )
            )
        )
        beta = numpyro.deterministic(
            site.beta,
            g_1[feature0, subject] + g_2[feature0, subject] * (1 / mu)
        )

        """ Observation """
        with numpyro.plate(site.data, n_data):
            return numpyro.sample(
                site.obs,
                dist.Gamma(concentration=mu * beta, rate=beta).to_event(1),
                obs=response_obs
            )

    @timing
    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """ Set up NUTS sampler """
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, **self.mcmc_params)
        rng_key = jax.random.PRNGKey(self.random_state)

        """ MCMC inference """
        logger.info(f"Running inference with {self.LINK} ...")
        mcmc.run(rng_key, *self._collect_regressors(df=df), *self._collect_response(df=df))
        posterior_samples = mcmc.get_samples()
        return mcmc, posterior_samples
