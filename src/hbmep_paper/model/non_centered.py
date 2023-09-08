import logging

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
from numpyro.distributions.transforms import AffineTransform, AbsTransform

from hbmep.config import Config
from hbmep.model import Baseline
from hbmep.model.utils import Site as site

logger = logging.getLogger(__name__)


class NonCenteredHierarchicalBayesian(Baseline):
    LINK = "non_centered_hbm"

    def __init__(self, config: Config):
        super(NonCenteredHierarchicalBayesian, self).__init__(config=config)

    def _model(self, subject, features, intensity, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_data = intensity.shape[0]
        n_subject = np.unique(subject).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        hyperpriors_reparam_config = {
            var: TransformReparam() for var in [site.a, site.b, site.L, site.H, site.v]
        }

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate(site.n_subject, n_subject, dim=-2):
                """ Hyper-priors """
                mu_a = numpyro.sample(
                    site.mu_a,
                    dist.TruncatedNormal(150, 50, low=0)
                )
                sigma_a = numpyro.sample(site.sigma_a, dist.HalfNormal(50))

                sigma_b = numpyro.sample(site.sigma_b, dist.HalfNormal(0.1))

                sigma_L = numpyro.sample(site.sigma_L, dist.HalfNormal(0.05))
                sigma_H = numpyro.sample(site.sigma_H, dist.HalfNormal(5))
                sigma_v = numpyro.sample(site.sigma_v, dist.HalfNormal(10))

                with numpyro.plate("n_feature0", n_feature0, dim=-3):
                    """ Priors """
                    with numpyro.handlers.reparam(config=hyperpriors_reparam_config):
                        a = numpyro.sample(
                            site.a,
                            dist.TransformedDistribution(
                                dist.Normal(0, 1),
                                [AffineTransform(mu_a, sigma_a), AbsTransform()]
                            )
                        )
                        b = numpyro.sample(
                            site.b,
                            dist.TransformedDistribution(
                                dist.Normal(0, 1),
                                [AffineTransform(0, sigma_b), AbsTransform()]
                            )
                        )

                        L = numpyro.sample(
                            site.L,
                            dist.TransformedDistribution(
                                dist.Normal(0, 1),
                                [AffineTransform(0, sigma_L), AbsTransform()]
                            )
                        )
                        H = numpyro.sample(
                            site.H,
                            dist.TransformedDistribution(
                                dist.Normal(0, 1),
                                [AffineTransform(0, sigma_H), AbsTransform()]
                            )
                        )
                        v = numpyro.sample(
                            site.v,
                            dist.TransformedDistribution(
                                dist.Normal(0, 1),
                                [AffineTransform(0, sigma_v), AbsTransform()]
                            )
                        )

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
