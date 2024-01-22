import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

from hbmep.config import Config
from hbmep.model import GammaModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site


def _model(intensity, response_obs=None):
    n_data = intensity.shape[0]

    """ Priors """
    a = numpyro.sample(
        site.a, dist.Normal(0., 50.)
    )
    b = numpyro.sample(site.b, dist.HalfNormal(2.))

    eps = numpyro.sample("eps", dist.HalfNormal(1.))

    with numpyro.plate(site.n_data, n_data):
        """ Model """
        mu = numpyro.deterministic(
            site.mu,
            jnp.multiply(b, intensity - a)
        )

        """ Observation """
        numpyro.sample(
            site.obs,
            dist.Normal(loc=mu, scale=eps),
            obs=response_obs
        )


def _guide(intensity, response_obs=None):
    n_data = intensity.shape[0]

    a_loc = numpyro.param("a_loc", jnp.zeros(1) * 89.11)
    a_scale = numpyro.param("a_scale", jnp.ones(1) * 1, constraint=constraints.positive)

    b_scale = numpyro.param("b_scale", jnp.ones(1) * .5, constraint=constraints.positive)
    eps_scale = numpyro.param("eps_scale", jnp.ones(1) * .5, constraint=constraints.positive)

    """ Priors """
    a = numpyro.sample(
        site.a, dist.Normal(a_loc, a_scale)
    )
    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

    eps = numpyro.sample("eps", dist.HalfNormal(eps_scale))

    with numpyro.plate(site.n_data, n_data):
        """ Model """
        mu = numpyro.deterministic(
            site.mu,
            jnp.multiply(b, intensity - a)
        )

        """ Observation """
        numpyro.sample(
            site.obs,
            dist.Normal(loc=mu, scale=eps),
            obs=response_obs
        )