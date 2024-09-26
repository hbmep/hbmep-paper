import numpy as np
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep import functional as F
from hbmep import smooth_functional as S
from hbmep.model import (
    GammaModel,
    NonHierarchicalBaseModel,
    BoundConstrainedOptimization
)
from hbmep.model.utils import Site as site


class Simulator(GammaModel):
    NAME = "simulator"

    def __init__(self, config: Config, a_loc_delta):
        super(Simulator, self).__init__(config=config)
        self.a_loc_delta = a_loc_delta

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_delta = n_features[1] - 1

        # # Fixed
        # a_loc_fixed_loc = numpyro.sample(
        #     "a_loc_fixed_loc", dist.TruncatedNormal(50., 50., low=0)
        # )
        # a_loc_fixed_scale = numpyro.sample(
        #     "a_loc_fixed_scale", dist.HalfNormal(50.)
        # )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_loc_fixed = numpyro.sample(
                    "a_loc_fixed", dist.TruncatedNormal(
                        50., 50., low=0
                    )
                )

        # # Delta
        # a_loc_delta_scale_scale = numpyro.sample(
        #     "a_loc_delta_scale_scale", dist.HalfNormal(50.)
        # )
        a_loc_delta = self.a_loc_delta

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                # a_loc_delta_scale_raw = numpyro.sample(
                #     "a_loc_delta_scale_raw", dist.HalfCauchy(scale=1.)
                # )
                # a_loc_delta_scale = numpyro.deterministic(
                #     "a_loc_delta_scale", a_loc_delta_scale_scale * a_loc_delta_scale_raw
                # )
                # a_loc_delta_raw = numpyro.sample(
                #     "a_loc_delta_raw", dist.Normal(0., 1.)
                # )
                # a_loc_delta = numpyro.deterministic(
                #     "a_loc_delta", a_loc_delta_scale * a_loc_delta_raw
                # )
                # # Penalty for negative a_loc
                # penalty_for_negative_a_loc = (
                #     jnp.fabs(a_loc_fixed + a_loc_delta) - (a_loc_fixed + a_loc_delta)
                # )
                # numpyro.factor(
                #     "penalty_for_negative_a_loc",
                #     -penalty_for_negative_a_loc
                # )
                a_loc_fixed_plus_delta = jax.nn.softplus(a_loc_fixed + a_loc_delta)

        # Global priors
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))
        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                a_loc = numpyro.deterministic(
                    "a_loc", jnp.concatenate([a_loc_fixed, a_loc_fixed_plus_delta])
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )
                    b = numpyro.sample(site.b, dist.HalfNormal(scale=b_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(scale=L_scale))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(scale=ell_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(scale=H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(scale=c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(scale=c_2_scale))

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


class HierarchicalBayesianModel(GammaModel):
    NAME = "hierarchical_bayesian_model"

    def __init__(self, config: Config):
        super(HierarchicalBayesianModel, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_delta = n_features[1] - 1

        # Fixed
        a_loc_fixed_loc = numpyro.sample(
            "a_loc_fixed_loc", dist.TruncatedNormal(50., 50., low=0)
        )
        a_loc_fixed_scale = numpyro.sample(
            "a_loc_fixed_scale", dist.HalfNormal(50.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_loc_fixed = numpyro.sample(
                    "a_loc_fixed", dist.TruncatedNormal(
                        a_loc_fixed_loc, a_loc_fixed_scale, low=0
                    )
                )

        # Delta
        a_loc_delta_scale_scale = numpyro.sample(
            "a_loc_delta_scale_scale", dist.HalfNormal(50.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                a_loc_delta_scale_raw = numpyro.sample(
                    "a_loc_delta_scale_raw", dist.HalfCauchy(scale=1.)
                )
                a_loc_delta_scale = numpyro.deterministic(
                    "a_loc_delta_scale", a_loc_delta_scale_scale * a_loc_delta_scale_raw
                )
                a_loc_delta_raw = numpyro.sample(
                    "a_loc_delta_raw", dist.Normal(0., 1.)
                )
                a_loc_delta = numpyro.deterministic(
                    "a_loc_delta", a_loc_delta_scale * a_loc_delta_raw
                )
                # Penalty for negative a_loc
                penalty_for_negative_a_loc = (
                    jnp.fabs(a_loc_fixed + a_loc_delta) - (a_loc_fixed + a_loc_delta)
                )
                numpyro.factor(
                    "penalty_for_negative_a_loc",
                    -penalty_for_negative_a_loc
                )
                a_loc_fixed_plus_delta = jax.nn.softplus(a_loc_fixed + a_loc_delta)

        # Global priors
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))
        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                a_loc = numpyro.deterministic(
                    "a_loc", jnp.concatenate([a_loc_fixed, a_loc_fixed_plus_delta], axis=0)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                    ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                    c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                    c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                    c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    S.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1],
                        eps=0.0077
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
