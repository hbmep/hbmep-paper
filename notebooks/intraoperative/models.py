import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel
from hbmep.model.utils import Site as site

from hbmep_paper.models import (
    NonHierarchicalBaseModel,
    ConstrainedOptimization
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
        a_fixed_loc = numpyro.sample(
            "a_fixed_loc", dist.TruncatedNormal(5., 10., low=0)
        )
        a_fixed_scale = numpyro.sample(
            "a_fixed_scale", dist.HalfNormal(10.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_loc, a_fixed_scale, low=0
                        )
                    )

        # Delta
        a_delta_loc_scale_scale = numpyro.sample(
            "a_delta_loc_scale_scale", dist.HalfNormal(10.)
        )
        a_delta_scale = numpyro.sample(
            "a_delta_scale", dist.HalfNormal(10.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                a_delta_loc_scale = numpyro.sample(
                    "a_delta_loc_scale", dist.HalfCauchy(a_delta_loc_scale_scale)
                )
                a_delta_loc = numpyro.sample(
                    "a_delta_loc", dist.Normal(0., a_delta_loc_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_delta = numpyro.sample(
                        "a_delta", dist.Normal(a_delta_loc, a_delta_scale)
                    )
                    a_fixed_plus_delta = numpyro.deterministic(
                        "a_fixed_plus_delta", jnp.maximum(1e-6, a_fixed + a_delta)
                    )

        # Hyper-priors
        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
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

        # Outlier Distribution
        outlier_prob = numpyro.sample(site.outlier_prob, dist.Uniform(0., .01))

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

                # Mixture
                q = numpyro.deterministic(
                    site.q,
                    jnp.multiply(outlier_prob, jnp.ones((n_data, self.n_response)))
                )
                bg_scale = numpyro.deterministic(
                    site.bg_scale, L[feature0, feature1] + H[feature0, feature1]
                )

                mixing_distribution = dist.Categorical(
                    probs=jnp.stack([1 - q, q], axis=-1)
                )
                component_distributions=[
                    dist.Gamma(concentration=alpha, rate=beta),
                    dist.HalfNormal(scale=bg_scale)
                ]

                Mixture = dist.MixtureGeneral(
                    mixing_distribution=mixing_distribution,
                    component_distributions=component_distributions
                )

                # Observation
                numpyro.sample(
                    site.obs,
                    Mixture,
                    obs=response_obs
                )


class NonHierarchicalBayesianModel(NonHierarchicalBaseModel, GammaModel):
    NAME = "non_hierarchical_bayesian_model"

    def __init__(self, config: Config):
        super(NonHierarchicalBayesianModel, self).__init__(config=config)
        self.n_jobs = -1

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Hyper-priors
                    a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(5., 10., low=0))
                    a_scale = numpyro.sample("a_scale", dist.HalfNormal(10.))

                    b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

                    L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
                    ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
                    H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

                    c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
                    c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

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


class MaximumLikelihoodModel(NonHierarchicalBaseModel, GammaModel):
    NAME = "maximum_likelihood_model"

    def __init__(self, config: Config):
        super(MaximumLikelihoodModel, self).__init__(config=config)
        self.n_jobs = -1

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Uniform priors (maximum likelihood estimation)
                    a = numpyro.sample(site.a, dist.Uniform(0., 15.))
                    b = numpyro.sample(site.b, dist.Uniform(0., 10.))

                    L = numpyro.sample(site.L, dist.Uniform(0., 10.))
                    ell = numpyro.sample(site.ell, dist.Uniform(0, 10.))
                    H = numpyro.sample(site.H, dist.Uniform(0., 50.))

                    c_1 = numpyro.sample(site.c_1, dist.Uniform(0., 10.))
                    c_2 = numpyro.sample(site.c_2, dist.Uniform(0., 10.))

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


class NelderMeadOptimization(ConstrainedOptimization):
    NAME = "nelder_mead_optimization"

    def __init__(self, config: Config):
        super(NelderMeadOptimization, self).__init__(config=config)
        # Required values
        self.method = "Nelder-Mead"
        self.named_args = [site.a, site.b, site.L, site.ell, site.H]
        self.bounds = [(1e-9, 15.), (1e-9, 10), (1e-9, 10), (1e-9, 10), (1e-9, 50)]
        self.informed_bounds = [(2, 8), (1e-3, 5.), (1e-3, .1), (1e-3, 5.),  (.5, 10)]
        self.num_reinit = 100
        self.n_jobs = -1

    def functional(self, x, a, b, L, ell, H):
        return F.rectified_logistic(
            x, a, b, L, ell, H
        )

    def cost_function(self, x, y_obs, *args):
        y_pred = self.functional(x, *args)
        return np.sum((y_obs - y_pred) ** 2)
