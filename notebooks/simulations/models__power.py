import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel, BoundedOptimization
from hbmep.model.utils import Site as site


class Simulator(GammaModel):
    NAME = "simulator"

    def __init__(self, config: Config, a_delta_loc, a_delta_scale):
        super(Simulator, self).__init__(config=config)
        self.a_delta_loc = a_delta_loc
        self.a_delta_scale = a_delta_scale

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_delta = n_features[1] - 1

        # Fixed
        a_fixed_loc = numpyro.sample(
            "a_fixed_loc", dist.TruncatedNormal(50., 50., low=0)
        )
        a_fixed_scale = numpyro.sample(
            "a_fixed_scale", dist.HalfNormal(50.)
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
        a_delta_loc, a_delta_scale = self.a_delta_loc, self.a_delta_scale

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_delta = numpyro.sample(
                        "a_delta", dist.Normal(a_delta_loc, a_delta_scale)
                    )

        # Hyper-priors
        b_scale = numpyro.sample(
            "b_scale", dist.HalfNormal(5.)
        )

        L_scale = numpyro.sample(
            "L_scale", dist.HalfNormal(.5)
        )
        ell_scale = numpyro.sample(
            "ell_scale", dist.HalfNormal(10.)
        )
        H_scale = numpyro.sample(
            "H_scale", dist.HalfNormal(5.)
        )

        c_1_scale = numpyro.sample(
            "c_1_scale", dist.HalfNormal(5.)
        )
        c_2_scale = numpyro.sample(
            "c_2_scale", dist.HalfNormal(5.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_delta], axis=1)
                    )

                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

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
        a_fixed_loc = numpyro.sample(
            "a_fixed_loc", dist.TruncatedNormal(50., 50., low=0)
        )
        a_fixed_scale = numpyro.sample(
            "a_fixed_scale", dist.HalfNormal(50.)
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
        a_delta_loc = numpyro.sample(
            "a_delta_loc", dist.Normal(0., 50.)
        )
        a_delta_scale = numpyro.sample(
            "a_delta_scale", dist.HalfNormal(50.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_delta = numpyro.sample(
                        "a_delta", dist.Normal(a_delta_loc, a_delta_scale)
                    )
                    a_fixed_plus_delta = numpyro.deterministic(
                        "a_fixed_plus_delta", jnp.maximum(1e-6, a_fixed + a_delta)
                    )

        # Hyper-priors
        b_scale = numpyro.sample(
            "b_scale", dist.HalfNormal(5.)
        )

        L_scale = numpyro.sample(
            "L_scale", dist.HalfNormal(.5)
        )
        ell_scale = numpyro.sample(
            "ell_scale", dist.HalfNormal(10.)
        )
        H_scale = numpyro.sample(
            "H_scale", dist.HalfNormal(5.)
        )

        c_1_scale = numpyro.sample(
            "c_1_scale", dist.HalfNormal(5.)
        )
        c_2_scale = numpyro.sample(
            "c_2_scale", dist.HalfNormal(5.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
                    )

                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

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


class NonHierarchicalBayesianModel(GammaModel):
    NAME = "non_hierarchical_bayesian_model"

    def __init__(self, config: Config):
        super(NonHierarchicalBayesianModel, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Hyper-priors
                    a_loc = numpyro.sample(
                        "a_loc", dist.TruncatedNormal(50., 50., low=0)
                    )
                    a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

                    b_scale = numpyro.sample(
                        "b_scale", dist.HalfNormal(5.)
                    )

                    L_scale = numpyro.sample(
                        "L_scale", dist.HalfNormal(.5)
                    )
                    ell_scale = numpyro.sample(
                        "ell_scale", dist.HalfNormal(10.)
                    )
                    H_scale = numpyro.sample(
                        "H_scale", dist.HalfNormal(5.)
                    )

                    c_1_scale = numpyro.sample(
                        "c_1_scale", dist.HalfNormal(5.)
                    )
                    c_2_scale = numpyro.sample(
                        "c_2_scale", dist.HalfNormal(5.)
                    )

                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )

                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

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


class MaximumLikelihoodModel(GammaModel):
    NAME = "maximum_likelihood_model"

    def __init__(self, config: Config):
        super(MaximumLikelihoodModel, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Uniform priors (maximum likelihood estimation)
                    a = numpyro.sample(
                        site.a, dist.Uniform(0., 150.)
                    )

                    b = numpyro.sample(site.b, dist.Uniform(0., 10.))

                    L = numpyro.sample(site.L, dist.Uniform(0., 10.))
                    ell = numpyro.sample(site.ell, dist.Uniform(0., 10.))
                    H = numpyro.sample(site.H, dist.Uniform(0., 10.))

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


class NelderMeadOptimization(BoundedOptimization):
    NAME = "nelder_mead_optimization"

    def __init__(self, config: Config):
        super(NelderMeadOptimization, self).__init__(config=config)
        self.solver = "Nelder-Mead"
        self.functional = F.rectified_logistic
        self.named_params = [site.a, site.b, site.L, site.ell, site.H]
        self.bounds = [(1e-9, 150.), (1e-9, 10), (1e-9, 10), (1e-9, 10), (1e-9, 10)]
        self.informed_bounds = [(20, 80), (1e-3, 5.), (1e-4, .1), (1e-2, 5), (.5, 5)]
        self.num_points = 1000
        self.num_iters = 100
        self.n_jobs = -1
