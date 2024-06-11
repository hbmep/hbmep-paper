import logging

import numpy as np
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel, BoundedOptimization
from hbmep.model.utils import Site as site

logger = logging.getLogger(__name__)


class HierarchicalBayesianModel(GammaModel):
    NAME = "hierarchical_bayesian_model"

    def __init__(self, config: Config):
        super(HierarchicalBayesianModel, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        # Hyper Priors
        a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
        a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
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
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
                        ell=ell[feature0],
                        H=H[feature0]
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


class NonHierarchicalBayesianModel(GammaModel):
    NAME = "non_hierarchical_bayesian_model"

    def __init__(self, config: Config):
        super(NonHierarchicalBayesianModel, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                # Hyper Priors
                a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

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
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
                        ell=ell[feature0],
                        H=H[feature0]
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


class MaximumLikelihoodModel(GammaModel):
    NAME = "maximum_likelihood_model"

    def __init__(self, config: Config):
        super(MaximumLikelihoodModel, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        with numpyro.plate(site.n_response, self.n_response):
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
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
                        ell=ell[feature0],
                        H=H[feature0]
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


class RectifiedLogisticS50(GammaModel):
    NAME = "rectified_logistic_in_S50_parameterization"

    def __init__(self, config: Config):
        super(RectifiedLogisticS50, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        with numpyro.plate(site.n_response, self.n_response):
            # Hyper Priors
            a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 50., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
            ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
            H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[0], n_features[0]):
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
                    F.rectified_logistic_s50(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
                        ell=ell[feature0],
                        H=H[feature0]
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
