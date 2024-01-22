import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.numpy as jnp

from hbmep.config import Config
from hbmep.model import GammaModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site


class HierarchicalBayesianModel(GammaModel):
    NAME = "hbm"

    def __init__(self, config: Config):
        super(HierarchicalBayesianModel, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Hyper-priors """
            a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
            v_scale = numpyro.sample("v_scale", dist.HalfNormal(5.))

            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
            ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
            H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Priors """
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                v = numpyro.sample(site.v, dist.HalfNormal(v_scale))

                L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        v=v[feature0],
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

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class NonHierarchicalBayesianModel(GammaModel):
    NAME = "nhbm"

    def __init__(self, config: Config):
        super(NonHierarchicalBayesianModel, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Hyper-priors """
                a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

                b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))
                v_scale = numpyro.sample("v_scale", dist.HalfNormal(5.))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
                ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

                """ Priors """
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                v = numpyro.sample(site.v, dist.HalfNormal(v_scale))

                L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        v=v[feature0],
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

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class MaximumLikelihoodModel(GammaModel):
    NAME = "mle"

    def __init__(self, config: Config):
        super(MaximumLikelihoodModel, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Hyper-priors """
                # a_loc = numpyro.sample("a_loc", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # a_scale = numpyro.sample("a_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # b_scale = numpyro.sample("b_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # v_scale = numpyro.sample("v_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # L_scale = numpyro.sample("L_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # ell_scale = numpyro.sample("ell_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # H_scale = numpyro.sample("H_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # c_1_scale = numpyro.sample("c_1_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # c_2_scale = numpyro.sample("c_2_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # a_loc = numpyro.sample("a_loc", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # a_scale = numpyro.sample("a_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # b_scale = numpyro.sample("b_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # L_scale = numpyro.sample("L_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # c_1_scale = numpyro.sample("c_1_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # c_2_scale = numpyro.sample("c_2_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # a_loc = numpyro.sample("a_loc", dist.Uniform(constraints.positive, (), event_shape=()))
                # a_scale = numpyro.sample("a_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # b_scale = numpyro.sample("b_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # L_scale = numpyro.sample("L_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                # c_1_scale = numpyro.sample("c_1_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))
                # c_2_scale = numpyro.sample("c_2_scale", dist.ImproperUniform(constraints.positive, (), event_shape=()))

                """ Priors """
                # a = numpyro.sample(
                #     site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                # )

                # b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                # v = numpyro.sample(site.v, dist.HalfNormal(v_scale))

                # L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                # ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                # H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                # a = numpyro.sample(
                #     site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                # )

                # b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                # L = numpyro.sample(site.L, dist.HalfNormal(L_scale))

                # c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                # c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

                a = numpyro.sample(
                    site.a, dist.Uniform(0., 150.)
                )

                b = numpyro.sample(site.b, dist.Uniform(0., 5.))
                L = numpyro.sample(site.L, dist.Uniform(0., 5.))

                c_1 = numpyro.sample(site.c_1, dist.Uniform(0., 10.))
                c_2 = numpyro.sample(site.c_2, dist.Uniform(0., 10.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.relu(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        L=L[feature0],
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


class MaximumLikelihoodModelRecLog(GammaModel):
    NAME = "mle_rec_log"

    def __init__(self, config: Config):
        super(MaximumLikelihoodModelRecLog, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Priors """
                a = numpyro.sample(
                    site.a, dist.Uniform(0., 150.)
                )

                b = numpyro.sample(site.b, dist.Uniform(0., 10.))
                v = numpyro.sample(site.v, dist.Uniform(0., 10.))

                L = numpyro.sample(site.L, dist.Uniform(0., 10.))
                ell = numpyro.sample(site.ell, dist.Uniform(0., 10.))
                H = numpyro.sample(site.H, dist.Uniform(0., 10.))

                c_1 = numpyro.sample(site.c_1, dist.Uniform(0., 10.))
                c_2 = numpyro.sample(site.c_2, dist.Uniform(0., 10.))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
                        v=v[feature0],
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

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )
