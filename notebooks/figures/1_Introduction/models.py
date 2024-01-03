import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site


class RectifiedLogistic(BaseModel):
    NAME = "rectified_logistic"

    def __init__(self, config: Config):
        super(RectifiedLogistic, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(100))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(100))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.05))
            ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(100))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(1))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(100))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(100))

            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(150, 100, low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.0))

                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(ell_scale_global_scale))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.sample(site.a, dist.TruncatedNormal(a_mean, a_scale, low=0))
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
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        v=v[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    c_1[feature0, feature1] + jnp.true_divide(c_2[feature0, feature1], mu)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta),
                    obs=response_obs
                )


class Logistic5(BaseModel):
    NAME = "logistic5"

    def __init__(self, config: Config):
        super(Logistic5, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(100))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(100))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.05))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(1))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(100))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(100))

            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(150, 100, low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.0))

                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.sample(site.a, dist.TruncatedNormal(a_mean, a_scale, low=0))
                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                    v = numpyro.sample(site.v, dist.HalfNormal(v_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.logistic5(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        v=v[feature0, feature1],
                        L=L[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    c_1[feature0, feature1] + jnp.true_divide(c_2[feature0, feature1], mu)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta),
                    obs=response_obs
                )


class Logistic4(BaseModel):
    NAME = "logistic4"

    def __init__(self, config: Config):
        super(Logistic4, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(100))
            # v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(100))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.05))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(1))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(100))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(100))

            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(150, 100, low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.0))

                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                # v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.sample(site.a, dist.TruncatedNormal(a_mean, a_scale, low=0))
                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                    # v = numpyro.sample(site.v, dist.HalfNormal(v_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.logistic4(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    c_1[feature0, feature1] + jnp.true_divide(c_2[feature0, feature1], mu)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta),
                    obs=response_obs
                )


class ReLU(BaseModel):
    NAME = "relu"

    def __init__(self, config: Config):
        super(ReLU, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(100))
            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.05))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(100))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(100))

            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(150, 100, low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(100.0))

                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.sample(site.a, dist.TruncatedNormal(a_mean, a_scale, low=0))
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
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    c_1[feature0, feature1] + jnp.true_divide(c_2[feature0, feature1], mu)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta),
                    obs=response_obs
                )
