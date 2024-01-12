import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site

import beta_functional as G


class Existing(BaseModel):
    NAME = "Existing"

    def __init__(self, config: Config):
        super(Existing, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Hyper Priors """
            a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(50., 20., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5))
            v_scale = numpyro.sample("v_scale", dist.HalfNormal(5))

            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
            ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(5))
            H_scale = numpyro.sample("H_scale", dist.HalfNormal(5))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5))

            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Priors """
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_mean, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                v = numpyro.deterministic(site.v, jnp.multiply(v_scale, v_raw))

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
                    G.Existing(mu, c_1[feature0], c_2[feature0])
                )
                alpha = numpyro.deterministic(
                    "alpha",
                    jnp.multiply(mu, beta)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class SD(BaseModel):
    NAME = "SD"

    def __init__(self, config: Config):
        super(SD, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Hyper Priors """
            a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(50., 20., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5))
            v_scale = numpyro.sample("v_scale", dist.HalfNormal(5))

            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
            ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(5))
            H_scale = numpyro.sample("H_scale", dist.HalfNormal(5))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5))

            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Priors """
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_mean, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                v = numpyro.deterministic(site.v, jnp.multiply(v_scale, v_raw))

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
                    G.SD(mu, c_1[feature0], c_2[feature0])
                )
                alpha = numpyro.deterministic(
                    "alpha",
                    jnp.multiply(mu, beta)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class PowerSD(BaseModel):
    NAME = "PowerSD"

    def __init__(self, config: Config):
        super(PowerSD, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Hyper Priors """
            a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(50., 20., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5))
            v_scale = numpyro.sample("v_scale", dist.HalfNormal(5))

            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
            ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(5))
            H_scale = numpyro.sample("H_scale", dist.HalfNormal(5))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5))
            c_3_scale = numpyro.sample("c_3_scale", dist.HalfNormal(2))

            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Priors """
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_mean, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                v = numpyro.deterministic(site.v, jnp.multiply(v_scale, v_raw))

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

                c_3_raw = numpyro.sample("c_3_raw", dist.HalfNormal(scale=1))
                c_3 = numpyro.deterministic("c_3", jnp.multiply(c_3_scale, c_3_raw))

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
                    G.PowerSD(mu, c_1[feature0], c_2[feature0], c_3[feature0])
                )
                alpha = numpyro.deterministic(
                    "alpha",
                    jnp.multiply(mu, beta)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class PowerSDMinusL(BaseModel):
    NAME = "PowerSDMinusL"

    def __init__(self, config: Config):
        super(PowerSDMinusL, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            """ Hyper Priors """
            a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(50., 20., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5))
            v_scale = numpyro.sample("v_scale", dist.HalfNormal(5))

            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
            ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(5))
            H_scale = numpyro.sample("H_scale", dist.HalfNormal(5))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5))
            c_3_scale = numpyro.sample("c_3_scale", dist.HalfNormal(2))

            with numpyro.plate(site.n_features[0], n_features[0]):
                """ Priors """
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_mean, a_scale, low=0)
                )

                b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                v = numpyro.deterministic(site.v, jnp.multiply(v_scale, v_raw))

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

                c_3_raw = numpyro.sample("c_3_raw", dist.HalfNormal(scale=1))
                c_3 = numpyro.deterministic("c_3", jnp.multiply(c_3_scale, c_3_raw))

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
                    G.PowerSDMinusL(mu, L[feature0], c_1[feature0], c_2[feature0], c_3[feature0])
                )
                alpha = numpyro.deterministic(
                    "alpha",
                    jnp.multiply(mu, beta)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )
