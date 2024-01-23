import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import GammaModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site


class RectifiedLogistic(GammaModel):
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
            # Global Priors
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5.))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5.))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(10.))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5.))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5.))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[1], n_features[1]):
                # Hyper Priors
                a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

                b_scale_raw = numpyro.sample("b_scale_raw", dist.HalfNormal(scale=1))
                b_scale = numpyro.deterministic("b_scale", jnp.multiply(b_scale_global_scale, b_scale_raw))

                v_scale_raw = numpyro.sample("v_scale_raw", dist.HalfNormal(scale=1))
                v_scale = numpyro.deterministic("v_scale", jnp.multiply(v_scale_global_scale, v_scale_raw))

                L_scale_raw = numpyro.sample("L_scale_raw", dist.HalfNormal(scale=1))
                L_scale = numpyro.deterministic("L_scale", jnp.multiply(L_scale_global_scale, L_scale_raw))

                ell_scale_raw = numpyro.sample("ell_scale_raw", dist.HalfNormal(scale=1))
                ell_scale = numpyro.deterministic("ell_scale", jnp.multiply(ell_scale_global_scale, ell_scale_raw))

                H_scale_raw = numpyro.sample("H_scale_raw", dist.HalfNormal(scale=1))
                H_scale = numpyro.deterministic("H_scale", jnp.multiply(H_scale_global_scale, H_scale_raw))

                c_1_scale_raw = numpyro.sample("c_1_scale_raw", dist.HalfNormal(scale=1))
                c_1_scale = numpyro.deterministic("c_1_scale", jnp.multiply(c_1_scale_global_scale, c_1_scale_raw))

                c_2_scale_raw = numpyro.sample("c_2_scale_raw", dist.HalfNormal(scale=1))
                c_2_scale = numpyro.deterministic("c_2_scale", jnp.multiply(c_2_scale_global_scale, c_2_scale_raw))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
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
                # Model
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


class Logistic5(GammaModel):
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
            # Global Priors
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5.))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5.))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5.))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5.))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[1], n_features[1]):
                # Hyper Priors
                a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

                b_scale_raw = numpyro.sample("b_scale_raw", dist.HalfNormal(scale=1))
                b_scale = numpyro.deterministic("b_scale", jnp.multiply(b_scale_global_scale, b_scale_raw))

                v_scale_raw = numpyro.sample("v_scale_raw", dist.HalfNormal(scale=1))
                v_scale = numpyro.deterministic("v_scale", jnp.multiply(v_scale_global_scale, v_scale_raw))

                L_scale_raw = numpyro.sample("L_scale_raw", dist.HalfNormal(scale=1))
                L_scale = numpyro.deterministic("L_scale", jnp.multiply(L_scale_global_scale, L_scale_raw))

                H_scale_raw = numpyro.sample("H_scale_raw", dist.HalfNormal(scale=1))
                H_scale = numpyro.deterministic("H_scale", jnp.multiply(H_scale_global_scale, H_scale_raw))

                c_1_scale_raw = numpyro.sample("c_1_scale_raw", dist.HalfNormal(scale=1))
                c_1_scale = numpyro.deterministic("c_1_scale", jnp.multiply(c_1_scale_global_scale, c_1_scale_raw))

                c_2_scale_raw = numpyro.sample("c_2_scale_raw", dist.HalfNormal(scale=1))
                c_2_scale = numpyro.deterministic("c_2_scale", jnp.multiply(c_2_scale_global_scale, c_2_scale_raw))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                    v = numpyro.deterministic(site.v, jnp.multiply(v_scale, v_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

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


class Logistic4(GammaModel):
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
            # Global Priors
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5.))
            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5.))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5.))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[1], n_features[1]):
                # Hyper Priors
                a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

                b_scale_raw = numpyro.sample("b_scale_raw", dist.HalfNormal(scale=1))
                b_scale = numpyro.deterministic("b_scale", jnp.multiply(b_scale_global_scale, b_scale_raw))

                L_scale_raw = numpyro.sample("L_scale_raw", dist.HalfNormal(scale=1))
                L_scale = numpyro.deterministic("L_scale", jnp.multiply(L_scale_global_scale, L_scale_raw))

                H_scale_raw = numpyro.sample("H_scale_raw", dist.HalfNormal(scale=1))
                H_scale = numpyro.deterministic("H_scale", jnp.multiply(H_scale_global_scale, H_scale_raw))

                c_1_scale_raw = numpyro.sample("c_1_scale_raw", dist.HalfNormal(scale=1))
                c_1_scale = numpyro.deterministic("c_1_scale", jnp.multiply(c_1_scale_global_scale, c_1_scale_raw))

                c_2_scale_raw = numpyro.sample("c_2_scale_raw", dist.HalfNormal(scale=1))
                c_2_scale = numpyro.deterministic("c_2_scale", jnp.multiply(c_2_scale_global_scale, c_2_scale_raw))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

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


class ReLU(GammaModel):
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
            # Global Priors
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5.))
            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5.))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5.))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[1], n_features[1]):
                # Hyper Priors
                a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

                b_scale_raw = numpyro.sample("b_scale_raw", dist.HalfNormal(scale=1))
                b_scale = numpyro.deterministic("b_scale", jnp.multiply(b_scale_global_scale, b_scale_raw))

                L_scale_raw = numpyro.sample("L_scale_raw", dist.HalfNormal(scale=1))
                L_scale = numpyro.deterministic("L_scale", jnp.multiply(L_scale_global_scale, L_scale_raw))

                H_scale_raw = numpyro.sample("H_scale_raw", dist.HalfNormal(scale=1))
                H_scale = numpyro.deterministic("H_scale", jnp.multiply(H_scale_global_scale, H_scale_raw))

                c_1_scale_raw = numpyro.sample("c_1_scale_raw", dist.HalfNormal(scale=1))
                c_1_scale = numpyro.deterministic("c_1_scale", jnp.multiply(c_1_scale_global_scale, c_1_scale_raw))

                c_2_scale_raw = numpyro.sample("c_2_scale_raw", dist.HalfNormal(scale=1))
                c_2_scale = numpyro.deterministic("c_2_scale", jnp.multiply(c_2_scale_global_scale, c_2_scale_raw))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.sample(
                        site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

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
                    F.relu(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1]
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
