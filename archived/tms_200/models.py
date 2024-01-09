import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site


class LearnPosterior(BaseModel):
    NAME = "learn_posterior"

    def __init__(self, config: Config):
        super(LearnPosterior, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)
        n_fixed = 1

        """ Fixed Effects (Baseline) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample("a_fixed_mean", dist.TruncatedNormal(50., 20., low=0))
                a_fixed_scale = numpyro.sample("a_fixed_scale", dist.HalfNormal(30.))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(a_fixed_mean, a_fixed_scale, low=0)
                    )

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(10))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5))

            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(ell_scale_global_scale))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.deterministic(
                        site.a, a_fixed
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


class Simulator(BaseModel):
    NAME = "simulator"

    def __init__(self, config: Config, a_random_mean, a_random_scale):
        super(Simulator, self).__init__(config=config)
        self.a_random_mean = a_random_mean
        self.a_random_scale = a_random_scale

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)
        n_fixed = 1
        n_random = n_features[1] - 1
        a_random_mean, a_random_scale = self.a_random_mean, self.a_random_scale

        """ Fixed Effects (Baseline) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample("a_fixed_mean", dist.TruncatedNormal(50., 20., low=0))
                a_fixed_scale = numpyro.sample("a_fixed_scale", dist.HalfNormal(30.))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(a_fixed_mean, a_fixed_scale, low=0)
                    )

        """ Random Effects (Delta) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_random", n_random):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_random = numpyro.sample("a_random", dist.Normal(a_random_mean, a_random_scale))

                    """ Penalty """
                    penalty_for_negative_a = (jnp.fabs(a_fixed + a_random) - (a_fixed + a_random))
                    numpyro.factor("penalty_for_negative_a", -penalty_for_negative_a)

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(10))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5))

            with numpyro.plate("n_fixed", n_fixed):
                """ Hyper-priors """
                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(ell_scale_global_scale))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    b_fixed = numpyro.sample("b_fixed", dist.HalfNormal(b_scale))
                    v_fixed = numpyro.sample("v_fixed", dist.HalfNormal(v_scale))

                    L_fixed = numpyro.sample("L_fixed", dist.HalfNormal(L_scale))
                    ell_fixed = numpyro.sample("ell_fixed", dist.HalfNormal(ell_scale))
                    H_fixed = numpyro.sample("H_fixed", dist.HalfNormal(H_scale))

                    c_1_fixed = numpyro.sample("c_1_fixed", dist.HalfNormal(c_1_scale))
                    c_2_fixed = numpyro.sample("c_2_fixed", dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=-2)
                    )
                    b = numpyro.deterministic(
                        site.b,
                        jnp.concatenate([b_fixed, b_fixed], axis=-2)
                    )
                    v = numpyro.deterministic(
                        site.v,
                        jnp.concatenate([v_fixed, v_fixed], axis=-2)
                    )
                    L = numpyro.deterministic(
                        site.L,
                        jnp.concatenate([L_fixed, L_fixed], axis=-2)
                    )
                    ell = numpyro.deterministic(
                        site.ell,
                        jnp.concatenate([ell_fixed, ell_fixed], axis=-2)
                    )
                    H = numpyro.deterministic(
                        site.H,
                        jnp.concatenate([H_fixed, H_fixed], axis=-2)
                    )
                    c_1 = numpyro.deterministic(
                        site.c_1,
                        jnp.concatenate([c_1_fixed, c_1_fixed], axis=-2)
                    )
                    c_2 = numpyro.deterministic(
                        site.c_2,
                        jnp.concatenate([c_2_fixed, c_2_fixed], axis=-2)
                    )

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


class HBModel(BaseModel):
    NAME = "hbm"

    def __init__(self, config: Config):
        super(HBModel, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)
        n_fixed = 1
        n_random = n_features[1] - 1

        """ Fixed Effects (Baseline) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample("a_fixed_mean", dist.TruncatedNormal(50., 20., low=0))
                a_fixed_scale = numpyro.sample("a_fixed_scale", dist.HalfNormal(30.))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(a_fixed_mean, a_fixed_scale, low=0)
                    )

        """ Random Effects (Delta) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_random", n_random):
                a_random_mean = numpyro.sample("a_random_mean", dist.Normal(0, 50))
                a_random_scale = numpyro.sample("a_random_scale", dist.HalfNormal(50.0))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_random = numpyro.sample("a_random", dist.Normal(a_random_mean, a_random_scale))

                    """ Penalty """
                    penalty_for_negative_a = (jnp.fabs(a_fixed + a_random) - (a_fixed + a_random))
                    numpyro.factor("penalty_for_negative_a", -penalty_for_negative_a)

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(10))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5))

            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(ell_scale_global_scale))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=-2)
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


class NHBModel(BaseModel):
    NAME = "nhbm"

    def __init__(self, config: Config):
        super(NHBModel, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)
        n_fixed = 1
        n_random = n_features[1] - 1

        """ Fixed Effects (Baseline) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed_mean = numpyro.sample("a_fixed_mean", dist.TruncatedNormal(50., 20., low=0))
                    a_fixed_scale = numpyro.sample("a_fixed_scale", dist.HalfNormal(30.))
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(a_fixed_mean, a_fixed_scale, low=0)
                    )

        """ Random Effects (Delta) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_random", n_random):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_random_mean = numpyro.sample("a_random_mean", dist.Normal(0, 50))
                    a_random_scale = numpyro.sample("a_random_scale", dist.HalfNormal(50.0))
                    a_random = numpyro.sample("a_random", dist.Normal(a_random_mean, a_random_scale))

                    """ Penalty """
                    penalty_for_negative_a = (jnp.fabs(a_fixed + a_random) - (a_fixed + a_random))
                    numpyro.factor("penalty_for_negative_a", -penalty_for_negative_a)

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Global Priors """
                    b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5))
                    v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5))

                    L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
                    ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(10))
                    H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5))

                    c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5))
                    c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5))

                    """ Hyper-priors """
                    b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                    v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                    L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                    ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(ell_scale_global_scale))
                    H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                    c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                    c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                    """ Priors """
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=-2)
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
