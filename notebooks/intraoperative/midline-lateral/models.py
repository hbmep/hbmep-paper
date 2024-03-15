import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel
from hbmep.model.utils import Site as site


class MixedEffects(GammaModel):
    NAME = "mixed_effects"

    def __init__(self, config: Config):
        super(MixedEffects, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_random = n_features[1] - 1

        # Fixed Effects (Baseline)
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample(
                    "a_fixed_mean", dist.TruncatedNormal(5., 10., low=0)
                )
                a_fixed_scale = numpyro.sample(
                    "a_fixed_scale", dist.HalfNormal(10.)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_mean, a_fixed_scale, low=0
                        )
                    )

        # Random Effects (Delta)
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_random", n_random):
                a_random_mean = numpyro.sample(
                    "a_random_mean", dist.Normal(0., 10.)
                )
                a_random_scale = numpyro.sample(
                    "a_random_scale", dist.HalfNormal(10.)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_random = numpyro.sample(
                        "a_random", dist.Normal(a_random_mean, a_random_scale)
                    )

                    # Penalty for negative a
                    penalty_for_negative_a = (
                        jnp.fabs(a_fixed + a_random) - (a_fixed + a_random)
                    )
                    numpyro.factor(
                        "penalty_for_negative_a", -penalty_for_negative_a
                    )

        with numpyro.plate(site.n_response, self.n_response):
            # Global Priors
            b_scale_global_scale = numpyro.sample(
                "b_scale_global_scale", dist.HalfNormal(5.)
            )

            L_scale_global_scale = numpyro.sample(
                "L_scale_global_scale", dist.HalfNormal(.5)
            )
            ell_scale_global_scale = numpyro.sample(
                "ell_scale_global_scale", dist.HalfNormal(10.)
            )
            H_scale_global_scale = numpyro.sample(
                "H_scale_global_scale", dist.HalfNormal(5.)
            )

            c_1_scale_global_scale = numpyro.sample(
                "c_1_scale_global_scale", dist.HalfNormal(5.)
            )
            c_2_scale_global_scale = numpyro.sample(
                "c_2_scale_global_scale", dist.HalfNormal(5.)
            )

            with numpyro.plate(site.n_features[1], n_features[1]):
                # Hyper-priors
                b_scale_raw = numpyro.sample(
                    "b_scale_raw",
                    dist.HalfNormal(scale=1)
                )
                b_scale = numpyro.deterministic(
                    "b_scale",
                    jnp.multiply(b_scale_global_scale, b_scale_raw)
                )

                L_scale_raw = numpyro.sample(
                    "L_scale_raw",
                    dist.HalfNormal(scale=1)
                )
                L_scale = numpyro.deterministic(
                    "L_scale",
                    jnp.multiply(L_scale_global_scale, L_scale_raw)
                )

                ell_scale_raw = numpyro.sample(
                    "ell_scale_raw",
                    dist.HalfNormal(scale=1)
                )
                ell_scale = numpyro.deterministic(
                    "sigma_ell",
                    jnp.multiply(ell_scale_global_scale, ell_scale_raw)
                )

                H_scale_raw = numpyro.sample(
                    "H_scale_raw",
                    dist.HalfNormal(scale=1)
                )
                H_scale = numpyro.deterministic(
                    "H_scale",
                    jnp.multiply(H_scale_global_scale, H_scale_raw)
                )

                c_1_scale_raw = numpyro.sample(
                    "c_1_scale_raw",
                    dist.HalfNormal(scale=1)
                )
                c_1_scale = numpyro.deterministic(
                    "c_1_scale",
                    jnp.multiply(c_1_scale_global_scale, c_1_scale_raw)
                )

                c_2_scale_raw = numpyro.sample(
                    "c_2_scale_raw",
                    dist.HalfNormal(scale=1)
                )
                c_2_scale = numpyro.deterministic(
                    "c_2_scale",
                    jnp.multiply(c_2_scale_global_scale, c_2_scale_raw)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=1)
                    )

                    b_raw = numpyro.sample(
                        "b_raw", dist.HalfNormal(scale=1)
                    )
                    b = numpyro.deterministic(
                        site.b, jnp.multiply(b_scale, b_raw)
                    )

                    L_raw = numpyro.sample(
                        "L_raw", dist.HalfNormal(scale=1)
                    )
                    L = numpyro.deterministic(
                        site.L, jnp.multiply(L_scale, L_raw)
                    )

                    ell_raw = numpyro.sample(
                        "ell_raw", dist.HalfNormal(scale=1)
                    )
                    ell = numpyro.deterministic(
                        "ell", jnp.multiply(ell_scale, ell_raw)
                    )

                    H_raw = numpyro.sample(
                        "H_raw", dist.HalfNormal(scale=1)
                    )
                    H = numpyro.deterministic(
                        site.H, jnp.multiply(H_scale, H_raw)
                    )

                    c_1_raw = numpyro.sample(
                        "c_1_raw", dist.HalfCauchy(scale=1)
                    )
                    c_1 = numpyro.deterministic(
                        site.c_1, jnp.multiply(c_1_scale, c_1_raw)
                    )

                    c_2_raw = numpyro.sample(
                        "c_2_raw", dist.HalfCauchy(scale=1)
                    )
                    c_2 = numpyro.deterministic(
                        site.c_2, jnp.multiply(c_2_scale, c_2_raw)
                    )

        # Outlier Distribution
        outlier_prob = numpyro.sample(site.outlier_prob, dist.Uniform(0., .05))
        outlier_scale = numpyro.sample(site.outlier_scale, dist.HalfNormal(30))

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
                    site.q, outlier_prob * jnp.ones((n_data, self.n_response))
                )
                bg_scale = numpyro.deterministic(
                    site.bg_scale,
                    outlier_scale * jnp.ones((n_data, self.n_response))
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
