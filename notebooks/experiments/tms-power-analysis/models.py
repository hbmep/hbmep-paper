import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel
from hbmep.model.utils import Site as site


class LearnPosterior(GammaModel):
    NAME = "learn_posterior"

    def __init__(self, config: Config):
        super(LearnPosterior, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        # n_random = n_features[1] - 1

        # Fixed Effects (Baseline)
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample(
                    "a_fixed_mean", dist.TruncatedNormal(50., 20., low=0)
                )
                a_fixed_scale = numpyro.sample(
                    "a_fixed_scale", dist.HalfNormal(30.)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_mean, a_fixed_scale, low=0
                        )
                    )

        # # Random Effects (Delta)
        # with numpyro.plate(site.n_response, self.n_response):
        #     with numpyro.plate("n_random", n_random):
        #         a_random_mean = numpyro.sample(
        #             "a_random_mean", dist.Normal(0., 20.)
        #         )
        #         a_random_scale = numpyro.sample(
        #             "a_random_scale", dist.HalfNormal(30.)
        #         )

        #         with numpyro.plate(site.n_features[0], n_features[0]):
        #             a_random = numpyro.sample(
        #                 "a_random", dist.Normal(a_random_mean, a_random_scale)
        #             )

        #             # Penalty for negative a
        #             penalty_for_negative_a = (
        #                 jnp.fabs(a_fixed + a_random) - (a_fixed + a_random)
        #             )
        #             numpyro.factor(
        #                 "penalty_for_negative_a", -penalty_for_negative_a
        #             )

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
                b_scale = numpyro.sample(
                    "b_scale", dist.HalfNormal(b_scale_global_scale)
                )

                L_scale = numpyro.sample(
                    "L_scale", dist.HalfNormal(L_scale_global_scale)
                )
                ell_scale = numpyro.sample(
                    "ell_scale", dist.HalfNormal(ell_scale_global_scale)
                )
                H_scale = numpyro.sample(
                    "H_scale", dist.HalfNormal(H_scale_global_scale)
                )

                c_1_scale = numpyro.sample(
                    "c_1_scale", dist.HalfNormal(c_1_scale_global_scale)
                )
                c_2_scale = numpyro.sample(
                    "c_2_scale", dist.HalfNormal(c_2_scale_global_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    # a = numpyro.deterministic(
                    #     site.a,
                    #     jnp.concatenate([a_fixed, a_fixed + a_random], axis=1)
                    # )
                    a = numpyro.deterministic(site.a, a_fixed)

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


class Simulator(GammaModel):
    NAME = "simulator"

    def __init__(self, config: Config, a_random_mean, a_random_scale):
        super(Simulator, self).__init__(config=config)
        self.a_random_mean = a_random_mean
        self.a_random_scale = a_random_scale

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
                    "a_fixed_mean", dist.TruncatedNormal(50., 20., low=0)
                )
                a_fixed_scale = numpyro.sample(
                    "a_fixed_scale", dist.HalfNormal(30.)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_mean, a_fixed_scale, low=0
                        )
                    )

        # Random Effects (Delta)
        a_random_mean, a_random_scale = self.a_random_mean, self.a_random_scale

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_random", n_random):
                # a_random_mean = numpyro.sample(
                #     "a_random_mean", dist.Normal(0., 20.)
                # )
                # a_random_scale = numpyro.sample(
                #     "a_random_scale", dist.HalfNormal(30.)
                # )

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
                b_scale = numpyro.sample(
                    "b_scale", dist.HalfNormal(b_scale_global_scale)
                )

                L_scale = numpyro.sample(
                    "L_scale", dist.HalfNormal(L_scale_global_scale)
                )
                ell_scale = numpyro.sample(
                    "ell_scale", dist.HalfNormal(ell_scale_global_scale)
                )
                H_scale = numpyro.sample(
                    "H_scale", dist.HalfNormal(H_scale_global_scale)
                )

                c_1_scale = numpyro.sample(
                    "c_1_scale", dist.HalfNormal(c_1_scale_global_scale)
                )
                c_2_scale = numpyro.sample(
                    "c_2_scale", dist.HalfNormal(c_2_scale_global_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=1)
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
        n_random = n_features[1] - 1

        # Fixed Effects (Baseline)
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample(
                    "a_fixed_mean", dist.TruncatedNormal(50., 20., low=0)
                )
                a_fixed_scale = numpyro.sample(
                    "a_fixed_scale", dist.HalfNormal(30.)
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
                    "a_random_mean", dist.Normal(0., 20.)
                )
                a_random_scale = numpyro.sample(
                    "a_random_scale", dist.HalfNormal(30.)
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
                b_scale = numpyro.sample(
                    "b_scale", dist.HalfNormal(b_scale_global_scale)
                )

                L_scale = numpyro.sample(
                    "L_scale", dist.HalfNormal(L_scale_global_scale)
                )
                ell_scale = numpyro.sample(
                    "ell_scale", dist.HalfNormal(ell_scale_global_scale)
                )
                H_scale = numpyro.sample(
                    "H_scale", dist.HalfNormal(H_scale_global_scale)
                )

                c_1_scale = numpyro.sample(
                    "c_1_scale", dist.HalfNormal(c_1_scale_global_scale)
                )
                c_2_scale = numpyro.sample(
                    "c_2_scale", dist.HalfNormal(c_2_scale_global_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=1)
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

        n_fixed = 1
        n_random = n_features[1] - 1

        # Fixed Effects (Baseline)
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed_mean = numpyro.sample(
                        "a_fixed_mean", dist.TruncatedNormal(50., 20., low=0)
                    )
                    a_fixed_scale = numpyro.sample(
                        "a_fixed_scale", dist.HalfNormal(30.)
                    )
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_mean, a_fixed_scale, low=0
                        )
                    )

        # Random Effects (Delta)
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_random", n_random):
                a_random_mean = numpyro.sample(
                    "a_random_mean", dist.Normal(0., 20.)
                )
                a_random_scale = numpyro.sample(
                    "a_random_scale", dist.HalfNormal(30.)
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
                b_scale = numpyro.sample(
                    "b_scale", dist.HalfNormal(b_scale_global_scale)
                )

                L_scale = numpyro.sample(
                    "L_scale", dist.HalfNormal(L_scale_global_scale)
                )
                ell_scale = numpyro.sample(
                    "ell_scale", dist.HalfNormal(ell_scale_global_scale)
                )
                H_scale = numpyro.sample(
                    "H_scale", dist.HalfNormal(H_scale_global_scale)
                )

                c_1_scale = numpyro.sample(
                    "c_1_scale", dist.HalfNormal(c_1_scale_global_scale)
                )
                c_2_scale = numpyro.sample(
                    "c_2_scale", dist.HalfNormal(c_2_scale_global_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=1)
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


# class NonHierarchicalBayesianModel(GammaModel):
#     NAME = "non_hierarchical_bayesian_model"

#     def __init__(self, config: Config):
#         super(NonHierarchicalBayesianModel, self).__init__(config=config)

#     def _model(self, intensity, features, response_obs=None):
#         n_data = intensity.shape[0]
#         n_features = np.max(features, axis=0) + 1
#         feature0 = features[..., 0]
#         feature1 = features[..., 1]

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate(site.n_features[1], n_features[1]):
#                 with numpyro.plate(site.n_features[0], n_features[0]):
#                     # Global Priors
#                     b_scale_global_scale = numpyro.sample(
#                         "b_scale_global_scale", dist.HalfNormal(5.)
#                     )

#                     L_scale_global_scale = numpyro.sample(
#                         "L_scale_global_scale", dist.HalfNormal(.5)
#                     )
#                     ell_scale_global_scale = numpyro.sample(
#                         "ell_scale_global_scale", dist.HalfNormal(10.)
#                     )
#                     H_scale_global_scale = numpyro.sample(
#                         "H_scale_global_scale", dist.HalfNormal(5.)
#                     )

#                     c_1_scale_global_scale = numpyro.sample(
#                         "c_1_scale_global_scale", dist.HalfNormal(5.)
#                     )
#                     c_2_scale_global_scale = numpyro.sample(
#                         "c_2_scale_global_scale", dist.HalfNormal(5.)
#                     )

#                     # Hyper-priors
#                     a_loc = numpyro.sample(
#                         "a_loc", dist.TruncatedNormal(50., 20., low=0)
#                     )
#                     a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

#                     b_scale = numpyro.sample(
#                         "b_scale", dist.HalfNormal(b_scale_global_scale)
#                     )

#                     L_scale = numpyro.sample(
#                         "L_scale", dist.HalfNormal(L_scale_global_scale)
#                     )
#                     ell_scale = numpyro.sample(
#                         "ell_scale", dist.HalfNormal(ell_scale_global_scale)
#                     )
#                     H_scale = numpyro.sample(
#                         "H_scale", dist.HalfNormal(H_scale_global_scale)
#                     )

#                     c_1_scale = numpyro.sample(
#                         "c_1_scale", dist.HalfNormal(c_1_scale_global_scale)
#                     )
#                     c_2_scale = numpyro.sample(
#                         "c_2_scale", dist.HalfNormal(c_2_scale_global_scale)
#                     )

#                     # Priors
#                     a = numpyro.sample(
#                         site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
#                     )

#                     b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

#                     L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
#                     ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
#                     H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

#                     c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
#                     c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate(site.n_data, n_data):
#                 # Model
#                 mu = numpyro.deterministic(
#                     site.mu,
#                     F.rectified_logistic(
#                         x=intensity,
#                         a=a[feature0, feature1],
#                         b=b[feature0, feature1],
#                         L=L[feature0, feature1],
#                         ell=ell[feature0, feature1],
#                         H=H[feature0, feature1]
#                     )
#                 )
#                 beta = numpyro.deterministic(
#                     site.beta,
#                     self.rate(
#                         mu,
#                         c_1[feature0, feature1],
#                         c_2[feature0, feature1]
#                     )
#                 )
#                 alpha = numpyro.deterministic(
#                     site.alpha,
#                     self.concentration(mu, beta)
#                 )

#                 # Observation
#                 numpyro.sample(
#                     site.obs,
#                     dist.Gamma(concentration=alpha, rate=beta),
#                     obs=response_obs
#                 )
