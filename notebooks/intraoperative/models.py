import numpy as np
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep import smooth_functional as S
from hbmep.model import GammaModel
from hbmep.model.utils import Site as site

EPS = 1e-3


# class HierarchicalBayesianModel(GammaModel):
#     NAME = "1"

#     def __init__(self, config: Config):
#         super(HierarchicalBayesianModel, self).__init__(config=config)
#         self.mcmc_params = {
#             "num_warmup": 20000,
#             "num_samples": 20000,
#             "num_chains": 4,
#             "thinning": 20,
#         }
#         self.run_kwargs = {
#             "max_tree_depth": (20, 20),
#             "target_accept_prob": .95,
#             "extra_fields": [
#                 "potential_energy",
#                 "num_steps",
#                 "accept_prob",
#             ]
#         }
#         self.NAME += (
#             f'_{self.mcmc_params["num_warmup"]}W'
#             f'_{self.mcmc_params["num_samples"]}S'
#             f'_{self.run_kwargs["max_tree_depth"][0]}D'
#             f'_{self.run_kwargs["target_accept_prob"]}A'
#         )

#     def _model(self, intensity, features, response_obs=None):
#         n_data = intensity.shape[0]
#         n_features = np.max(features, axis=0) + 1
#         feature0 = features[..., 0]
#         feature1 = features[..., 1]

#         n_fixed = 1
#         n_delta = n_features[1] - 1

#         # Fixed
#         a_fixed_loc = numpyro.sample(
#             "a_fixed_loc", dist.TruncatedNormal(5., 10., low=0.)
#         )
#         a_fixed_scale = numpyro.sample(
#             "a_fixed_scale", dist.HalfNormal(10.)
#         )

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate("n_fixed", n_fixed):
#                 with numpyro.plate(site.n_features[0], n_features[0]):
#                     a_fixed = numpyro.sample(
#                         "a_fixed", dist.TruncatedNormal(
#                             a_fixed_loc, a_fixed_scale, low=0.
#                         )
#                     )

#         # Delta
#         a_delta_scale = numpyro.sample(
#             "a_delta_scale", dist.HalfNormal(10.)
#         )

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate("n_delta", n_delta):
#                 a_delta_loc = numpyro.sample(
#                     "a_delta_loc", dist.Normal(0., 10.)
#                 )

#                 with numpyro.plate(site.n_features[0], n_features[0]):
#                     a_delta_raw = numpyro.sample("a_delta_raw", dist.Normal(0., 1.))
#                     a_delta = numpyro.deterministic(
#                         "a_delta", a_delta_loc + a_delta_raw * a_delta_scale
#                     )

#                     # Penalty for negative a
#                     penalty_for_negative_a = (
#                         jnp.fabs(a_fixed + a_delta) - (a_fixed + a_delta)
#                     )
#                     numpyro.factor(
#                         "penalty_for_negative_a", -penalty_for_negative_a
#                     )
#                     a_fixed_plus_delta = jax.nn.softplus(a_fixed + a_delta)

#         # Hyper-priors
#         b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

#         L_scale = numpyro.sample("L_scale", dist.HalfNormal(.1))
#         ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(1.))
#         H_scale = numpyro.sample("H_scale", dist.HalfNormal(10.))

#         c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
#         c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(.5))

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate(site.n_features[1], n_features[1]):
#                 with numpyro.plate(site.n_features[0], n_features[0]):
#                     # Priors
#                     a = numpyro.deterministic(
#                         site.a,
#                         jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
#                     )

#                     b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
#                     b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

#                     L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
#                     L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

#                     ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
#                     ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

#                     H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
#                     H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

#                     c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
#                     c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

#                     c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
#                     c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

#         q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate(site.n_data, n_data):
#                 # Model
#                 mu = numpyro.deterministic(
#                     site.mu,
#                     S.rectified_logistic(
#                         x=intensity,
#                         a=a[feature0, feature1],
#                         b=b[feature0, feature1],
#                         L=L[feature0, feature1],
#                         ell=ell[feature0, feature1],
#                         H=H[feature0, feature1],
#                         eps=EPS
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

#                 # Mixture
#                 mixing_distribution = dist.Categorical(
#                     probs=jnp.stack([1 - q, q], axis=-1)
#                 )
#                 component_distributions=[
#                     dist.Gamma(concentration=alpha, rate=beta),
#                     dist.HalfNormal(scale=L[feature0, feature1] + H[feature0, feature1])
#                 ]
#                 Mixture = dist.MixtureGeneral(
#                     mixing_distribution=mixing_distribution,
#                     component_distributions=component_distributions
#                 )

#                 # Observation
#                 numpyro.sample(
#                     site.obs,
#                     Mixture,
#                     obs=response_obs
#                 )


# class HierarchicalBayesianModel(GammaModel):
#     NAME = "2"

#     def __init__(self, config: Config):
#         super(HierarchicalBayesianModel, self).__init__(config=config)
#         self.mcmc_params = {
#             "num_warmup": 20000,
#             "num_samples": 20000,
#             "num_chains": 4,
#             "thinning": 20,
#         }
#         self.run_kwargs = {
#             "max_tree_depth": (20, 20),
#             "target_accept_prob": .95,
#             "extra_fields": [
#                 "potential_energy",
#                 "num_steps",
#                 "accept_prob",
#             ]
#         }
#         self.NAME += (
#             f'_{self.mcmc_params["num_warmup"]}W'
#             f'_{self.mcmc_params["num_samples"]}S'
#             f'_{self.run_kwargs["max_tree_depth"][0]}D'
#             f'_{self.run_kwargs["target_accept_prob"]}A'
#         )

#     def _model(self, intensity, features, response_obs=None):
#         n_data = intensity.shape[0]
#         n_features = np.max(features, axis=0) + 1
#         feature0 = features[..., 0]
#         feature1 = features[..., 1]

#         n_fixed = 1
#         n_delta = n_features[1] - 1

#         # Fixed
#         a_fixed_loc = numpyro.sample(
#             "a_fixed_loc", dist.TruncatedNormal(5., 10., low=0.)
#         )
#         a_fixed_scale = numpyro.sample(
#             "a_fixed_scale", dist.HalfNormal(10.)
#         )

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate("n_fixed", n_fixed):
#                 with numpyro.plate(site.n_features[0], n_features[0]):
#                     a_fixed = numpyro.sample(
#                         "a_fixed", dist.TruncatedNormal(
#                             a_fixed_loc, a_fixed_scale, low=0.
#                         )
#                     )

#         # Delta
#         a_delta_scale_scale = numpyro.sample(
#             "a_delta_scale_scale", dist.HalfNormal(10.)
#         )

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate("n_delta", n_delta):
#                 a_delta_scale = numpyro.sample(
#                     "a_delta_scale", dist.HalfNormal(a_delta_scale_scale)
#                 )
#                 a_delta_loc = numpyro.sample(
#                     "a_delta_loc", dist.Normal(0., 10.)
#                 )

#                 with numpyro.plate(site.n_features[0], n_features[0]):
#                     a_delta_raw = numpyro.sample("a_delta_raw", dist.Normal(0., 1.))
#                     a_delta = numpyro.deterministic(
#                         "a_delta", a_delta_loc + a_delta_raw * a_delta_scale
#                     )

#                     # Penalty for negative a
#                     penalty_for_negative_a = (
#                         jnp.fabs(a_fixed + a_delta) - (a_fixed + a_delta)
#                     )
#                     numpyro.factor(
#                         "penalty_for_negative_a", -penalty_for_negative_a
#                     )
#                     a_fixed_plus_delta = jax.nn.softplus(a_fixed + a_delta)

#         # Hyper-priors
#         b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

#         L_scale = numpyro.sample("L_scale", dist.HalfNormal(.1))
#         ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(1.))
#         H_scale = numpyro.sample("H_scale", dist.HalfNormal(10.))

#         c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
#         c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(.5))

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate(site.n_features[1], n_features[1]):
#                 with numpyro.plate(site.n_features[0], n_features[0]):
#                     # Priors
#                     a = numpyro.deterministic(
#                         site.a,
#                         jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=1)
#                     )

#                     b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
#                     b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

#                     L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
#                     L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

#                     ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
#                     ell = numpyro.deterministic(site.ell, jnp.multiply(ell_scale, ell_raw))

#                     H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
#                     H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

#                     c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
#                     c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

#                     c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
#                     c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

#         q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

#         with numpyro.plate(site.n_response, self.n_response):
#             with numpyro.plate(site.n_data, n_data):
#                 # Model
#                 mu = numpyro.deterministic(
#                     site.mu,
#                     S.rectified_logistic(
#                         x=intensity,
#                         a=a[feature0, feature1],
#                         b=b[feature0, feature1],
#                         L=L[feature0, feature1],
#                         ell=ell[feature0, feature1],
#                         H=H[feature0, feature1],
#                         eps=EPS
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

#                 # Mixture
#                 mixing_distribution = dist.Categorical(
#                     probs=jnp.stack([1 - q, q], axis=-1)
#                 )
#                 component_distributions=[
#                     dist.Gamma(concentration=alpha, rate=beta),
#                     dist.HalfNormal(scale=L[feature0, feature1] + H[feature0, feature1])
#                 ]
#                 Mixture = dist.MixtureGeneral(
#                     mixing_distribution=mixing_distribution,
#                     component_distributions=component_distributions
#                 )

#                 # Observation
#                 numpyro.sample(
#                     site.obs,
#                     Mixture,
#                     obs=response_obs
#                 )


class HierarchicalBayesianModel(GammaModel):
    NAME = "3"

    def __init__(self, config: Config):
        super(HierarchicalBayesianModel, self).__init__(config=config)
        self.mcmc_params = {
            "num_warmup": 1600,
            "num_samples": 1600,
            "num_chains": 4,
            "thinning": 1,
        }
        self.run_kwargs = {
            "max_tree_depth": (15, 15),
            "target_accept_prob": .95,
            "extra_fields": [
                "potential_energy",
                "num_steps",
                "accept_prob",
            ]
        }
        self.NAME += (
            f'_{self.mcmc_params["num_warmup"]}W'
            f'_{self.mcmc_params["num_samples"]}S'
            f'_{self.run_kwargs["max_tree_depth"][0]}D'
            f'_{self.run_kwargs["target_accept_prob"]}A'
        )

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_delta = n_features[1] - 1

        # Fixed
        a_fixed_loc = numpyro.sample(
            "a_fixed_loc", dist.TruncatedNormal(5., 10., low=0.)
        )
        a_fixed_scale_scale = numpyro.sample(
            "a_fixed_scale_scale", dist.HalfNormal(10.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_scale = numpyro.sample(
                    "a_fixed_scale", dist.HalfNormal(a_fixed_scale_scale)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_loc, a_fixed_scale, low=0.
                        )
                    )

        # Delta
        a_delta_scale_scale = numpyro.sample(
            "a_delta_scale_scale", dist.HalfNormal(10.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                a_delta_scale = numpyro.sample(
                    "a_delta_scale", dist.HalfNormal(a_delta_scale_scale)
                )
                a_delta_loc = numpyro.sample(
                    "a_delta_loc", dist.Normal(0., 10.)
                )

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_delta_raw = numpyro.sample("a_delta_raw", dist.Normal(0., 1.))
                    a_delta = numpyro.deterministic(
                        "a_delta", a_delta_loc + a_delta_raw * a_delta_scale
                    )

                    # Penalty for negative a
                    penalty_for_negative_a = (
                        jnp.fabs(a_fixed + a_delta) - (a_fixed + a_delta)
                    )
                    numpyro.factor(
                        "penalty_for_negative_a", -penalty_for_negative_a
                    )
                    a_fixed_plus_delta = jax.nn.softplus(a_fixed + a_delta)

        # Hyper-priors
        b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

        L_scale = numpyro.sample("L_scale", dist.HalfNormal(.1))
        ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(1.))
        H_scale = numpyro.sample("H_scale", dist.HalfNormal(10.))

        c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
        c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(.5))

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

        q = numpyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    S.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1],
                        eps=EPS
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
                mixing_distribution = dist.Categorical(
                    probs=jnp.stack([1 - q, q], axis=-1)
                )
                component_distributions=[
                    dist.Gamma(concentration=alpha, rate=beta),
                    dist.HalfNormal(scale=L[feature0, feature1] + H[feature0, feature1])
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
