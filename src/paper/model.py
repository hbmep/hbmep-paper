import logging

import numpy as np
import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

from hbmep import functional as F, smooth_functional as SF
from hbmep.model import BaseModel
from hbmep.util import site

from paper.util import get_subname

logger = logging.getLogger(__name__)
EPS = 1e-3


class HB(BaseModel):
    def __init__(self, *args, **kw):
        super(HB, self).__init__(*args, **kw)
        self.run_id = None
        self.use_mixture = False

    @property
    def name(self): return get_subname(self)

    @name.setter
    def name(self, value): return value

    def rectified_logistic(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        run_id = self.run_id

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        match run_id:
            case "rat":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(150., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(150.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
            
            case "tms":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(50., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "intraoperative":
                a_loc = pyro.sample(
                    site.a.log, dist.TruncatedNormal(5., 10., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))

            case _:
                raise ValueError

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

                    if self.use_mixture:
                        log_probs = Mixture.component_log_probs(y_)
                        pyro.deterministic(
                            "p", log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            )
                        )

    def logistic5(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        run_id = self.run_id

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        match run_id:
            case "rat":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(150., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(150.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
            
            case "tms":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(50., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "intraoperative":
                a_loc = pyro.sample(
                    site.a.log, dist.TruncatedNormal(5., 10., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))

            case _:
                raise ValueError

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        F.logistic5,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            # EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

                    if self.use_mixture:
                        log_probs = Mixture.component_log_probs(y_)
                        pyro.deterministic(
                            "p", log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            )
                        )

    def logistic4(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        run_id = self.run_id

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        match run_id:
            case "rat":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(150., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(150.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
            
            case "tms":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(50., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "intraoperative":
                a_loc = pyro.sample(
                    site.a.log, dist.TruncatedNormal(5., 10., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))

            case _:
                raise ValueError

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                # v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                # v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        F.logistic4,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            # v[*features.T],
                            # EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

                    if self.use_mixture:
                        # Thanks to https://dfm.io/posts/intro-to-numpyro/
                        # Until here, where we can track the membership probability of each sample
                        log_probs = Mixture.component_log_probs(y_)
                        pyro.deterministic(
                            "p", log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            )
                        )

    def rectified_linear(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        run_id = self.run_id

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        match run_id:
            case "rat":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(150., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(150.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "tms":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(50., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "intraoperative":
                a_loc = pyro.sample(
                    site.a.log, dist.TruncatedNormal(5., 10., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
                # h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))
            
            case _:
                raise ValueError

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        # v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                # h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                # h = pyro.deterministic(site.h, h_scale * h_raw)

                # v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                # v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture: raise ValueError

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        F.rectified_linear,
                        intensity,
                        (
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            # h[*features.T],
                            # v[*features.T],
                            # EPS,
                        ),
                        c1[*features.T],
                        c2[*features.T],
                    )
                    pyro.deterministic(site.mu, mu)

                    # Observations
                    pyro.sample(
                        site.obs,
                        dist.Gamma(concentration=alpha, rate=beta),
                        obs=response
                    )

    def lognormal_rlog(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        run_id = self.run_id

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        match run_id:
            case "rat":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(150., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(150.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
            
            case "tms":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(50., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "intraoperative":
                a_loc = pyro.sample(
                    site.a.log, dist.TruncatedNormal(5., 10., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))

            case _:
                raise ValueError

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(.5))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = pyro.deterministic(
                        site.mu,
                        SF.rectified_logistic(
                            intensity,
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            EPS
                        )
                    )
                    loc = jnp.log(mu)
                    scale = c1[*features.T] + c2[*features.T] * mu

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.LogNormal(loc=loc, scale=scale),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.LogNormal(loc=loc, scale=scale)
                        ),
                        obs=response
                    )

                    if self.use_mixture:
                        log_probs = Mixture.component_log_probs(y_)
                        pyro.deterministic(
                            "p", log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            )
                        )

    def ln_rlog(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        run_id = self.run_id

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        match run_id:
            case "rat":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(150., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(150.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
            
            case "tms":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(50., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "intraoperative":
                a_loc = pyro.sample(
                    site.a.log, dist.TruncatedNormal(5., 10., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))

            case _:
                raise ValueError

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(5.))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = pyro.deterministic(
                        site.mu,
                        SF.rectified_logistic(
                            intensity,
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            EPS
                        )
                    )
                    loc = jnp.log(mu)
                    scale = c1[*features.T] + c2[*features.T] * mu

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.LogNormal(loc=loc, scale=scale),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.LogNormal(loc=loc, scale=scale)
                        ),
                        obs=response
                    )

                    if self.use_mixture:
                        log_probs = Mixture.component_log_probs(y_)
                        pyro.deterministic(
                            "p", log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            )
                        )

    def normal_rlog(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        run_id = self.run_id

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        match run_id:
            case "rat":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(150., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(150.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
            
            case "tms":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(50., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "intraoperative":
                a_loc = pyro.sample(
                    site.a.log, dist.TruncatedNormal(5., 10., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))

            case _:
                raise ValueError

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(5.))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = pyro.deterministic(
                        site.mu,
                        SF.rectified_logistic(
                            intensity,
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            EPS
                        )
                    )
                    loc = mu
                    scale = c1[*features.T] + c2[*features.T] * mu

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Normal(loc=loc, scale=scale),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Normal(loc=loc, scale=scale)
                        ),
                        obs=response
                    )

                    if self.use_mixture:
                        log_probs = Mixture.component_log_probs(y_)
                        pyro.deterministic(
                            "p", log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            )
                        )

    def constln_rlog(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1
        run_id = self.run_id

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        match run_id:
            case "rat":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(150., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(150.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))
            
            case "tms":
                a_loc = pyro.sample(
                    site.a.loc, dist.TruncatedNormal(50., 50., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(50.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(1.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(5.))

            case "intraoperative":
                a_loc = pyro.sample(
                    site.a.log, dist.TruncatedNormal(5., 10., low=0)
                )
                a_scale = pyro.sample(site.a.scale, dist.HalfNormal(10.))
                b_scale = pyro.sample(site.b.scale, dist.HalfNormal(5.))
                h_scale = pyro.sample(site.h.scale, dist.HalfNormal(10.))

            case _:
                raise ValueError

        g_scale = pyro.sample(site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(site.v.scale, dist.HalfNormal(1.))

        c1_scale = pyro.sample(site.c1.scale, dist.HalfNormal(5.))
        # c2_scale = pyro.sample(site.c2.scale, dist.HalfNormal(5.))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            with pyro.plate_stack(
                site.num_features, num_features, rightmost_dim=-2
            ):
                a = pyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )

                b_raw = pyro.sample(site.b.raw, dist.HalfNormal(1))
                b = pyro.deterministic(site.b, b_scale * b_raw)

                g_raw = pyro.sample(site.g.raw, dist.HalfNormal(1))
                g = pyro.deterministic(site.g, g_scale * g_raw)

                h_raw = pyro.sample(site.h.raw, dist.HalfNormal(1))
                h = pyro.deterministic(site.h, h_scale * h_raw)

                v_raw = pyro.sample(site.v.raw, dist.HalfNormal(1))
                v = pyro.deterministic(site.v, v_scale * v_raw)

                c1_raw = pyro.sample(site.c1.raw, dist.HalfNormal(1))
                c1 = pyro.deterministic(site.c1, c1_scale * c1_raw)

                # c2_raw = pyro.sample(site.c2.raw, dist.HalfNormal(1))
                # c2 = pyro.deterministic(site.c2, c2_scale * c2_raw)

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = pyro.deterministic(
                        site.mu,
                        SF.rectified_logistic(
                            intensity,
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            EPS
                        )
                    )
                    loc = jnp.log(mu)
                    # scale = c1[*features.T] + c2[*features.T] * mu
                    scale = c1[*features.T]

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.LogNormal(loc=loc, scale=scale),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.LogNormal(loc=loc, scale=scale)
                        ),
                        obs=response
                    )

                    if self.use_mixture:
                        log_probs = Mixture.component_log_probs(y_)
                        pyro.deterministic(
                            "p", log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            )
                        )
