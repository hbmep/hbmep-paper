import numpy as np
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer import NUTS, MCMC, Predictive

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel, BoundedOptimization
from hbmep.model.utils import Site as site


class HB():
    NAME = "HB"

    def __init__(self):
        self.rng_key = random.PRNGKey(0)
        self.build_dir = None
        self.mcmc_params = {
            "num_chains": 4,
            "num_warmup": 5000,
            "num_samples": 1000,
            "thinning": 1,
        }

    def _model(self, y_obs=None):
        n_features0 = y_obs.shape[0]
        n_response = y_obs.shape[-1]

        y_loc_loc = numpyro.sample(
            "y_loc_loc", dist.Normal(0., 50.)
        )
        y_loc_scale = numpyro.sample(
            "y_loc_scale", dist.HalfNormal(50.)
        )
        y_scale = numpyro.sample(
            "y_scale", dist.HalfNormal(50.)
        )

        with numpyro.plate("n_response", n_response):
            y_loc = numpyro.sample(
                "y_loc",
                dist.Normal(y_loc_loc, y_loc_scale)
            )

            with numpyro.plate("n_features0", n_features0):
                numpyro.sample(
                    "obs",
                    dist.Normal(y_loc, y_scale),
                    obs=y_obs
                )

    def run_inference(
        self,
        y_obs: np.ndarray,
        **kwargs
    ) -> tuple[MCMC, dict]:
        # Set up sampler
        sampler = NUTS(self._model, **kwargs)
        mcmc = MCMC(sampler, **self.mcmc_params)

        # Run MCMC inference
        mcmc.run(self.rng_key, y_obs=y_obs)
        posterior_samples = mcmc.get_samples()
        posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
        return mcmc, posterior_samples


class HBTight_above_sigma_g(HB):
    NAME = "HBTight_above_sigma_g"

    def __init__(self):
        super(HBTight_above_sigma_g, self).__init__()

    def _model(self, y_obs=None):
        n_features0 = y_obs.shape[0]
        n_response = y_obs.shape[-1]

        y_loc_loc = numpyro.sample(
            "y_loc_loc", dist.Normal(0., 0.1)
        )
        y_loc_scale = numpyro.sample(
            "y_loc_scale", dist.HalfNormal(50.)
        )
        y_scale = numpyro.sample(
            "y_scale", dist.HalfNormal(50.)
        )

        with numpyro.plate("n_response", n_response):
            y_loc = numpyro.sample(
                "y_loc",
                dist.Normal(y_loc_loc, y_loc_scale)
            )

            with numpyro.plate("n_features0", n_features0):
                numpyro.sample(
                    "obs",
                    dist.Normal(y_loc, y_scale),
                    obs=y_obs
                )


class HBTight_sigma_g(HB):
    NAME = "HBTight_sigma_g"

    def __init__(self):
        super(HBTight_sigma_g, self).__init__()

    def _model(self, y_obs=None):
        n_features0 = y_obs.shape[0]
        n_response = y_obs.shape[-1]

        y_loc_loc = numpyro.sample(
            "y_loc_loc", dist.Normal(0., 50.)
        )
        y_loc_scale = numpyro.sample(
            "y_loc_scale", dist.HalfNormal(0.1)
        )
        y_scale = numpyro.sample(
            "y_scale", dist.HalfNormal(50.)
        )

        with numpyro.plate("n_response", n_response):
            y_loc = numpyro.sample(
                "y_loc",
                dist.Normal(y_loc_loc, y_loc_scale)
            )

            with numpyro.plate("n_features0", n_features0):
                numpyro.sample(
                    "obs",
                    dist.Normal(y_loc, y_scale),
                    obs=y_obs
                )


class HBZero(HB):
    NAME = "HBZero"

    def __init__(self):
        super(HBZero, self).__init__()

    def _model(self, y_obs=None):
        n_features0 = y_obs.shape[0]
        n_response = y_obs.shape[-1]

        # y_loc_loc = numpyro.sample(
        #     "y_loc_loc", dist.Normal(0., 50.)
        # )
        y_loc_scale = numpyro.sample(
            "y_loc_scale", dist.HalfNormal(50.)
        )
        y_scale = numpyro.sample(
            "y_scale", dist.HalfNormal(50.)
        )

        with numpyro.plate("n_response", n_response):
            y_loc = numpyro.sample(
                "y_loc",
                dist.Normal(0., y_loc_scale)
            )

            with numpyro.plate("n_features0", n_features0):
                numpyro.sample(
                    "obs",
                    dist.Normal(y_loc, y_scale),
                    obs=y_obs
                )


class Simulator(GammaModel):
    NAME = "simulator"

    def __init__(self, config: Config, a_delta_loc, a_delta_scale):
        super(Simulator, self).__init__(config=config)
        self.a_delta_loc = a_delta_loc
        self.a_delta_scale = a_delta_scale

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        n_fixed = 1
        n_delta = n_features[1] - 1

        # Fixed
        a_fixed_loc = numpyro.sample(
            "a_fixed_loc", dist.TruncatedNormal(50., 50., low=0)
        )
        a_fixed_scale = numpyro.sample(
            "a_fixed_scale", dist.HalfNormal(50.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(
                            a_fixed_loc, a_fixed_scale, low=0
                        )
                    )

        # Delta
        a_delta_loc, a_delta_scale = self.a_delta_loc, self.a_delta_scale

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_delta", n_delta):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_delta = numpyro.sample(
                        "a_delta", dist.Normal(a_delta_loc, a_delta_scale)
                    )

                    # Penalty for negative a
                    penalty_for_negative_a = (
                        jnp.fabs(a_fixed + a_delta) - (a_fixed + a_delta)
                    )
                    numpyro.factor(
                        "penalty_for_negative_a", -penalty_for_negative_a
                    )

        # Hyper-priors
        b_scale = numpyro.sample(
            "b_scale", dist.HalfNormal(5.)
        )

        L_scale = numpyro.sample(
            "L_scale", dist.HalfNormal(.5)
        )
        ell_scale = numpyro.sample(
            "ell_scale", dist.HalfNormal(10.)
        )
        H_scale = numpyro.sample(
            "H_scale", dist.HalfNormal(5.)
        )

        c_1_scale = numpyro.sample(
            "c_1_scale", dist.HalfNormal(5.)
        )
        c_2_scale = numpyro.sample(
            "c_2_scale", dist.HalfNormal(5.)
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    # Priors
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_delta], axis=1)
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
