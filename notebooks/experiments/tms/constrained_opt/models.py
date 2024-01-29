import os
import shutil
import pickle
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

import jax
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel, GammaModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site
from hbmep.utils import timing

logger = logging.getLogger(__name__)


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


class NelderMeadOptimization(BaseModel):
    NAME = "nelder_mead"

    def __init__(self, config: Config):
        super(NelderMeadOptimization, self).__init__(config=config)
        self.solver = "Nelder-Mead"
        self.fn = F.rectified_logistic      # a, b, v, L, ell, H
        self.params = [site.a, site.b, site.v, site.L, site.ell, site.H]
        self.bounds = [(1e-9, 150.), (1e-9, 10), (1e-9, 10), (1e-9, 10), (1e-9, 10), (1e-9, 10)]
        self.informed_bounds = [(20, 80), (1e-3, 5.), (1e-3, 5.), (1e-4, .1), (1e-2, 5), (.5, 5)]
        self.num_points = 1000
        self.n_repeats = 100
        self.n_jobs = -1

    def cost_function(self, x, y, *args):
        y_pred = self.fn(x, *args)
        return np.sum((y - y_pred) ** 2)

    def optimize(self, x, y, param, dest):
        res = minimize(
            lambda coeffs: self.cost_function(x, y, *coeffs),
            x0=param,
            bounds=self.bounds,
            method=self.solver
        )
        with open(dest, "wb") as f:
            pickle.dump((res,), f)

    @timing
    def run_inference(self, df: pd.DataFrame):
        x, y = df[self.intensity].values, df[self.response[0]].values
        rng_keys = jax.random.split(self.rng_key, num=len(self.bounds))
        rng_keys = list(rng_keys)
        grid = [np.linspace(lo, hi, self.num_points) for lo, hi in self.informed_bounds]
        grid = [
            jax.random.choice(key=rng_key, a=arr, shape=(self.n_repeats,), replace=True)
            for arr, rng_key in zip(grid, rng_keys)
        ]
        grid = [np.array(arr).tolist() for arr in grid]
        grid = list(zip(*grid))

        self._make_dir(self.build_dir)
        results_dir = os.path.join(self.build_dir, "optimize_results")
        if os.path.exists(results_dir): shutil.rmtree(results_dir)
        assert not os.path.exists(results_dir)
        self._make_dir(results_dir)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel(
                delayed(self.optimize)(x, y, param, os.path.join(results_dir, f"param{i}.pkl"))
                for i, param in enumerate(grid)
            )

        # res = []
        # for param in grid:
        #     res.append(
        #         minimize(
        #             lambda coeffs: self.cost_function(x, y, *coeffs),
        #             x0=param,
        #             bounds=self.bounds,
        #             method=self.solver
        #         )
        #     )

        # params = [r.x for r in res]
        # errors = [r.fun for r in res]
        # argmin = np.argmin(errors)
        # logger.info("Optimal params:")
        # logger.info(res[argmin])
        # return params[argmin]

        res = []
        for i, _ in enumerate(grid):
            src = os.path.join(results_dir, f"param{i}.pkl")
            with open(src, "rb") as g:
                res.append(pickle.load(g)[0])

        params = [r.x for r in res]
        errors = [r.fun for r in res]
        argmin = np.argmin(errors)
        logger.info("Optimal params:")
        logger.info(res[argmin])
        logger.info(type(params[argmin]))
        logger.info(type(params[argmin][0]))
        return params[argmin]

    @timing
    def render_recruitment_curves(self, df, prediction_df, params):
        fig, axes = plt.subplots(1, 1, figsize=(8, 6), squeeze=False, constrained_layout=True)
        ax = axes[0, 0]
        sns.scatterplot(x=df[self.intensity], y=df[self.response[0]], ax=ax)
        sns.lineplot(x=prediction_df[self.intensity], y=self.fn(prediction_df[self.intensity].values, *params), color="k", ax=ax)
        fig.savefig(os.path.join(self.build_dir, "recruitment_curves.png"))
        return
