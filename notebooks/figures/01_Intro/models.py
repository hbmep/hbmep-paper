import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import GammaModel, BoundedOptimization
from hbmep.model.utils import Site as site


class RectifiedLogistic(GammaModel):
    NAME = "rectified_logistic"

    def __init__(self, config: Config):
        super(RectifiedLogistic, self).__init__(config=config)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]

        with numpyro.plate(site.n_response, self.n_response):
            # Hyper Priors
            a_loc = numpyro.sample("a_loc", dist.TruncatedNormal(50., 20., low=0))
            a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

            b_scale = numpyro.sample("b_scale", dist.HalfNormal(5.))

            L_scale = numpyro.sample("L_scale", dist.HalfNormal(.5))
            ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(10.))
            H_scale = numpyro.sample("H_scale", dist.HalfNormal(5.))

            c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(5.))
            c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(5.))

            with numpyro.plate(site.n_features[0], n_features[0]):
                # Priors
                a = numpyro.sample(
                    site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
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

        s50 = numpyro.deterministic(
            "s50",
            a
            - jnp.true_divide(
                jnp.log(jnp.multiply(
                    jnp.true_divide(ell, H),
                    - 1
                    + jnp.true_divide(
                        H + ell,
                        jnp.true_divide(H, 2) + ell
                    )
                )),
                b
            )
        )

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                # Model
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0],
                        b=b[feature0],
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

                # Observation
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=alpha, rate=beta),
                    obs=response_obs
                )


class NelderMeadOptimization(BoundedOptimization):
    NAME = "nelder_mead"

    def __init__(self, config: Config):
        super(NelderMeadOptimization, self).__init__(config=config)
        self.solver = "Nelder-Mead"
        self.functional = F.logistic4      # a, b, L, H
        self.named_params = [site.a, site.b, site.L, site.H]
        self.bounds = [(1e-9, 150.), (1e-9, 10), (1e-9, 10), (1e-9, 10)]
        self.informed_bounds = [(20, 80), (1e-3, 5.), (1e-4, .1), (.5, 5)]
        self.num_points = 1000
        self.num_iters = 5000
        self.n_jobs = -1

    # def cost_function(self, x, y, *args):
    #     y_pred = self.fn(x, *args)
    #     cost = np.sum((y - y_pred) ** 2)
    #     return cost

    # def optimize(self, x, y, param, dest):
    #     res = minimize(
    #         lambda coeffs: self.cost_function(x, y, *coeffs),
    #         x0=param,
    #         bounds=self.bounds,
    #         method=self.solver
    #     )
    #     with open(dest, "wb") as f:
    #         pickle.dump((res,), f)

    # @timing
    # def run_inference(self, df: pd.DataFrame):
    #     # Intialize fresh directory for storing optimiztion results
    #     self._make_dir(self.build_dir)
    #     results_dir = os.path.join(self.build_dir, "optimize_results")
    #     if os.path.exists(results_dir): shutil.rmtree(results_dir)
    #     assert not os.path.exists(results_dir)
    #     self._make_dir(results_dir)

    #     x = df[self.intensity].values
    #     rng_keys = random.split(self.rng_key, num=len(self.bounds))
    #     rng_keys = list(rng_keys)
    #     # rng_keys = np.array(rng_keys).reshape(len(self.bounds), self.n_response)

    #     response_dir_prefix = "response"
    #     minimize_result = []
    #     for r, response in enumerate(self.response):
    #         # Initialize response directory
    #         response_dir = os.path.join(results_dir, f"{response_dir_prefix}{r}")
    #         self._make_dir(response_dir)

    #         # Grid space
    #         grid = [np.linspace(lo, hi, self.num_points) for lo, hi in self.informed_bounds]
    #         # grid = [
    #         #     random.choice(key=rng_key, a=arr, shape=(self.n_repeats,), replace=True)
    #         #     for arr, rng_key in zip(grid, rng_keys[:, r])
    #         # ]
    #         grid = [
    #             random.choice(key=rng_key, a=arr, shape=(self.n_repeats,), replace=True)
    #             for arr, rng_key in zip(grid, rng_keys)
    #         ]
    #         grid = [np.array(arr).tolist() for arr in grid]
    #         grid = list(zip(*grid))

    #         y = df[response].values

    #         with Parallel(n_jobs=self.n_jobs) as parallel:
    #             parallel(
    #                 delayed(self.optimize)(x, y, param, os.path.join(response_dir, f"param{i}.pkl"))
    #                 for i, param in enumerate(grid)
    #             )

    #         res = []
    #         for i, _ in enumerate(grid):
    #             src = os.path.join(response_dir, f"param{i}.pkl")
    #             with open(src, "rb") as g:
    #                 res.append(pickle.load(g)[0])

    #         errors = [r.fun for r in res]
    #         argmin = np.argmin(errors)
    #         logger.info(f"Optimal params for response {response}:")
    #         logger.info(res[argmin])
    #         minimize_result.append(res[argmin])

    #     return minimize_result

    # @timing
    # def render_recruitment_curves(self, df, prediction_df, minimize_result):
    #     nrows, ncols = 1, self.n_response
    #     fig, axes = plt.subplots(nrows, ncols, figsize=(self.subplot_cell_width * ncols, self.subplot_cell_height * nrows), squeeze=False, constrained_layout=True)
    #     for r, response in enumerate(self.response):
    #         ax = axes[0, r]
    #         params = minimize_result[r].x
    #         sns.scatterplot(x=df[self.intensity], y=df[response], ax=ax)
    #         sns.lineplot(x=prediction_df[self.intensity], y=self.fn(prediction_df[self.intensity].values, *params), color=self.response_colors[r], ax=ax)

    #     dest = os.path.join(self.build_dir, "recruitment_curves.png")
    #     fig.savefig(dest)
    #     logger.info(f"Saved to {dest}")
    #     plt.close(fig)
    #     return


# class BestPest(BaseModel):
#     NAME = "best_pest"

#     def __init__(self, config: Config):
#         super(BestPest, self).__init__(config=config)
#         self.solver = "Nelder-Mead"
#         self.fn = stats.norm.cdf      # a, b, v, L, ell, H
#         self.params = ["loc", "scale"]
#         self.bounds = [(1e-9, 150.), (1e-9, 10)]
#         self.informed_bounds = [(20, 80), (1e-3, 5)]
#         # self.params = ["loc"]
#         # self.bounds = [(1e-9, 150.)]
#         # self.informed_bounds = [(20, 80)]
#         self.num_points = 1000
#         self.n_repeats = 5000
#         self.n_jobs = -1

#     def cost_function(self, x, y, *args):
#         # y_pred = self.fn(x, *args)
#         # cost = np.sum((y - y_pred) ** 2)
#         # cost = - np.sum(y * np.log(self.fn(x, *args))) + np.sum((1 - y) * np.log(1 - self.fn(x, *args)))
#         epsilon = 1e-9
#         cost = np.where(
#             y == 1,
#             - np.log(self.fn(x, *args) + epsilon),
#             - np.log(1 - self.fn(x, *args) + epsilon)
#         )
#         cost = np.sum(cost)
#         # yp = self.fn(x, *args)
#         # cost = - np.average(y * np.log(yp + epsilon) + (1. - y) * np.log(1. - yp + epsilon))
#         return cost

#     def optimize(self, x, y, param, dest):
#         res = minimize(
#             lambda coeffs: self.cost_function(x, y, *coeffs),
#             x0=param,
#             bounds=self.bounds,
#             method=self.solver
#         )
#         with open(dest, "wb") as f:
#             pickle.dump((res,), f)

#     @timing
#     def run_inference(self, df: pd.DataFrame):
#         # Intialize fresh directory for storing optimiztion results
#         self._make_dir(self.build_dir)
#         results_dir = os.path.join(self.build_dir, "optimize_results")
#         if os.path.exists(results_dir): shutil.rmtree(results_dir)
#         assert not os.path.exists(results_dir)
#         self._make_dir(results_dir)

#         x = df[self.intensity].values
#         rng_keys = random.split(self.rng_key, num=len(self.bounds))
#         rng_keys = list(rng_keys)
#         # rng_keys = np.array(rng_keys).reshape(len(self.bounds), self.n_response)

#         response_dir_prefix = "response"
#         minimize_result = []
#         for r, response in enumerate(self.response):
#             # Initialize response directory
#             response_dir = os.path.join(results_dir, f"{response_dir_prefix}{r}")
#             self._make_dir(response_dir)

#             # Grid space
#             grid = [np.linspace(lo, hi, self.num_points) for lo, hi in self.informed_bounds]
#             # grid = [
#             #     random.choice(key=rng_key, a=arr, shape=(self.n_repeats,), replace=True)
#             #     for arr, rng_key in zip(grid, rng_keys[:, r])
#             # ]
#             grid = [
#                 random.choice(key=rng_key, a=arr, shape=(self.n_repeats,), replace=True)
#                 for arr, rng_key in zip(grid, rng_keys)
#             ]
#             grid = [np.array(arr).tolist() for arr in grid]
#             grid = list(zip(*grid))

#             y = df[response].values

#             with Parallel(n_jobs=self.n_jobs) as parallel:
#                 parallel(
#                     delayed(self.optimize)(x, y, param, os.path.join(response_dir, f"param{i}.pkl"))
#                     for i, param in enumerate(grid)
#                 )

#             res = []
#             for i, _ in enumerate(grid):
#                 src = os.path.join(response_dir, f"param{i}.pkl")
#                 with open(src, "rb") as g:
#                     res.append(pickle.load(g)[0])

#             errors = [r.fun for r in res]
#             argmin = np.argmin(errors)
#             logger.info(f"Optimal params for response {response}:")
#             logger.info(res[argmin])
#             minimize_result.append(res[argmin])

#         return minimize_result

#     @timing
#     def render_recruitment_curves(self, df, prediction_df, minimize_result):
#         nrows, ncols = 1, self.n_response
#         fig, axes = plt.subplots(nrows, ncols, figsize=(self.subplot_cell_width * ncols, self.subplot_cell_height * nrows), squeeze=False, constrained_layout=True)
#         for r, response in enumerate(self.response):
#             ax = axes[0, r]
#             params = minimize_result[r].x
#             sns.scatterplot(x=df[self.intensity], y=df[response], ax=ax)
#             sns.lineplot(x=prediction_df[self.intensity], y=self.fn(prediction_df[self.intensity].values, *params), color=self.response_colors[r], ax=ax)

#         dest = os.path.join(self.build_dir, "recruitment_curves.png")
#         fig.savefig(dest)
#         logger.info(f"Saved to {dest}")
#         plt.close(fig)
#         return
