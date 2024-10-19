import os
import shutil
import pickle

import numpy as np
import jax.numpy as jnp
from jax import random
from joblib import Parallel, delayed

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

from hbmep.model.bounded_optimization import abstractvariables
from hbmep.utils import timing


class BaseModel():
    NAME = "base_model"

    def __init__(self):
        self.rng_key = random.PRNGKey(0)
        self.build_dir = None
        self.mcmc_params = {
            "num_chains": 4,
            "num_warmup": 2000,
            "num_samples": 1000,
            "thinning": 1,
        }
        self.n_response = None

    @timing
    def run_inference(
        self,
        subject_ind,
        group_ind,
        y_obs,
        **kwargs
    ) -> tuple[MCMC, dict]:
        # Set up sampler
        sampler = NUTS(self._model, **kwargs)
        mcmc = MCMC(sampler, **self.mcmc_params)

        # Run MCMC inference
        self.n_tests = y_obs.shape[-1]
        mcmc.run(self.rng_key, subject_ind, group_ind, y_obs=y_obs)
        posterior_samples = mcmc.get_samples()
        posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
        return mcmc, posterior_samples


class HB(BaseModel):
    NAME = "HB"

    def __init__(self):
        super(HB, self).__init__()

    def _model(self, subject_ind, group_ind, y_obs=None):
        n_subjects = np.max(subject_ind) + 1
        n_groups = np.max(group_ind) + 1

        n_tests = self.n_tests
        n_data = subject_ind.shape[0]

        # a_loc_loc_scale = numpyro.sample('a_loc_loc_scale', dist.HalfNormal(50.))
        a_loc_scale = numpyro.sample('a_loc_scale', dist.HalfNormal(50.))
        a_scale = numpyro.sample('a_scale', dist.HalfNormal(50.))

        scale = numpyro.sample('scale', dist.HalfNormal(5))
        a_loc_loc = numpyro.sample('a_loc_loc', dist.Normal(50., 50.))

        with numpyro.plate('n_tests', n_tests):
            with numpyro.plate("n_groups", n_groups):
                a_loc = numpyro.sample('a_loc', dist.Normal(a_loc_loc, a_loc_scale))

                with numpyro.plate("n_subjects", n_subjects):
                    a = numpyro.sample('a', dist.Normal(a_loc, a_scale))

        with numpyro.plate('n_tests', n_tests):
            with numpyro.plate('n_data', n_data):
                numpyro.sample('obs', dist.Normal(a[subject_ind, group_ind], scale), obs=y_obs)


@abstractvariables(
    ("n_jobs", "Number of parallel jobs not specified.")
)
class NonHierarchicalBaseModel(BaseModel):
    NAME = "non_hierarchical_base_model"

    def __init__(self):
        super(NonHierarchicalBaseModel, self).__init__()

    @timing
    def run_inference(
        self,
        subject_ind,
        group_ind,
        y_obs,
        **kwargs
    ):
        n_response = y_obs.shape[-1]
        combinations = list(set(zip(subject_ind, group_ind)))
        temp_dir = os.path.join(self.build_dir, "optimize_results")

        def body_run_inference(combination, response, destination_path):
            ind = [c == combination for c in list(zip(subject_ind, group_ind))]
            subject_ind_ = subject_ind[ind]
            group_ind_ = group_ind[ind]
            subject_ind_ = 0 * subject_ind_
            group_ind_ = 0 * group_ind_
            y_obs_ = y_obs[ind, ...]
            y_obs_ = y_obs_[..., [response]]
            self.n_response = 1

            _, posterior_samples = BaseModel.run_inference(
                self, subject_ind_, group_ind_, y_obs_, **kwargs
            )
            dest = os.path.join(temp_dir, destination_path)
            with open(dest, "wb") as f:
                pickle.dump((posterior_samples,), f)

        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=False)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel(
                delayed(body_run_inference)(
                    combination,
                    response,
                    os.path.join(temp_dir, f"{'_'.join(map(str, combination))}_{response}.pkl")
                )
                for combination in combinations
                for response in range(n_response)
            )

        self.n_response = n_response
        n_subjects = np.max(subject_ind) + 1
        n_groups = np.max(group_ind) + 1
        n_features = [n_subjects, n_groups]

        posterior_samples = None
        for combination in combinations:
            for response in range(n_response):
                src = os.path.join(
                    temp_dir,
                    f"{'_'.join(map(str, combination))}_{response}.pkl"
                )
                with open(src, "rb") as f:
                    samples, = pickle.load(f)

                if posterior_samples is None:
                    named_param = list(samples.keys())[0]
                    num_samples = samples[named_param].shape[0]

                    posterior_samples = {
                        u: np.full(
                            (num_samples, *n_features, n_response), np.nan
                        )
                        for u in samples.keys()
                        if np.array([dim in [num_samples, 1] for dim in samples[u].shape]).all()
                    }

                for named_param in posterior_samples.keys():
                    posterior_samples[named_param][:, *combination, response] = samples[named_param].reshape(-1,)

        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return None, posterior_samples


class NHB(NonHierarchicalBaseModel):
    NAME = "non_hier"

    def __init__(self):
        super(NHB, self).__init__()
        self.n_jobs = -1

    def _model(self, subject_ind, group_ind, y_obs=None):
        n_subjects = np.max(subject_ind) + 1
        n_groups = np.max(group_ind) + 1

        n_tests = self.n_tests
        n_data = subject_ind.shape[0]

        with numpyro.plate('n_tests', n_tests):
            with numpyro.plate("n_groups", n_groups):
                with numpyro.plate("n_subjects", n_subjects):
                    scale = numpyro.sample('scale', dist.HalfNormal(5))
                    # a_loc_loc_scale = numpyro.sample('a_loc_loc_scale', dist.HalfNormal(50.))
                    # a_loc_scale = numpyro.sample('a_loc_scale', dist.HalfNormal(50.))
                    # a_scale = numpyro.sample('a_scale', dist.HalfNormal(50.))

                    # a_loc_loc = numpyro.sample('a_loc_loc', dist.Normal(50., a_loc_loc_scale))
                    # a_loc = numpyro.sample('a_loc', dist.Normal(a_loc_loc, a_loc_scale))

                    a_loc = numpyro.sample('a_loc', dist.Normal(50., 50.))
                    a_scale = numpyro.sample('a_scale', dist.HalfNormal(50.))

                    a = numpyro.sample('a', dist.Normal(a_loc, a_scale))

        with numpyro.plate('n_tests', n_tests):
            with numpyro.plate('n_data', n_data):
                numpyro.sample('obs', dist.Normal(a[subject_ind, group_ind], scale), obs=y_obs)


class HB2(BaseModel):
    NAME = "HB2"

    def __init__(self):
        super(HB2, self).__init__()

    # def _model(self, subject_ind, group_ind, y_obs=None):
    #     n_subjects = np.max(subject_ind) + 1
    #     n_groups = np.max(group_ind) + 1

    #     n_tests = self.n_tests
    #     n_data = subject_ind.shape[0]

    #     a_scale = numpyro.sample('a_scale', dist.HalfNormal(100.))
    #     scale = numpyro.sample('scale', dist.HalfNormal(5))

    #     with numpyro.plate('n_tests', n_tests):
    #         with numpyro.plate("n_groups", n_groups):
    #             a_loc = numpyro.sample('a_loc', dist.Normal(50., 100.))

    #             with numpyro.plate("n_subjects", n_subjects):
    #                 a = numpyro.sample('a', dist.Normal(a_loc, a_scale))

    #     with numpyro.plate('n_tests', n_tests):
    #         with numpyro.plate('n_data', n_data):
    #             numpyro.sample('obs', dist.Normal(a[subject_ind, group_ind], scale), obs=y_obs)

    def _model(self, subject_ind, group_ind, y_obs=None):
        n_subjects = np.max(subject_ind) + 1
        n_groups = np.max(group_ind) + 1

        n_tests = self.n_tests
        n_data = subject_ind.shape[0]

        a_scale = numpyro.sample('a_scale', dist.HalfNormal(50.))
        scale = numpyro.sample('scale', dist.HalfNormal(5))

        n_fixed = 1
        n_delta = 1

        with numpyro.plate('n_tests', n_tests):
            with numpyro.plate("n_fixed", n_fixed):
                a_loc_fixed = numpyro.sample(
                    "a_loc_fixed", dist.Normal(50., 50.)
                )

        with numpyro.plate('n_tests', n_tests):
            with numpyro.plate("n_delta", n_delta):
                a_loc_delta = numpyro.sample(
                    "a_loc_delta", dist.Normal(0., 50.)
                )
                a_loc_random = numpyro.deterministic(
                    "a_loc_random", a_loc_fixed + a_loc_delta
                )

        with numpyro.plate('n_tests', n_tests):
            with numpyro.plate("n_groups", n_groups):
                a_loc = numpyro.deterministic(
                    "a_loc", jnp.concatenate([a_loc_fixed, a_loc_random], axis=0)
                )

                with numpyro.plate("n_subjects", n_subjects):
                    a = numpyro.sample('a', dist.Normal(a_loc, a_scale))

        with numpyro.plate('n_tests', n_tests):
            with numpyro.plate('n_data', n_data):
                numpyro.sample('obs', dist.Normal(a[subject_ind, group_ind], scale), obs=y_obs)
