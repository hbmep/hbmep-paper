import os
import shutil
import pickle
from collections import defaultdict

import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

from hbmep.model.utils import abstractvariables
from hbmep.utils import timing


class BaseModel():
    NAME = "base_model"

    def __init__(self):
        self.rng_key = random.PRNGKey(0)
        self.build_dir = None
        self.mcmc_params = {
            "num_chains": 4,
            "num_warmup": 1000,
            "num_samples": 1000,
            "thinning": 1,
        }

    def _model(self, subject_ind, y_obs=None):
        raise NotImplementedError

    @timing
    def run(
        self,
        subject_ind,
        y_obs: np.ndarray,
        **kwargs
    ) -> tuple[MCMC, dict]:
        # Set up sampler
        sampler = NUTS(self._model, **kwargs)
        mcmc = MCMC(sampler, **self.mcmc_params)

        # Run MCMC inference
        mcmc.run(self.rng_key, subject_ind, y_obs=y_obs)
        posterior_samples = mcmc.get_samples()
        posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
        return mcmc, posterior_samples


class HB(BaseModel):
    NAME = "HB"

    def __init__(self):
        super().__init__()

    def _model(self, subject_ind, y_obs=None):
        n_subjects = np.max(subject_ind) + 1
        n_tests = y_obs.shape[-1]
        n_data = y_obs.shape[0]

        a_scale = numpyro.sample("a_delta_scale", dist.HalfNormal(50.))
        scale = numpyro.sample('scale', dist.HalfNormal(50.))

        a_loc_scale = numpyro.sample("a_loc_scale", dist.HalfNormal(50.))

        with numpyro.plate("n_tests", n_tests):
            a_low_raw = numpyro.sample(
                "a_low_raw", dist.Normal(5., 1.)
            )
            a_loc = numpyro.deterministic(
                "a_loc", -5 + a_low_raw * a_loc_scale
            )
            # a_loc = numpyro.sample(
            #     "a_loc", dist.Normal(0., a_loc_scale)
            # )

            with numpyro.plate("n_subjects", n_subjects):
                a = numpyro.sample(
                    "a", dist.Normal(a_loc, a_scale)
                )

        with numpyro.plate("n_tests", n_tests):
            with numpyro.plate("n_data", n_data):
                numpyro.sample(
                    "obs",
                    dist.Normal(a[subject_ind], scale),
                    obs=y_obs
                )

    # def _model(self, subject_ind, y_obs=None):
    #     n_subjects = np.max(subject_ind) + 1
    #     n_tests = y_obs.shape[-1]
    #     n_data = y_obs.shape[0]

    #     a_scale = numpyro.sample("a_delta_scale", dist.HalfNormal(50.))
    #     scale = numpyro.sample('scale', dist.HalfNormal(50.))

    #     a_loc_scale = numpyro.sample("a_loc_scale", dist.HalfNormal(50.))

    #     with numpyro.plate("n_tests", n_tests):
    #         a_loc = numpyro.sample(
    #             "a_loc", dist.Normal(0., a_loc_scale)
    #         )

    #         with numpyro.plate("n_subjects", n_subjects):
    #             a = numpyro.sample(
    #                 "a", dist.Normal(a_loc, a_scale)
    #             )

    #     with numpyro.plate("n_tests", n_tests):
    #         with numpyro.plate("n_data", n_data):
    #             numpyro.sample(
    #                 "obs",
    #                 dist.Normal(a[subject_ind], scale),
    #                 obs=y_obs
    #             )


@abstractvariables(
    ("n_jobs", "Number of parallel jobs not specified.")
)
class NonBaseModel(BaseModel):
    NAME = "non_base_model"

    def __init__(self):
        super().__init__()

    @timing
    def load(self, df: pd.DataFrame, features):
        encoder_dict = defaultdict(LabelEncoder)
        df[features] = (
            df[features]
            .apply(lambda x: encoder_dict[x.name].fit_transform(x))
        )
        return df, encoder_dict

    @timing
    def run(
        self,
        subject_ind,
        y_obs: np.ndarray,
        **kwargs
    ) -> tuple[MCMC, dict]:
        df = pd.DataFrame(
            {
                "subject_ind": subject_ind,
            }
        )
        response_columns = [f"response_{i}" for i in range(y_obs.shape[-1])]
        n_response = len(response_columns)
        df[response_columns] = y_obs.tolist()

        features = ["subject_ind"]
        combinations = df[features].apply(tuple, axis=1).unique().tolist()
        combinations = sorted(combinations)
        temp_dir = os.path.join(self.build_dir, "optimize_results")


        def body_run(combination, response, destination_path):
            ind = df[features].apply(tuple, axis=1).isin([combination])
            df_ = df[ind].reset_index(drop=True).copy()
            df_, _ = self.load(df=df_, features=features)

            subject_ind_ = df_["subject_ind"].values
            y_obs_ = df_[[response]].values

            mcmc, posterior_samples = BaseModel.run(
                self, subject_ind_, y_obs=y_obs_, **kwargs
            )
            dest = os.path.join(temp_dir, destination_path)
            with open(dest, "wb") as f: pickle.dump((mcmc, posterior_samples,), f)

            ind, df_, _, posterior_samples = None, None, None, None
            del ind, df_, _, posterior_samples
            return


        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=False)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            parallel(
                delayed(body_run)(
                    combination,
                    response,
                    os.path.join(temp_dir, f"{'_'.join(map(str, combination))}_{response}.pkl")
                )
                for combination in combinations
                for response in response_columns
            )

        n_features = (
            df[features].max().astype(int).to_numpy() + 1
        )
        n_features = n_features.tolist()

        posterior_samples = None
        for combination in combinations:
            for response_ind, response in enumerate(response_columns):
                src = os.path.join(
                    temp_dir,
                    f"{'_'.join(map(str, combination))}_{response}.pkl"
                )
                with open(src, "rb") as f: mcmc, samples, = pickle.load(f)

                if posterior_samples is None:
                    num_samples = samples["a"].shape[0]
                    posterior_samples = {
                        u: np.full(
                            (num_samples, *n_features, n_response), np.nan
                        )
                        for u in samples.keys() if u == "a"
                    }

                for named_param in posterior_samples.keys():
                    posterior_samples[named_param][:, *combination, response_ind] = (
                        samples[named_param].reshape(-1,)
                    )

        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        return None, posterior_samples


class NON(NonBaseModel):
    NAME = "NON"

    def __init__(self):
        super().__init__()
        self.n_jobs = -1

    def _model(self, subject_ind, y_obs=None):
        n_subjects = np.max(subject_ind) + 1
        n_tests = y_obs.shape[-1]
        n_data = y_obs.shape[0]

        with numpyro.plate("n_tests", n_tests):
            with numpyro.plate("n_subjects", n_subjects):
                a_loc = numpyro.sample("a_loc", dist.Normal(50., 50.))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(50.))

                a = numpyro.sample(
                    "a", dist.Normal(a_loc, a_scale)
                )

        scale = numpyro.sample('scale', dist.HalfNormal(50.))

        with numpyro.plate("n_tests", n_tests):
            with numpyro.plate("n_data", n_data):
                numpyro.sample(
                    "obs",
                    dist.Normal(a[subject_ind], scale),
                    obs=y_obs
                )
