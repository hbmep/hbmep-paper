import logging
import itertools

import numpy as np
import pandas as pd

from hbmep.config import Config
from hbmep.utils import timing

from hbmep_paper.simulator import HierarchicalBayesianModel

logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self, config: Config):
        self.simulator = HierarchicalBayesianModel(config=config)
        pass

    @timing
    def simulate(
        self,
        n_subject=3,
        n_feature0=2,
        n_repeats=10,
        downsample_rate: int = 1
    ):
        n_features = [n_subject, n_feature0]
        combinations = itertools.product(*[range(i) for i in n_features])
        combinations = list(combinations)
        combinations = sorted(combinations)

        logger.info("Simulating data ...")
        df = pd.DataFrame(combinations, columns=self.simulator.combination_columns)
        x_space = np.arange(0, 360, 4)

        df[self.intensity] = df.apply(lambda _: x_space, axis=1)
        df = df.explode(column=self.simulator.intensity).reset_index(drop=True).copy()
        df[self.simulator.intensity] = df[self.simulator.intensity].astype(float)

        posterior_samples = self.simulator.predict(df=df, num_samples=n_repeats)
        posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}

        # x_space = x_space[1:] ?? Remove 0 from x-space?
        x_space_downsampled = x_space[::downsample_rate]
        ind = df[self.intensity].isin(x_space_downsampled)
        df = df[ind].reset_index(drop=True).copy()

        for var in [site.obs, site.mu, site.beta]:
            posterior_samples[var] = posterior_samples[var][:, ind, :]

        return df, posterior_samples


# @timing
# def run_experiment(
#     config: Config,
#     models: list[Baseline],
#     df: pd.DataFrame,
#     posterior_samples_true: dict
# ):
#     obs_true = posterior_samples_true[site.obs]
#     n_repeats = obs_true.shape[0]
#     n_models = len(models)

#     baseline = Baseline(config=config)
#     df, encoder_dict = baseline.load(df=df)

#     combinations = baseline._make_combinations(df=df, columns=baseline.combination_columns)
#     combinations = [[slice(None)] + list(c) for c in combinations]
#     combinations = [tuple(c[::-1]) for c in combinations]

#     evaluation = []
#     evaluation_path = os.path.join(baseline.build_dir, "evaluation.csv")
#     columns = ["run", "model", "mse", "mae"]

#     for i in range(n_repeats):
#         a_true = np.array(posterior_samples_true[site.a])       # n_repeats x ... x n_muscles
#         a_true = a_true[i, ...]      # ... x n_muscles
#         a_true = np.array([a_true[c] for c in combinations])    # n_combinations x n_muscles
#         a_true = a_true.reshape(-1, )

#         for j, m in enumerate(models):
#             logger.info("\n\n")
#             logger.info(f" Experiment: {i + 1}/{n_repeats}, Model: {j + 1}/{n_models} ({m.LINK}) ".center(20, "="))
#             model = m(config=config)
#             df[model.response] = obs_true[i, ...]
#             mcmc, posterior_samples = model.run_inference(df=df)

#             a = np.array(posterior_samples[site.a])     # n_posterior_samples x ... x n_muscles
#             a = a.mean(axis=0)      # ... x n_muscles
#             a = np.array([a[c] for c in combinations])      # n_combinations x n_muscles
#             a = a.reshape(-1, )

#             # Evaluate
#             mse = mean_squared_error(y_true=a_true, y_pred=a)
#             mae = mean_absolute_error(y_true=a_true, y_pred=a)

#             row = [i + 1, m.LINK, mse, mae]
#             evaluation.append(row)
#             logger.info(
#                 f"Run: {i + 1}/{n_repeats}, Model: {j + 1}/{n_models} ({m.LINK}), MSE: {mse}, MAE: {mae}"
#             )

#             # if not (i + 1) % 10:
#             if True:
#                 evaluation_df = pd.DataFrame(evaluation, columns=columns)
#                 evaluation_df.to_csv(evaluation_path, index=False)

#                 model.build_dir = os.path.join(baseline.build_dir, f"run{i + 1}/{m.LINK}")
#                 model._make_dir(model.build_dir)

#                 model.recruitment_curves_path = os.path.join(model.build_dir, RECRUITMENT_CURVES)
#                 model.mcmc_path = os.path.join(model.build_dir, MCMC_NC)
#                 model.diagnostics_path = os.path.join(model.build_dir, DIAGNOSTICS_CSV)
#                 model.loo_path = os.path.join(model.build_dir, LOO_CSV)
#                 model.waic_path = os.path.join(model.build_dir, WAIC_CSV)

#                 logger.info(f"Saving artefacts to {model.build_dir}")
#                 model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)
#                 model.save(mcmc=mcmc)

#     evaluation_df = pd.DataFrame(evaluation, columns=columns)
#     evaluation_df.to_csv(evaluation_path, index=False)
#     return
