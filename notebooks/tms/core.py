import os
import pickle
import logging

import arviz as az
import pandas as pd

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import (
    HierarchicalBayesianModel,
    Logistic5
    # NonHierarchicalBayesianModel,
    # MaximumLikelihoodModel,
    # NelderMeadOptimization
)
from constants import (
    TOML_PATH,
    DATA_PATH,
    INFERENCE_FILE,
    BUILD_DIR,
    BOOTSTRAP_DIR,
    BOOTSTRAP_FILE
)

logger = logging.getLogger(__name__)


def main(M):
    # Load data
    src = os.path.join(BOOTSTRAP_DIR, BOOTSTRAP_FILE)
    with open(src, "rb") as f:
        df, encoder_dict, _, _, _, _, _, _ = pickle.load(f)

    match M.NAME:
        case HierarchicalBayesianModel.NAME | Logistic5.NAME:
            config = Config(toml_path=TOML_PATH)
            config.BUILD_DIR = os.path.join(
                BUILD_DIR,
                M.NAME
            )
            model = M(config=config)
            model.mcmc_params = {
                "num_samples": 4000,
                "num_warmup": 4000,
                "num_chains": 4,
                "thinning": 4,
            }
            # model.mcmc_params = {
            #     "num_samples": 20000,
            #     "num_warmup": 20000,
            #     "num_chains": 4,
            #     "thinning": 20,
            # }

            # Logging
            os.makedirs(model.build_dir, exist_ok=True)
            setup_logging(
                dir=model.build_dir,
                fname=os.path.basename(__file__)
            )
            for u, v in model.mcmc_params.items():
                logger.info(f"{u} = {v}")

            # Load data (including outliers)
            df = pd.read_csv(DATA_PATH)
            df[model.features[1]] = (
                df[model.features[1]]
                .replace({
                    "Uninjured": "NOT_SCI"
                })
            )
            df, encoder_dict = model.load(df=df)

            # Run inference
            logger.info(f"{df.shape[0]} observations")
            mcmc, posterior_samples = model.run(
                df=df,
                target_accept_prob=.95,
                max_tree_depth=(15, 15),
                extra_fields=[
                    "num_steps",
                    "accept_prob",
                ]
            )

            # Save posterior
            dest = os.path.join(model.build_dir, INFERENCE_FILE)
            with open(dest, "wb") as f:
                pickle.dump((model, mcmc, posterior_samples,), f)
            logger.info(f"Saved inference data to {dest}")

            # Model evaluation
            summary_df = model.summary(posterior_samples)
            logger.info(f"Summary:\n{summary_df.to_string()}")
            dest = os.path.join(model.build_dir, "summary.csv")
            summary_df.to_csv(dest)
            logger.info(f"Saved summary to {dest}")

            # Turn off mixture distribution
            if site.outlier_prob in posterior_samples:
                posterior_samples[site.outlier_prob] = (
                    0 * posterior_samples[site.outlier_prob]
                )
                model.sample_sites = [
                    s for s in model.sample_sites
                    if s != site.outlier_prob
                ]

            # Predictions and recruitment curves
            prediction_df = model.make_prediction_dataset(df=df)
            posterior_predictive = model.predict(
                df=prediction_df, posterior_samples=posterior_samples
            )
            model.render_recruitment_curves(
                df=df,
                encoder_dict=encoder_dict,
                posterior_samples=posterior_samples,
                prediction_df=prediction_df,
                posterior_predictive=posterior_predictive
            )
            model.render_predictive_check(
                df=df,
                encoder_dict=encoder_dict,
                prediction_df=prediction_df,
                posterior_predictive=posterior_predictive
            )
            logger.info(f"Finished running {model.NAME}")
            logger.info(f"Saved results to {model.build_dir}")

        case NonHierarchicalBayesianModel.NAME | MaximumLikelihoodModel.NAME:
            config = Config(toml_path=TOML_PATH)
            config.BUILD_DIR = os.path.join(
                BUILD_DIR,
                M.NAME
            )
            model = M(config=config)

            # Logging
            os.makedirs(model.build_dir, exist_ok=True)
            setup_logging(
                dir=model.build_dir,
                fname=os.path.basename(__file__)
            )

            # Run inference
            _, posterior_samples = model.run_inference(df=df)

            # Predictions and recruitment curves
            prediction_df = model.make_prediction_dataset(df=df)
            posterior_predictive = model.predict(
                df=prediction_df, posterior_samples=posterior_samples
            )
            model.render_recruitment_curves(
                df=df,
                encoder_dict=encoder_dict,
                posterior_samples=posterior_samples,
                prediction_df=prediction_df,
                posterior_predictive=posterior_predictive
            )
            model.render_predictive_check(
                df=df,
                encoder_dict=encoder_dict,
                prediction_df=prediction_df,
                posterior_predictive=posterior_predictive
            )

            # Save
            src = os.path.join(model.build_dir, INFERENCE_FILE)
            with open(src, "wb") as f:
                pickle.dump((posterior_samples,), f)

        case NelderMeadOptimization.NAME:
            config = Config(toml_path=TOML_PATH)
            config.BUILD_DIR = os.path.join(
                BUILD_DIR,
                M.NAME
            )
            model = M(config=config)

            # Logging
            os.makedirs(model.build_dir, exist_ok=True)
            setup_logging(
                dir=model.build_dir,
                fname=os.path.basename(__file__)
            )

            # Run inference
            params = model.run_inference(df=df)

            # Predictions and recruitment curves
            prediction_df = model.make_prediction_dataset(df=df)
            prediction_df = model.predict(df=prediction_df, params=params)
            model.render_recruitment_curves(
                df=df,
                encoder_dict=encoder_dict,
                params=params,
                prediction_df=prediction_df,
            )

            # Save
            src = os.path.join(model.build_dir, INFERENCE_FILE)
            with open(src, "wb") as f:
                pickle.dump((params,), f)

    return


if __name__ == "__main__":
    M = HierarchicalBayesianModel
    # M = Logistic5
    # M = NonHierarchicalBayesianModel
    # M = MaximumLikelihoodModel
    # M = NelderMeadOptimization
    main(M=M)
