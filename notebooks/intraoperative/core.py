import os
import pickle
import logging

import numpy as np
import pandas as pd

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel
from bootstrap__models import (
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    NelderMeadOptimization
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
        df, encoder_dict, _, _, _ = pickle.load(f)

    match M.NAME:
        case HierarchicalBayesianModel.NAME:
            # Build model
            config = Config(toml_path=TOML_PATH)
            # config.BUILD_DIR = os.path.join(
            #     BUILD_DIR,
            #     M.NAME
            # )
            model = M(config=config)
            model.build_dir = os.path.join(BUILD_DIR, model.NAME)

            # Load data
            df = pd.read_csv(DATA_PATH)
            ind = ~df[model.response].isna().values.any(axis=-1)
            df = df[ind].reset_index(drop=True).copy()
            df, encoder_dict = model.load(df=df)
            logger.info(f"DF shape: {df.shape}")

            # Logging
            os.makedirs(model.build_dir, exist_ok=True)
            setup_logging(
                dir=model.build_dir,
                fname=os.path.basename(__file__)
            )
            for u, v in model.mcmc_params.items(): logger.info(f"{u} = {v}")
            for u, v in model.run_kwargs.items(): logger.info(f"{u} = {v}")

            # Run inference
            logger.info(f"{df.shape[0]} observations")
            mcmc, posterior_samples = model.run(
                df=df, **model.run_kwargs
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
            logger.info(f"DF shape: {df.shape}")
            _, posterior_samples = model.run(df=df, max_tree_depth=(20, 20))

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

            # Compute error and save results
            a_pred = posterior_samples[site.a]
            np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

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
            logger.info(f"DF shape: {df.shape}")
            params = model.run(df=df)

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
            src = os.path.join(model.build_dir, "a_pred.npy")
            np.save(src, params[site.a])

        case _:
            raise ValueError(f"Unknown model: {M.NAME}")

    return


if __name__ == "__main__":
    M = HierarchicalBayesianModel
    # M = MaximumLikelihoodModel
    # M = NelderMeadOptimization
    main(M=M)
