import os
import pickle
import logging

import arviz as az

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import (
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    NelderMeadOptimization,
)
from constants import (
    TOML_PATH,
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
        df, encoder_dict, _, _, _, _ = pickle.load(f)

    match M.NAME:
        case HierarchicalBayesianModel.NAME:
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
            mcmc, posterior_samples_ = model.run_inference(df=df)

            # Predictions and recruitment curves
            posterior_samples = posterior_samples_.copy()
            posterior_samples[site.outlier_prob] = 0 * posterior_samples[site.outlier_prob]
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

            # Save posterior
            dest = os.path.join(model.build_dir, INFERENCE_FILE)
            with open(dest, "wb") as f:
                pickle.dump((model, mcmc, posterior_samples_,), f)
            logger.info(f"Saved inference data to {dest}")

            # Model evaluation
            inference_data = az.from_numpyro(mcmc)
            logger.info("Evaluating model ...")
            logger.info("LOO ...")
            score = az.loo(inference_data, var_name=site.obs)
            logger.info(score)
            logger.info("WAIC ...")
            score = az.waic(inference_data, var_name=site.obs)
            logger.info(score)
            logger.info(model.print_summary(posterior_samples))

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
    # M = HierarchicalBayesianModel
    # M = NonHierarchicalBayesianModel
    # M = MaximumLikelihoodModel
    M = NelderMeadOptimization
    main(M=M)
