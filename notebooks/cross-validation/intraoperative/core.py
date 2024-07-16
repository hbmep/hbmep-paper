import os
import gc
import pickle
import logging

import arviz as az
import pandas as pd
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import (
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    RectifiedLinear,
    MixtureModel
)
from constants import (
    DATA_PATH,
    TOML_PATH,
    INFERENCE_FILE,
    BUILD_DIR
)

logger = logging.getLogger(__name__)


def main():
    # Load data
    data = pd.read_csv(DATA_PATH)


    # Define experiment
    def run_inference(M):
        # Build model
        config = Config(toml_path=TOML_PATH)
        config.BUILD_DIR = os.path.join(
            BUILD_DIR,
            M.NAME
        )
        model = M(config=config)

        # Set up logging
        model._make_dir(model.build_dir)
        setup_logging(
            dir=model.build_dir,
            fname=os.path.basename(__file__)
        )
        for u, v in model.mcmc_params.items():
            logger.info(f"{u} = {v}")

        # Run inference
        df = data.copy()
        ind = ~df[model.response].isna().values.any(axis=-1)
        df = df[ind].reset_index(drop=True).copy()
        df, encoder_dict = model.load(df=df)
        logger.info(f"Running inference for {model.NAME} with {df.shape[0]} samples ...")
        mcmc, posterior_samples = model.run_inference(df=df)

        # Turn off mixture distribution
        if site.outlier_prob in posterior_samples:
            outlier_prob = posterior_samples[site.outlier_prob]
            posterior_samples[site.outlier_prob] = 0 * posterior_samples[site.outlier_prob]

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

        if site.outlier_prob in posterior_samples:
            posterior_samples[site.outlier_prob] = outlier_prob

        # Save posterior
        dest = os.path.join(model.build_dir, INFERENCE_FILE)
        with open(dest, "wb") as f:
            pickle.dump((model, mcmc, posterior_samples,), f)
        logger.info(f"Saved inference data to {dest}")

        # Model evaluation
        inference_data = az.from_numpyro(mcmc)
        logger.info("Evaluating model ...")
        logger.info("LOO ...")
        score = az.loo(inference_data)
        logger.info(score)
        logger.info("WAIC ...")
        score = az.waic(inference_data)
        logger.info(score)
        vars_to_exclude = [site.mu, site.alpha, site.beta, site.obs]
        vars_to_exclude = ["~" + var for var in vars_to_exclude]
        logger.info(az.summary(inference_data, var_names=vars_to_exclude).to_string())

        config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
        model, posterior_samples, posterior_predictive = None, None, None
        inference_data, score = None, None
        del config, df, prediction_df, encoder_dict, _
        del model, posterior_samples, posterior_predictive
        del inference_data, score
        gc.collect()


    # # Run multiple models in parallel
    # n_jobs = -1
    # models = [
    #     RectifiedLogistic,
    #     Logistic5,
    #     Logistic4,
    #     RectifiedLinear,
    #     MixtureModel
    # ]

    # with Parallel(n_jobs=n_jobs) as parallel:
    #     parallel(
    #         delayed(run_inference)(M) for M in models
    #     )

    # Run single model
    M = MixtureModel
    run_inference(M)

    return


if __name__ == "__main__":
    main()
