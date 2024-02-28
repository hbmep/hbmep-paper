import os
import gc
import pickle
import logging

import arviz as az
import pandas as pd
import numpy as np
from numpyro.infer.util import log_likelihood
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import (
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    RectifiedLinear
)
from constants import (
    DATA_PATH,
    TOML_PATH,
    INFERENCE_FILE,
    SVI_LOO_DIR
)

logger = logging.getLogger(__name__)

BUILD_DIR = SVI_LOO_DIR
CV_PREFIX = "cv"


@timing
def main():
    # Load data
    data = pd.read_csv(DATA_PATH)

    data = data.reset_index(drop=True).copy()
    data = data.head(n=40).resrt_index(drop=True).copy()


    def run_svi

    # Define experiment
    def run_inference(M, i):
        # Build model
        config = Config(toml_path=TOML_PATH)
        config.BUILD_DIR = os.path.join(
            BUILD_DIR,
            M.NAME,
            CV_PREFIX + str(i),
        )
        model = M(config=config)

        # Set up logging
        model._make_dir(model.build_dir)
        setup_logging(
            dir=model.build_dir,
            fname="logs"
        )

        # Split data
        ind = data.index.isin([i])
        train = data[~ind].reset_index(drop=True).copy()
        test = data[ind].reset_index(drop=True).copy()

        # Run inference
        df, encoder_dict = model.load(df=train)
        _, posterior_samples = run_svi(df=df)

        # # Predictions and recruitment curves
        # prediction_df = model.make_prediction_dataset(df=df)
        # posterior_predictive = model.predict(
        #     df=prediction_df, posterior_samples=posterior_samples
        # )
        # model.render_recruitment_curves(
        #     df=df,
        #     encoder_dict=encoder_dict,
        #     posterior_samples=posterior_samples,
        #     prediction_df=prediction_df,
        #     posterior_predictive=posterior_predictive
        # )
        # model.render_predictive_check(
        #     df=df,
        #     encoder_dict=encoder_dict,
        #     prediction_df=prediction_df,
        #     posterior_predictive=posterior_predictive
        # )

        # # Save posterior
        # dest = os.path.join(model.build_dir, INFERENCE_FILE)
        # with open(dest, "wb") as f:
        #     pickle.dump((model, mcmc, posterior_samples,), f)
        # logger.info(f"Saved inference data to {dest}")

        # Cross-validation
        test, encoder_dict_ = model.preprocess(df=test, encoder_dict=encoder_dict)
        test_log_likelihood = log_likelihood(
            model._model,
            posterior_samples,
            *model._get_regressors(df=test),
            *model._get_response(df=test)
        )
        test_log_likelihood = test_log_likelihood[site.obs]
        test_log_likelihood = np.array(test_log_likelihood)
        dest = os.path.join(model.build_dir, "test_log_likelihood.npy")
        np.save(dest, test_log_likelihood)
        logger.info(f"Saved test log likelihood to {dest}")

        # # Model evaluation
        # inference_data = az.from_numpyro(mcmc)
        # logger.info("Evaluating model ...")
        # logger.info("LOO ...")
        # score = az.loo(inference_data)
        # logger.info(score)
        # logger.info("WAIC ...")
        # score = az.waic(inference_data)
        # logger.info(score)
        # vars_to_exclude = [site.mu, site.alpha, site.beta, site.obs]
        # vars_to_exclude = ["~" + var for var in vars_to_exclude]
        # logger.info(az.summary(inference_data, var_names=vars_to_exclude).to_string())

        config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
        model, posterior_samples, posterior_predictive = None, None, None
        # inference_data, score = None, None
        del config, df, prediction_df, encoder_dict, _
        del model, posterior_samples, posterior_predictive
        # del inference_data, score
        gc.collect()


    # # Run multiple models in parallel
    # n_jobs = -1
    # models = [
    #     RectifiedLogistic
    # ]

    # with Parallel(n_jobs=n_jobs) as parallel:
    #     parallel(
    #         delayed(run_inference)(M, i)
    #         for M in models
    #         for i in range(data.shape[0])
    #     )

    # Run single model
    M = RectifiedLogistic
    run_inference(M, 2)

    return


if __name__ == "__main__":
    main()
