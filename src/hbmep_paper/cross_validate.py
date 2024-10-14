import os
import pickle
import logging

import arviz as az

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging

logger = logging.getLogger(__name__)


@timing
def run(
    data,
    M,
    TOML_PATH,
    BUILD_DIR,
    INFERENCE_FILE,
    **kwargs
):
    # Build model
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = os.path.join(
        BUILD_DIR,
        M.NAME
    )
    model = M(config=config)

    # Set up logging
    os.makedirs(model.build_dir, exist_ok=True)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )
    for u, v in model.mcmc_params.items(): logger.info(f"{u} = {v}")

    # Run inference
    df, encoder_dict = model.load(df=data)
    logger.info(f"{df.shape[0]} observations")
    mcmc, posterior_samples = model.run(df=df, **kwargs)

    # Save posterior
    dest = os.path.join(model.build_dir, INFERENCE_FILE)
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples,), f)
    logger.info(f"Saved inference data to {dest}")

    # Model evaluation
    inference_data = az.from_numpyro(mcmc)
    logger.info("Evaluating model ...")
    logger.info("LOO ...")
    score = az.loo(inference_data); logger.info(score);
    logger.info("WAIC ...")
    score = az.waic(inference_data); logger.info(score);

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
    return
