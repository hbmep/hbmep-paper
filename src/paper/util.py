import os
import pickle
import logging

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import arviz as az
from hbmep.util import timing, site

logger = logging.getLogger(__name__)


def get_subname(model):
    return (
        f'{model.mcmc_params["num_warmup"]}w'
        + f'_{model.mcmc_params["num_samples"]}s'
        + f'_{model.mcmc_params["num_chains"]}c'
        + f'_{model.mcmc_params["thinning"]}t'
        + f'_{model.nuts_params["max_tree_depth"][0]}d'
        + f'_{model.nuts_params["target_accept_prob"] * 100:.0f}a'
        + f'_{"t" if model.use_mixture else "f"}m'
    )


def clear_axes(axes):
    for ax in axes.reshape(-1,): ax.clear()
    return


def make_pdf(figs, output_path):
    print("Making pdf...")
    with PdfPages(output_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight') 
    print(f"Saved to {output_path}")
    return


def run(data, model, encoder=None, **kw):
    # Run
    if encoder is None:
        df, encoder = model.load(df=data)
    else:
        df = data.copy()
    logger.info(f"df.shape {df.shape}")
    # model.plot(df, encoder=encoder)
    # return
    mcmc, posterior = model.run(df=df, **kw)

    # Save
    output_path = os.path.join(model.build_dir, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(model.build_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(model.build_dir, "model_dict.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model.__dict__,), f)
    logger.info(f"Saved to {output_path}")

    if mcmc is not None:
        output_path = os.path.join(model.build_dir, "mcmc.pkl")
        with open(output_path, "wb") as f:
            pickle.dump((mcmc,), f)
        logger.info(f"Saved to {output_path}")

    predict(df, encoder, posterior, model, mcmc)
    return


def predict(df, encoder, posterior, model, mcmc):
    # Model evaluation
    logger.info("Evaluating model ...")
    inference_data = az.from_numpyro(mcmc)
    logger.info("LOO ...")
    score = az.loo(inference_data)
    logger.info(score)
    logger.info("WAIC ...")
    score = az.waic(inference_data)
    logger.info(score)

    # Predictions
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] *= 0
    prediction_df = model.make_prediction_dataset(df=df)
    predictive = model.predict(prediction_df, posterior=posterior)
    model.plot_curves(
        df=df,
        encoder=encoder,
        prediction_df=prediction_df,
        predictive=predictive,
        posterior=posterior,
    )

    if site.outlier_prob in posterior.keys():
        posterior.pop(site.outlier_prob)
    summary_df = model.summary(posterior)
    logger.info(f"Summary:\n{summary_df.to_string()}")
    dest = os.path.join(model.build_dir, "summary.csv")
    summary_df.to_csv(dest)
    logger.info(f"Saved summary to {dest}")
    logger.info(f"Finished running {model.name}")
    try:
        divergences = mcmc.get_extra_fields()["diverging"].sum().item()
        logger.info(f"No. of divergences {divergences}")
        num_steps = mcmc.get_extra_fields()["num_steps"]
        tree_depth = np.floor(np.log2(num_steps)).astype(int)
        logger.info(f"Tree depth statistics:")
        logger.info(f"Min: {tree_depth.min()}")
        logger.info(f"Max: {tree_depth.max()}")
        logger.info(f"Mean: {tree_depth.mean()}")
    except: pass
    logger.info(f"Saved results to {model.build_dir}")
    return
