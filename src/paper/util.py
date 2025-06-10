import os
import pickle
import logging

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
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


def save_model(model, df, encoder, posterior, mcmc=None):
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
    save_model(model, df, encoder, posterior, mcmc)
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


def load_model(
    model_dir,
    inference_file="inf.pkl",
    model_file="model.pkl",
    mcmc_file="mcmc.pkl",
):
    src = os.path.join(model_dir, inference_file)
    with open(src, "rb") as f:
        df, encoder, posterior, = pickle.load(f)

    model = None
    try:
        src = os.path.join(model_dir, model_file)
        with open(src, "rb") as f:
            model, = pickle.load(f)
    except ModuleNotFoundError as e:
        logger.info(e)

    mcmc = None
    try:
        src = os.path.join(model_dir, mcmc_file)
        with open(src, "rb") as f:
            mcmc, = pickle.load(f)
    except FileNotFoundError:
        logger.info(
            f"{mcmc_file} not found. Attempting to read from model_dict.pkl"
        )
    except ValueError as e:
        logger.info("Encountered ValueError, trace is below")
        logger.info(e)
    else:
        logger.info(f"Found {model_file}")

    if mcmc is None:
        try:
            src = os.path.join(model_dir, "model_dict.pkl")
            with open(src, "rb") as f:
                mcmc, _ = pickle.load(f)
        except FileNotFoundError:
            logger.info("model_dict.pkl not found.")
        except ValueError as e:
            logger.info(
                "Encountered ValueError, trace is below."
                + " Possible issue with unpacking."
            )
            logger.info(e)
        else:
            logger.info("Found model_dict.pkl")

    return df, encoder, posterior, model, mcmc


def _adjust_brightness(rgb_in, val):
    hsv = rgb_to_hsv(rgb_in)
    hsv[2] = val  # Adjust the brightness (value component in HSV)
    rgb_out = hsv_to_rgb(hsv)
    return rgb_out


def _get_cmap_muscles_alt():
    vec_muscle = np.array(["Trapezius", "Deltoid", "Biceps", "Triceps", "ECR", "FCR", "APB", "ADM", "TA", "EDB", "AH", "FDI", "auc_target"])
    cmap_mus_dark = np.array([
        _adjust_brightness(np.array([0.6350, 0.0780, 0.1840]), 0.5),  # trapz
        np.array([1, 133, 113]) / 255,  # delt
        np.array([166, 97, 26]) / 255,  # biceps
        np.array([44, 123, 182]) / 255,  # triceps
        np.array([52, 0, 102]) / 255,  # ecr
        _adjust_brightness(np.array([0.5, 0.5, 0.5]), 0.3),  # fcr
        np.array([208, 28, 139]) / 255,  # apb
        np.array([77, 172, 38]) / 255,  # adm
        np.array([215, 25, 28]) / 255,  # ta
        np.array([123, 50, 148]) / 255,  # edb
        _adjust_brightness(np.array([153, 79, 0]) / 256, 0.4),  # ah
        np.array([231, 226, 61]) / 255,  # fdi
        np.array([255, 100, 0]) / 255,  # auc_target
    ])
    cmap_mus_light = np.array([
        _adjust_brightness(np.array([0.6350, 0.0780, 0.1840]), 0.8),  # trapz
        np.array([128, 205, 193]) / 255,  # delt
        np.array([223, 194, 125]) / 255,  # biceps
        np.array([171, 217, 233]) / 255,  # triceps
        _adjust_brightness(np.array([200, 40, 0]) / 255, 0.6),  # ecr
        _adjust_brightness(np.array([0.5, 0.5, 0.5]), 0.6),  # fcr
        np.array([241, 182, 218]) / 255,  # apb
        np.array([184, 225, 134]) / 255,  # adm
        np.array([253, 174, 97]) / 255,  # ta
        np.array([194, 165, 207]) / 255,  # edb
        _adjust_brightness(np.array([153, 79, 0]) / 256, 0.6),  # ah
        _adjust_brightness(np.array([23, 54, 124]) / 256, 0.6),  # fdi
        np.array([255, 100, 0]) / 255,  # auc_target
    ])
    # Create a DataFrame to hold muscle names and corresponding colors
    T_color = pd.DataFrame({
        'muscle': vec_muscle,
        'cmap_mus_light': [tuple(c) for c in cmap_mus_light],
        'cmap_mus_dark': [tuple(c) for c in cmap_mus_dark],
    })
    # Convert RGB to hex
    T_color['cmap_mus_light_hex'] = T_color['cmap_mus_light'].apply(
        lambda x: '#%02x%02x%02x' % tuple([int(255 * v) for v in x]))
    T_color['cmap_mus_dark_hex'] = T_color['cmap_mus_dark'].apply(
        lambda x: '#%02x%02x%02x' % tuple([int(255 * v) for v in x]))
    return cmap_mus_dark, cmap_mus_light, vec_muscle, T_color


def get_response_colors(response: list[str]):
    cmap_mus_dark, cmap_mus_light, vec_muscle, T_color = _get_cmap_muscles_alt()
    cmap_dict = dict(zip(vec_muscle, cmap_mus_dark))
    colors = []
    for response in response: colors.append(cmap_dict[response[5:]])
    return colors
