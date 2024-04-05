import os
import pickle
import logging

import arviz as az
import matplotlib.pyplot as plt

from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import (
    RectifiedLogistic
)
from constants import (
    DATA_PATH,
    INFERENCE_FILE,
    BUILD_DIR
)

logger = logging.getLogger(__name__)


@timing
def main():
    M = RectifiedLogistic
    src = os.path.join(BUILD_DIR, M.NAME, INFERENCE_FILE)
    with open(src, "rb") as f:
        model, mcmc, posterior_samples = pickle.load(f)

    for u, v in posterior_samples.items():
        logger.info(f"Posterior samples: {u} {v.shape}")

    from hbmep.nn import functional as F
    import pandas as pd
    import numpy as np

    df = pd.read_csv(DATA_PATH)
    df, _ = model.load(df=df)

    combinations = model._get_combinations(df=df, columns=model.features)
    logger.info(f"df: {df.shape}")
    logger.info(f"combinations: {len(combinations)}")

    from hbmep.model.utils import Site as site
    named_params = [site.a, site.b, site.L, site.ell, site.H]

    saturation = posterior_samples[site.L] + posterior_samples[site.H]
    saturation = saturation.mean(axis=0)
    logger.info(saturation.shape)

    response_at_100 = F.rectified_logistic(
        100, *[posterior_samples[p] for p in named_params]
    )
    response_at_100 = response_at_100.mean(axis=0)
    logger.info(response_at_100.shape)

    proportions = None
    for c in combinations:
        ind = df[model.features].apply(tuple, axis=1).isin([c])
        temp_df = df[ind].reset_index(drop=True).copy()

        max_intensity = temp_df[model.intensity].max()

        response_at_max_intensity = F.rectified_logistic(
            max_intensity, *[posterior_samples[p] for p in named_params]
        )
        numerator = response_at_max_intensity.mean(axis=0)[*c, :]
        denominator = saturation[*c, :]
        observed_saturation_proportion = numerator / denominator

        if proportions is None:
            proportions = observed_saturation_proportion[..., None]
        else:
            proportions = np.concatenate(
                [proportions, observed_saturation_proportion[..., None]],
                axis=-1
            )

    logger.info(proportions.shape)

    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = model._get_colors(model.n_response)

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        constrained_layout=True,
        squeeze=False
    )

    for i, response in enumerate(model.response):
        ax = axes[i // ncols, i % ncols]
        sns.histplot(proportions[i, ...], ax=ax, color=colors[i])
        ax.set_title(response)

    ax = axes[-1, -2]
    sns.histplot(proportions.reshape(-1,), ax=ax)
    ax.set_title("All responses")

    ax = axes[-1, -1]
    muscle_ind = 1
    sns.histplot(response_at_100[:, muscle_ind] / saturation[:, muscle_ind], ax=ax)
    ax.set_title(f"{model.response[muscle_ind]} at 100")

    dest = os.path.join(BUILD_DIR, "saturation_proportions.png")
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    setup_logging(BUILD_DIR, os.path.basename(__file__))
    main()
