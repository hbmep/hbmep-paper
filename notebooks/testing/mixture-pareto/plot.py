import os
import logging
import pickle

import pandas as pd

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import MixtureModel
from core import DATA_PATH

logger = logging.getLogger(__name__)


def main():
    src = "/home/vishu/repos/hbmep-paper/reports/testing/mixture-pareto/mixture_model/inference.pkl"
    with open(src, "rb") as f:
        model, mcmc, posterior_samples = pickle.load(f)

    setup_logging(
        dir=model.build_dir,
        fname="plot",
    )

    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)

    # Render plots
    posterior_samples[site.outlier_prob] = 0 * posterior_samples[site.outlier_prob]
    posterior_samples[site.q] = 0 * posterior_samples[site.q]

    # for u, v in posterior_samples.items():
    #     print(u, v.shape)

    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    return


if __name__ == "__main__":
    main()
