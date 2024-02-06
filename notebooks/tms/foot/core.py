import os
import logging

import numpy as np
import pandas as pd

from hbmep.config import Config

from hbmep_paper.utils import setup_logging
from models import RectifiedLogistic
from utils import save

logger = logging.getLogger(__name__)

DATA_DIR = "/home/vishu/data/hbmep-processed/foot"
TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/tms/config.toml"
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/foot/"

FEATURES = [["participant", "visit"]]
RESPONSE = ['PKPK_ADM', 'PKPK_APB', 'PKPK_Biceps', 'PKPK_ECR', 'PKPK_FCR', 'PKPK_Triceps']
MEP_SIZE_WINDOW = [0.005, 0.005 + .085]


def main(Model):
    config = Config(TOML_PATH)
    config.BUILD_DIR = os.path.join(BUILD_DIR, Model.NAME)
    config.FEATURES = FEATURES
    config.RESPONSE = RESPONSE
    config.MEP_SIZE_TIME_RANGE = MEP_SIZE_WINDOW
    model = RectifiedLogistic(config)

    # Setup logging
    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__),
        # level=logging.DEBUG
    )

    src = os.path.join(DATA_DIR, "data.csv")
    df = pd.read_csv(src)
    src = os.path.join(DATA_DIR, "mat.npy")
    mat = np.load(src)

    # Remove NaNs
    ind =  df[model.response].isna().any(axis=1)
    df = df[~ind].reset_index(drop=True).copy()
    mat = mat[~ind, ...]

    df, encoder_dict = model.load(df)
    model.plot(df=df, encoder_dict=encoder_dict, mep_matrix=mat)

    mcmc, posterior_samples = model.run_inference(df=df)

    # Recruitement curves
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(
        df=prediction_df,
        posterior_samples=posterior_samples
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
    save(model, mcmc, posterior_samples)
    return


if __name__ == "__main__":
    Model = RectifiedLogistic
    main(Model)
