import os
import logging

from scipy.io import loadmat
import numpy as np
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import RectifiedLogistic
from utils import read_data, save

logger = logging.getLogger(__name__)

DATA_DIR = "/home/vishu/data/raw/foot"
TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/tms/config.toml"
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/foot/"

FEATURES = ["participant"]
RESPONSE = ['PKPK_ADM', 'PKPK_APB', 'PKPK_Biceps', 'PKPK_ECR', 'PKPK_FCR', 'PKPK_Triceps']
MEP_SIZE_WINDOW = [0.005, 0.005 + .085]


def main(visit, participant):
    config = Config(TOML_PATH)
    config.BUILD_DIR = os.path.join(BUILD_DIR, visit, participant)
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

    subdir = os.path.join(DATA_DIR, visit, participant)
    df, mat = read_data(subdir)

    df, encoder_dict = model.load(df)
    dest = os.path.join(model.build_dir, "mat.npy")
    np.save(dest, mat)
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
    subset = [
        ("visit1", "SCS08"),
        ("visit2", "SCA07")
    ]

    # # Run a single job
    # visit, participant = subset[1]
    # main(visit, participant)

    # Run multiple jobs
    n_jobs = -1
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(main)(
                visit, participant
            ) \
            for visit, participant in subset
        )
