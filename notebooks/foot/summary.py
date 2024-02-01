import os
import glob
import pickle
import logging

import arviz as az
from scipy.io import loadmat
import pandas as pd
from numpyro.diagnostics import summary
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import RectifiedLogistic
from utils import read_data
from core import DATA_DIR, BUILD_DIR
from utils import MUSCLES

logger = logging.getLogger(__name__)

INFERENCE_FILE = "inference.pkl"
NETCODE_FILE = "numpyro_data.nc"


def main(visit, participant):
    dir = os.path.join(BUILD_DIR, visit, participant)
    src = os.path.join(dir, INFERENCE_FILE)
    with open(src, "rb") as f:
        model, mcmc, _ = pickle.load(f)

    # Setup logging
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__),
        # level=logging.DEBUG
    )

    src = os.path.join(model.build_dir, NETCODE_FILE)
    numpyro_data = az.from_netcdf(src)

    # Read data
    subdir = os.path.join(DATA_DIR, visit, participant)
    df, mat = read_data(subdir)
    df, encoder_dict = model.load(df)

    # Make summary dataframe
    var_names = [site.a, site.b, site.v, site.L, site.ell, site.H]
    summary_df = az.summary(
        numpyro_data,
        hdi_prob=.95,
        var_names=var_names
    )
    summary_df["parameter"] = summary_df.index
    summary_df = summary_df.reset_index(drop=True).copy()
    # keep_columns = ['parameter', 'mean', 'sd', 'hdi_2.5%', 'hdi_97.5%', 'r_hat']
    keep_columns = ['parameter', 'mean', 'hdi_2.5%', 'hdi_97.5%', 'r_hat']
    summary_df = summary_df[keep_columns].copy()

    target_muscle = df["target_muscle"].unique()[0]
    side = target_muscle[0]
    summary_df["muscle"] = (
        summary_df["parameter"]
        .apply(lambda x: int(x[-2]))
        .apply(lambda x: model.response[x])
        .apply(lambda x: x.split("_"))
        .apply(lambda x: x[0].lower() + "_" + side + x[1])
    )
    summary_df["target_muscle"] = target_muscle

    # Read foot-estimate
    src = os.path.join(model.build_dir, "foot-estimate.csv")
    foot_df = pd.read_csv(src)
    summary_df = pd.concat([summary_df, foot_df], axis=0).reset_index(drop=True).copy()

    # summary_df = summary_df.sort_values(by=["parameter"]).reset_index(drop=True).copy()

    # Save summary data frame
    pattern = os.path.join(subdir, "*REC_table.csv")
    src = glob.glob(pattern)[0]
    base_name = os.path.basename(src).split("-")[0]
    dest = os.path.join(model.build_dir, f"{base_name}_summary.csv")
    summary_df.to_csv(dest)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    subset = [
        ("visit1", "SCS08"),
        ("visit2", "SCA07")
    ]

    # Run a single job
    visit, participant = subset[1]
    main(visit, participant)

    # # Run multiple jobs
    # n_jobs = -1
    # with Parallel(n_jobs=n_jobs) as parallel:
    #     parallel(
    #         delayed(main)(
    #             visit, participant
    #         ) \
    #         for visit, participant in subset
    #     )
