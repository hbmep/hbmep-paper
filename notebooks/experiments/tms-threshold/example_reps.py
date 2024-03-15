import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel

logger = logging.getLogger(__name__)

PPD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/example_reps"
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data"


def main():
    n_reps_space = [8, 4, 2, 1]
    models = {}
    ps = {}
    a = []
    for n_reps in n_reps_space:
        dir = os.path.join(PPD_DIR, f"reps_{n_reps}")
        src = os.path.join(dir, f"inference.pkl")
        with open(src, "rb") as f:
            model, posterior_samples, = pickle.load(f)
        models[n_reps] = model
        ps[n_reps] = posterior_samples
        src = os.path.join(dir, f"a_true.npy")
        a_true = np.load(src)
        src = os.path.join(dir, f"a_pred.npy")
        a_pred = np.load(src)
        logger.info(f"n_reps: {n_reps}, a_true: {a_true.shape}, a_pred: {a_pred.shape}")

    model = models[n_reps_space[0]]
    prediction_df = pd.DataFrame(
        np.linspace(0, 100, 100),
        columns=[model.intensity]
    )
    logger.info(prediction_df.head())
    prediction_df[model.features[0]] = 0

    ppd = {}
    for n_reps in n_reps_space:
        ppd[n_reps] = model.predict(df=prediction_df, posterior_samples=ps[n_reps])


if __name__ == "__main__":
    setup_logging(dir=BUILD_DIR, fname=os.path.basename(__file__))
    main()
