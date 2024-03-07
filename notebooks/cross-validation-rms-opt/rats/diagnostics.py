import os
import gc
import pickle
import logging

import arviz as az
import pandas as pd
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from test_models import (
    VBounded
)
from constants import (
    DATA_PATH,
    TOML_PATH,
    INFERENCE_FILE,
    BUILD_DIR
)

logger = logging.getLogger(__name__)
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/cross-validation/rats/test_models/"


@timing
def main():
    # Load data
    data = pd.read_csv(DATA_PATH)


    # Define experiment
    def run_diagnostics(M, var_names):
        config = Config(toml_path=TOML_PATH)
        config.BUILD_DIR = os.path.join(
            BUILD_DIR,
            M.NAME,
        )
        model = M(config=config)

        src = os.path.join(model.build_dir, INFERENCE_FILE)
        with open(src, "rb") as f:
            _, _, posterior_samples = pickle.load(f)

        # Set up logging
        model.build_dir = os.path.join(model.build_dir, "diagnostics")
        model._make_dir(model.build_dir)
        setup_logging(
            dir=model.build_dir,
            fname=os.path.basename(__file__)
        )

        df, encoder_dict = model.load(df=data)

        model.render_diagnostics(
            df=df,
            destination_path=os.path.join(model.build_dir, "diagnostics.pdf"),
            posterior_samples=posterior_samples,
            var_names=var_names,
            encoder_dict=encoder_dict
        )


    # Run multiple models in parallel
    n_jobs = -1
    models = [
        (VBounded, [site.a, site.b, site.v, site.L, site.ell, site.H]),
        # (Logistic5, [site.a, site.b, site.v, site.L, site.H]),
        # (Logistic4, [site.a, site.b, site.L, site.H]),
        # (RectifiedLinear, [site.a, site.b, site.L])
    ]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_diagnostics)(M, var_names)
            for M, var_names in models
        )

    # # Run single model
    # M = RectifiedLogistic
    # run_inference(M)

    return


if __name__ == "__main__":
    main()
