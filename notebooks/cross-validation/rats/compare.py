import os
import pickle
import logging

import arviz as az

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import (
    MixtureModel,
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    RectifiedLinear
)
from constants import (
    INFERENCE_FILE,
    BUILD_DIR
)

logger = logging.getLogger(__name__)

CROSS_VALIDATION_DIR = "/home/vishu/repos/hbmep-paper/reports/cross-validation"
RAT_DIR = BUILD_DIR
TMS_DIR = os.path.join(CROSS_VALIDATION_DIR, "tms")
INTRAOP_DIR = os.path.join(CROSS_VALIDATION_DIR, "intraoperative")

POSTERIOR_DIRS = [
    ("rats", RAT_DIR),
    ("tms", TMS_DIR),
    ("intraoperative", INTRAOP_DIR)
]

FUNCTIONAL_MODELS = [
    (
        "rectified_logistic",
        "Rectified-\nlogistic",
        F.rectified_logistic,
        [site.a, site.b, site.L, site.ell, site.H]
    ),
    (
        "logistic5",
        "Logistic-5",
        F.logistic5,
        [site.a, site.b, site.v, site.L, site.H]
    ),
    (
        "logistic4",
        "Logistic-4",
        F.logistic4,
        [site.a, site.b, site.L, site.H]
    ),
    (
        "rectified_linear",
        "Rectified-\nlinear",
        F.rectified_linear,
        [site.a, site.b, site.L]
    )
]
MIXTURE_MODELS = [
    (
        "mixture_model",
        "Mixture\ndistribution",
        F.rectified_logistic,
        [site.a, site.b, site.L, site.ell, site.H]
    ),
    (
        "rectified_logistic",
        "Gamma\ndistribution",
        F.rectified_logistic,
        [site.a, site.b, site.L, site.ell, site.H]
    )
]
FUNCTIONAL_MODELS_NO_LOGISTIC_5 = [
    (
        "rectified_logistic",
        "Rectified-\nlogistic",
        F.rectified_logistic,
        [site.a, site.b, site.L, site.ell, site.H]
    ),
    (
        "logistic4",
        "Logistic-4",
        F.logistic4,
        [site.a, site.b, site.L, site.H]
    ),
    (
        "rectified_linear",
        "Rectified-\nlinear",
        F.rectified_linear,
        [site.a, site.b, site.L]
    )
]
COLUMNS = ["rank", "elpd_diff", "dse"]


def compare(posterior_dirs, models, destination_path):
    logger.info(f"Building {destination_path} ...")
    model_dict, compare_dfs = {}, {}
    for dataset, posterior_dir in posterior_dirs:
        model_dict[dataset] = {}

        for model_dir, model_name, _, _ in models:
            src = os.path.join(posterior_dir, model_dir, INFERENCE_FILE)
            with open(src, "rb") as f:
                _, mcmc, _ = pickle.load(f)

            model_dict[dataset][model_name] = az.from_numpyro(mcmc)

    for dataset, _ in posterior_dirs:
        compare_dfs[dataset] = az.compare(model_dict[dataset]);
        logger.info(f"Dataset: {dataset}")
        logger.info(compare_dfs[dataset].to_string(columns=COLUMNS))

    dest = destination_path
    with open(dest, "wb") as f:
        pickle.dump((compare_dfs,), f)
    logger.info(f"Saved to {dest}")
    return


def main():
    compare(
        posterior_dirs=POSTERIOR_DIRS,
        models=FUNCTIONAL_MODELS,
        destination_path=os.path.join(BUILD_DIR, "compare_functional.pkl")
    )
    compare(
        posterior_dirs=POSTERIOR_DIRS,
        models=MIXTURE_MODELS,
        destination_path=os.path.join(BUILD_DIR, "compare_mixture.pkl")
    )
    compare(
        posterior_dirs=POSTERIOR_DIRS,
        models=FUNCTIONAL_MODELS_NO_LOGISTIC_5,
        destination_path=os.path.join(BUILD_DIR, "compare_functional_no_logistic5.pkl")
    )
    return


if __name__ == "__main__":
    os.makedirs(BUILD_DIR, exist_ok=True)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
