import os
import sys
import logging

import pandas as pd
from hbmep.util import timing, setup_logging
from joblib import Parallel, delayed

from paper.model import HB
from paper.util import run
from paper.constants import (
    RAT_TOML, RAT_DATA, 
    TMS_TOML, TMS_DATA,
    INTRAOPERATIVE_TOML, INTRAOPERATIVE_DATA
)
from constants import BUILD_DIR

logger = logging.getLogger(__name__)
PLOT_DATA, TEST_RUN = False, False
# PLOT_DATA, TEST_RUN = False, True


@timing
def run_model(model, data_path, use_higher_depth=False):
    run_id = model.run_id
    df = pd.read_csv(data_path)
    if run_id == "intraoperative":
        idx = ~df[model.response].isna().values.any(axis=-1)
        df = df[idx].reset_index(drop=True).copy()
    if PLOT_DATA:
        model.plot(df=df)
        return

    if TEST_RUN:
        model.response = model.response[:3]
        if run_id == "rat":
            idx = (
                df[model.features[0]].isin(["amap01", "amap02"])
                & df[model.features[1]].isin(["-C5L", "-C6L"]) 
            )
        elif run_id == "tms":
            idx = df[model.features[0]].isin(["SCA01", "SCA04"])
        elif run_id == "intraoperative":
            idx = df[model.features[0]].isin(
                ["cornptio001", "cornptio003", "scapptio001"]
            )
        else:
            raise ValueError
        df = df[idx].reset_index(drop=True).copy()
        model.mcmc_params = {
            "thinning": 1,
            "num_chains": 4,
            "num_warmup": 400,
            "num_samples": 400,
        }

    logger.info(f"*** run id: {run_id} ***")
    logger.info(f"*** model: {model._model.__name__} ***")
    for u, v in model.mcmc_params.items(): logger.info(f"{u}: {v}")
    for u, v in model.nuts_params.items(): logger.info(f"{u}: {v}")
    logger.info(f"use_mixture: {model.use_mixture}")
    logger.info(f"response: {model.response}")
    model.features = [model.features]
    run(df, model, extra_fields=["num_steps"])
    return


def main(
    run_id: str,
	model_name: str,
	use_mixture: int = 0,
	response_id: int = -1,
    depth: int = 15
):
    use_mixture = int(use_mixture)
    response_id = int(response_id)
    depth = int(depth)
    match run_id:
        case "rat": 
            toml_path = RAT_TOML
            data_path = RAT_DATA
        case "tms":
            toml_path = TMS_TOML
            data_path = TMS_DATA
        case "intraoperative":
            toml_path = INTRAOPERATIVE_TOML
            data_path = INTRAOPERATIVE_DATA
        case _:
            raise ValueError
    model = HB(toml_path=toml_path)
    match model_name:
        case "rlog": model._model = model.rectified_logistic
        case "l5": model._model = model.logistic5
        case "l4": model._model = model.logistic4
        case "rlin": model._model = model.rectified_linear
        case "ln_rlog": model._model = model.lognormal_rlog
        case "ln2_rlog": model._model = model.ln_rlog
        case _: raise ValueError
    if use_mixture: model.use_mixture = True
    if response_id != -1:
        assert response_id in range(6)
        model.response = model.response[response_id: response_id + 1]

    model.mcmc_params = {
        "thinning": 4,
        "num_chains": 4,
        "num_warmup": 4000,
        "num_samples": 4000,
    }
    model.nuts_params = {
        "max_tree_depth": (depth, depth),
        "target_accept_prob": .95,
    }

    model.run_id = run_id
    model.build_dir = os.path.join(
        BUILD_DIR, model.run_id, model.name, model._model.__name__
    )
    if TEST_RUN:
        model.build_dir = os.path.join(model.build_dir, "test_run")
    if response_id != -1:
        assert len(model.response) == 1
        assert model.num_response == 1
        model.build_dir = os.path.join(model.build_dir, model.response[0])
    setup_logging(model.build_dir)
    run_model(model, data_path)


if __name__ == "__main__":

    args = sys.argv[1:]
    main(*args)

    # args_space = [
    #     # ["ln2_rlog", 0],
    #     ["ln_rlog", 0],
    #     # ["rlog", 1],
    #     # ["rlog", 0],
    #     # ["l5", 0],
    #     # ["l4", 0],
    #     # ["rlin", 0],
    # ] 
    # datasets = [
    #     # "rat",
    #     "tms",
    #     # "intraoperative"
    # ]
    # with Parallel(n_jobs=-1) as parallel:
    #     parallel(
    #         delayed(main)(dataset, *args, -1)
    #         for dataset in datasets
    #         for args in args_space
    #     )

    # with Parallel(n_jobs=-1) as parallel:
    #     parallel(
    #         delayed(main)("rat", model_name, 0, response_id, 0)
    #         for response_id in range(6)
    #         for model_name in ["rlog"]
    #     )
