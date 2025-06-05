import os
import sys
import logging

import pandas as pd
from hbmep.util import timing, setup_logging

from paper.util import run
from paper.constants import RAT_DATA
from model import HB

from constants import (
    DATA_PATH,
    TOML_PATH,
    BUILD_DIR
)

logger = logging.getLogger(__name__)


@timing
def main(model):
    df = pd.read_csv(DATA_PATH)
    # model.features = model.features[0]
    # model.plot(df=df)
    # return

    if model.test_run:
        model.build_dir = os.path.join(model.build_dir, "test_run")
        os.makedirs(model.build_dir, exist_ok=True)
        model.response = model.response[:3]
        idx = (
            df[model.features[0][0]].isin(["amap01", "amap02"])
            & df[model.features[0][1]].isin(["-C5L", "-C6L"]) 
        )
        df = df[idx].reset_index(drop=True).copy()
        model.mcmc_params = {
            "thinning": 1,
            "num_chains": 4,
            "num_warmup": 400,
            "num_samples": 400,
        }

    logger.info(f"*** model: {model._model.__name__} ***")
    run(df, model, extra_fields=["num_steps"])
    return


if __name__ == "__main__":
    model = HB(toml_path=TOML_PATH)
    model.features = [model.features]

    # model.test_run = True
    # model.use_mixture = True
    # model._model = model.rectified_logistic
    # model._model = model.logistic5
    # model._model = model.logistic4
    # model._model = model.rectified_linear

    args = sys.argv[1:]
    model_name, use_mixture, response_id = args
    use_mixture = int(use_mixture)
    response_id = int(response_id)
    match model_name:
        case "rlog": model._model = model.rectified_logistic
        case "l5": model._model = model.logistic5
        case "l4": model._model = model.logistic4
        case "rlin": model._model = model.rectified_linear
        case _: raise ValueError
    if use_mixture: model.use_mixture = True
    if response_id != -1:
        model.response = model.response[response_id]

    model.mcmc_params = {
        "thinning": 4,
        "num_chains": 4,
        "num_warmup": 4000,
        "num_samples": 4000,
    }
    model.nuts_params = {
        "max_tree_depth": (15, 15),
        "target_accept_prob": .95,
    }

    model.build_dir = os.path.join(
        BUILD_DIR, model.name, model._model.__name__
    )
    if response_id != -1:
        assert len(model.response) == 1
        assert model.num_response == 1
        model.build_dir = os.path.join(model.build_dir, model.repsonse[0])
    setup_logging(model.build_dir)
    main(model)
