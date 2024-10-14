import logging
import pandas as pd
from hbmep.utils import timing

from hbmep_paper.cross_validate import run
from models import (
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    RectifiedLinear,
    MixtureModel,
)
from constants import (
    DATA_PATH,
    TOML_PATH,
    INFERENCE_FILE,
    BUILD_DIR
)

logger = logging.getLogger(__name__)


def main(M):
    # Load data
    data = pd.read_csv(DATA_PATH)
    response = ["ADM", "APB", "Biceps", "Triceps"]
    ind = ~data[response].isna().values.any(axis=-1)
    data = data[ind].reset_index(drop=True).copy()
    # Run inference
    run(
        data,
        M,
        TOML_PATH,
        BUILD_DIR,
        INFERENCE_FILE,
        max_tree_depth=(15, 15),
        extra_fields=[
            "potential_energy",
            "num_steps",
            "accept_prob",
        ]
    )
    return


if __name__ == "__main__":
    M = RectifiedLogistic
    # M = Logistic5
    # M = Logistic4
    # M = RectifiedLinear
    # M = MixtureModel
    main(M=M)
