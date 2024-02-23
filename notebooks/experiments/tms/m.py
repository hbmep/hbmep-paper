import logging

import numpy as np

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model import BoundedOptimization
from hbmep.model.utils import Site as site

logger = logging.getLogger(__name__)

BUILD_DIR = "/home/vishu/testing/opt"


class NelderMeadOptimization(BoundedOptimization):
    NAME = "nelder_mead_optimization"

    def __init__(self, config: Config):
        super(NelderMeadOptimization, self).__init__(config=config)
        self.solver = "Nelder-Mead"
        self.functional = F.rectified_logistic
        self.named_params = [site.a, site.b, site.v, site.L, site.ell, site.H]
        self.bounds = [(1e-9, 150.), (1e-9, 10), (1e-9, 10), (1e-9, 10), (1e-9, 10), (1e-9, 10)]
        self.informed_bounds = [(20, 80), (1e-3, 5.), (1e-3, 5.), (1e-4, .1), (1e-2, 5), (.5, 5)]
        self.num_points = 1000
        self.num_iters = 10
        self.n_jobs = -1


from constants import TOML_PATH, DATA_PATH
import pandas as pd

def main():
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR
    model = NelderMeadOptimization(config=config)
    df = pd.read_csv(DATA_PATH)
    df, _ = model.load(df=df)
    params = model.run_inference(df=df)
    print(params)
    return


if __name__ == "__main__":
    main()
