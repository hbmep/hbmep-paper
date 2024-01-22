import os
import gc
import time
import pickle
import logging

import numpy as np
import pandas as pd

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from utils import fix_nested_pulses

logger = logging.getLogger(__name__)


def main():
    dir = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/"
    setup_logging(dir=dir, fname=os.path.basename(__file__))

    # old_dir = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_-3.0_a_random_scale_1.5/old"
    # new_dir = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_-3.0_a_random_scale_1.5"

    # src = os.path.join(old_dir, "simulation_df.csv")
    # old_df = pd.read_csv(src)

    # src = os.path.join(new_dir, "simulation_df.csv")
    # new_df = pd.read_csv(src)

    # src = os.path.join(old_dir, "simulation_ppd.pkl")
    # with open(src, "rb") as g:
    #     _, old_ppd = pickle.load(g)

    # src = os.path.join(new_dir, "simulation_ppd.pkl")
    # with open(src, "rb") as g:
    #     _, new_ppd = pickle.load(g)

    # assert (new_df == old_df).all().all()

    # subset = [
    #     site.a,
    #     site.b,
    #     site.v,
    #     site.L,
    #     site.ell,
    #     site.H,
    #     site.c_1,
    #     site.c_2,
    #     site.mu,
    #     site.beta,
    #     site.alpha,
    #     site.obs
    # ]
    # for u in subset:
    #     flag = (old_ppd[u] == new_ppd[u]).all()
    #     logger.info(f"{u}: {flag}")

    pulses_map = fix_nested_pulses()
    return


if __name__ == "__main__":
    main()
