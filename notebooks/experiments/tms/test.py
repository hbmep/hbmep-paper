import os
import pickle
import logging

from utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    dir = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/"
    setup_logging(dir=dir, fname=os.path.basename(__file__))

    # src = "/home/vishu/a_random_mean_-2.5_a_random_scale_1.5/simulation_ppd.pkl"
    # with open(src, "rb") as g:
    #     _, original_ppd = pickle.load(g)

    # src = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/a_random_mean_-2.5_a_random_scale_1.5/simulation_ppd.pkl"
    # with open(src, "rb") as g:
    #     _, new_ppd = pickle.load(g)

    # for k, v in original_ppd.items():
    #     flag = (original_ppd[k] == new_ppd[k]).all()
    #     logger.info(f"{k}: {flag}")


if __name__ == "__main__":
    main()
