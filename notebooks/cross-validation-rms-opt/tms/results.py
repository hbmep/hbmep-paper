import os
import logging

import numpy as np
from scipy import stats

from hbmep_paper.utils import setup_logging

from models import (
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    RectifiedLinear
)
from constants import (
    BUILD_DIR, FOLD_COLUMNS, PARAMS_FILE, MSE_FILE
)

logger = logging.getLogger(__name__)


def main():
    models = [RectifiedLogistic, Logistic5, Logistic4, RectifiedLinear]
    mse = []
    for fold_column in FOLD_COLUMNS:
        for M in models:
            src = os.path.join(BUILD_DIR, M.NAME, fold_column, MSE_FILE)
            curr_mse = np.load(src)
            curr_mse = curr_mse[0]
            mse.append(curr_mse)

    mse = np.array(mse).reshape(len(FOLD_COLUMNS), len(models))
    logger.info(f"MSE averaged across {len(FOLD_COLUMNS)} splits: {mse.mean(axis=0)}")

    diff = mse[:, 0] - mse[:, 1]
    ttest = stats.ttest_1samp(diff, popmean=0, alternative="less")
    logger.info(f"ttest_1sample: RectifiedLogistic vs Logistic5: {ttest}")

    diff = mse[:, 0] - mse[:, 2]
    ttest = stats.ttest_1samp(diff, popmean=0, alternative="less")
    logger.info(f"ttest_1sample: RectifiedLogistic vs Logistic4: {ttest}")

    diff = mse[:, 0] - mse[:, 3]
    ttest = stats.ttest_1samp(diff, popmean=0, alternative="less")
    logger.info(f"ttest_1sample: RectifiedLogistic vs RectifiedLinear: {ttest}")

    diff = mse[:, 0] - mse[:, 1]
    ttest = stats.wilcoxon(diff, alternative="less")
    logger.info(f"wilcoxon: RectifiedLogistic vs Logistic5: {ttest}")

    diff = mse[:, 0] - mse[:, 2]
    ttest = stats.wilcoxon(diff, alternative="less")
    logger.info(f"wilcoxon: RectifiedLogistic vs Logistic4: {ttest}")

    diff = mse[:, 0] - mse[:, 3]
    ttest = stats.wilcoxon(diff, alternative="less")
    logger.info(f"wilcoxon: RectifiedLogistic vs RectifiedLinear: {ttest}")

    return


if __name__ == "__main__":
    setup_logging(dir=BUILD_DIR, fname=os.path.basename(__file__))
    main()
