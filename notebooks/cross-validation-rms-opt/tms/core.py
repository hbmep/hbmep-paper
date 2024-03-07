import os
import gc
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from hbmep.config import Config
from hbmep.model import BaseModel

from hbmep_paper.utils import setup_logging
from models import (
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    RectifiedLinear
)
from constants import (
    DATA_PATH,
    TOML_PATH,
    BUILD_DIR,
    N_SPLITS,
    FOLD_COLUMNS,
    PARAMS_FILE,
    MSE_FILE
)

logger = logging.getLogger(__name__)


def _get_kfolds(df, n_splits=2, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i, (_, test_index) in enumerate(kf.split(df.index)):
        df[FOLD_COLUMNS[i]] = False
        df.loc[df.index[test_index], FOLD_COLUMNS[i]] = True

    return df


def main():
    # Load data
    data = pd.read_csv(DATA_PATH)

    M = BaseModel

    # Build model
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR
    model = M(config=config)

    data, encoder_dict = model.load(df=data)
    data = data[[model.intensity, *model.features, *model.response]]

    data = (
        data
        .groupby(by=model.features)
        .apply(_get_kfolds, n_splits=N_SPLITS)
        .reset_index(drop=True)
        .copy()
    )
    logger.info(f"\n{data.to_string()}")

    combinations = model._get_combinations(df=data, columns=model.features)
    for c in combinations:
        ind = data[model.features].apply(tuple, axis=1).isin([c])
        logger.info(c)
        temp_df = data[ind].reset_index(drop=True).copy()
        for column in FOLD_COLUMNS: logger.info(temp_df[column].sum())


    # Define experiment
    def run_inference(M):
        df = data.copy()

        for fold_column in FOLD_COLUMNS:
            # Build model
            config = Config(toml_path=TOML_PATH)
            config.BUILD_DIR = os.path.join(
                BUILD_DIR,
                M.NAME,
                fold_column
            )
            model = M(config=config)

            # Set up logging
            model._make_dir(model.build_dir)
            setup_logging(
                dir=model.build_dir,
                fname=os.path.basename(__file__)
            )

            ind = df[fold_column].isin([True])
            test = df[ind].reset_index(drop=True).copy()
            train = df[~ind].reset_index(drop=True).copy()

            logger.info(
                f"{fold_column}: Running inference for {model.NAME} with {train.shape[0]} samples ..."
            )
            params = model.run_inference(df=train)

            # # Predictions and recruitment curves
            # prediction_df = model.make_prediction_dataset(df=train)
            # prediction_df = model.predict(df=prediction_df, params=params)
            # model.render_recruitment_curves(
            #     df=train,
            #     encoder_dict=encoder_dict,
            #     params=params,
            #     prediction_df=prediction_df
            # )

            # Evaluate
            logger.info(f"{fold_column}: Evaluating {model.NAME} on held out set with {test.shape[0]} samples ...")
            y_true = test[model.response].values
            y_pred = model.predict(df=test, params=params)[model.response].values
            mse = mean_squared_error(y_true, y_pred)
            logger.info(f"{fold_column}: Mean Squared Error: {mse}")

            # Save
            dest = os.path.join(model.build_dir, PARAMS_FILE)
            np.save(dest, params)
            dest = os.path.join(model.build_dir, MSE_FILE)
            np.save(dest, np.array([mse]))

            config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
            model, params = None, None
            train, test = _, _
            del config, df, prediction_df, encoder_dict, _
            del model, params
            del train, test
            gc.collect()


    models = [
        RectifiedLogistic,
        Logistic5,
        Logistic4,
        RectifiedLinear,
    ]
    for M in models: run_inference(M)

    return


if __name__ == "__main__":
    setup_logging(dir=BUILD_DIR, fname=os.path.basename(__file__))
    main()
