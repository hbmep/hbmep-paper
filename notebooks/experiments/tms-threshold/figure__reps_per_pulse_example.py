import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel
from core__number_of_reps_per_pulse import (
    N_REPS_PER_PULSE_SPACE,
    N_SUBJECTS
)
from utils import generate_nested_pulses
from constants import (
    TOML_PATH,
    REP,
    SIMULATION_DF,
    INFERENCE_FILE,
    SIMULATE_DATA_DIR,
    NUMBER_OF_REPS_PER_PULSE_DIR
)

logger = logging.getLogger(__name__)

SIMULATION_DF_PATH = os.path.join(SIMULATE_DATA_DIR, SIMULATION_DF)
SIMULATION_PPD_PATH = os.path.join(SIMULATE_DATA_DIR, INFERENCE_FILE)
BUILD_DIR = os.path.join(NUMBER_OF_REPS_PER_PULSE_DIR, "example")

N_PULSES = 64
N_SUBJECTS = 1
DRAW = 0


@timing
def main():
    # Load simulated dataframe
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    ppd_a = simulation_ppd[site.a]
    ppd_obs = simulation_ppd[site.obs]
    simulation_ppd = None
    del simulation_ppd
    gc.collect()

    # Set up logging
    simulator._make_dir(BUILD_DIR)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )

    # Generate nested pulses
    pulses_map = generate_nested_pulses(simulator, simulation_df)

    # Experiment space
    n_reps_space = N_REPS_PER_PULSE_SPACE
    pulses = pulses_map[N_PULSES]
    ind = \
        (simulation_df[simulator.features[0]] < N_SUBJECTS) & \
        (simulation_df[simulator.intensity].isin(pulses))
    simulation_df = simulation_df[ind].reset_index(drop=True).copy()
    ppd_a = ppd_a[:, :N_SUBJECTS, ...]
    ppd_obs = ppd_obs[:, ind, ...]

    for n_reps in n_reps_space:
        pulses = pulses_map[N_PULSES][::n_reps]
        ind = (
            (simulation_df[REP] < n_reps) &
            (simulation_df[simulator.intensity].isin(pulses))
        )
        df = simulation_df[ind].reset_index(drop=True).copy()
        df[simulator.response[0]] = ppd_obs[DRAW, ind, 0]

        ind = df[simulator.response[0]] > 0
        df = df[ind].reset_index(drop=True).copy()

        # Build model
        toml_path = TOML_PATH
        config = Config(toml_path=toml_path)
        config.BUILD_DIR = os.path.join(
            BUILD_DIR,
            f"reps_{n_reps}"
        )
        model = HierarchicalBayesianModel(config=config)
        model._make_dir(model.build_dir)

        model.plot(df=df)

        # Run inference
        df, encoder_dict = model.load(df=df)
        _, posterior_samples = model.run_inference(df=df)

        # Predictions and recruitment curves
        prediction_df = model.make_prediction_dataset(df=df, num_points=100)
        posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
        model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

        a_true = ppd_a[DRAW, :N_SUBJECTS, ...]
        a_pred = posterior_samples[site.a]
        a_pred_map = a_pred.mean(axis=0)
        logger.info(f"a_true: {a_true.shape}")
        logger.info(f"a_pred: {a_pred.shape}")
        logger.info(f"a_pred_map: {a_pred_map.shape}")

        mae = np.abs(a_true - a_pred_map).mean()
        logger.info(f"n_reps:{n_reps} MAE: {mae:.4f}")

        # Save
        dest = os.path.join(model.build_dir, INFERENCE_FILE)
        with open(dest, "wb") as g:
            pickle.dump((model, posterior_samples,), g)

        dest = os.path.join(model.build_dir, "a_true.npy")
        np.save(dest, a_true)
        dest = os.path.join(model.build_dir, "a_pred.npy")
        np.save(dest, a_pred)

        dest = os.path.join(model.build_dir, "df.csv")
        df.to_csv(dest, index=False)

    return


if __name__ == "__main__":
    main()
