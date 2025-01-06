import os
import gc
import pickle
import logging

import numpy as np
from joblib import Parallel, delayed

from hbmep.utils import timing

from models import (
    Simulator,
    HB,
    HBTight_above_sigma_g,
    HBTight_sigma_g,
    HBZero
)
from constants import (
    BUILD_DIR
)

logger = logging.getLogger(__name__)
EDGE_CASES = [
    "/home/vishu/repos/hbmep-paper/reports/simulations-multiple-comparisons/with_no_effect/inference.pkl",
    "/home/vishu/repos/hbmep-paper/reports/simulations-multiple-comparisons/with_effect/inference.pkl",
]


@timing
def main(
    simulation_data_src,
    build_dir,
	draws_space,
	n_subjects_space,
    n_muscles_space,
	models,
	n_jobs=-1
):
    os.makedirs(build_dir, exist_ok=True)
    # Load simulation data
    if simulation_data_src not in EDGE_CASES:
        with open(simulation_data_src, "rb") as g:
            _, _, y_obs = pickle.load(g)
    else:
        with open(simulation_data_src, "rb") as g:
            _, simulation_ppd = pickle.load(g)

        ppd_a = simulation_ppd["a"]
        y_obs = ppd_a[..., 1, :] - ppd_a[..., 0, :]


    # Define experiment
    def run_experiment(
        n_subjects,
        n_muscles,
        draw,
        M
    ):
        # Required for build directory
        n_subjects_dir = f"n{n_subjects}"
        n_muscles_dir = f"m{n_muscles}"
        draw_dir = f"d{draw}"

        # Build model
        model = M()
        model.build_dir = os.path.join(
            build_dir,
            n_muscles_dir,
            draw_dir,
            n_subjects_dir,
            M.NAME
        )

        y_true = y_obs[draw, :n_subjects, :n_muscles]
        print(f"y_true: {y_true.shape}")

        # Run inference
        os.makedirs(model.build_dir, exist_ok=True)
        _, posterior_samples = model.run_inference(y_obs=y_true)

        # Predictions and recruitment curves
        y_loc = posterior_samples["y_loc"]
        np.save(os.path.join(model.build_dir, "y_loc.npy"), y_loc)

        model, y_true, y_loc, _, posterior_samples = None, None, None, None, None
        del model, y_true, y_loc, _, posterior_samples
        gc.collect()

        return


    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(n_subjects, n_muscles, draw, M)
            for draw in draws_space
            for n_subjects in n_subjects_space
            for n_muscles in n_muscles_space
            for M in models
        )


if __name__ == "__main__":
    draws_space = range(0, 2000)
    n_subjects_space = [2, 4, 8, 12, 16, 20]
    n_muscles_space = [4]

    models = [HB, HBZero]

    # simulation_data_src = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/with_effect_individual.pkl"
    # build_dir = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/with_effect_individual"

    # simulation_data_src = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/with_effect.pkl"
    # build_dir = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/with_effect"

    # simulation_data_src = "/home/vishu/repos/hbmep-paper/reports/simulations-multiple-comparisons/with_no_effect/inference.pkl"
    # build_dir = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/paper/with_no_effect"

    simulation_data_src = "/home/vishu/repos/hbmep-paper/reports/simulations-multiple-comparisons/with_effect/inference.pkl"
    build_dir = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/paper/with_effect"

    main(
        simulation_data_src=simulation_data_src,
        build_dir=build_dir,
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        n_muscles_space=n_muscles_space,
        models=models,
        n_jobs=-1
    )
