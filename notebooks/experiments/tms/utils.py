import logging

import numpy as np
import jax

from hbmep.config import Config

from constants import (
    TOTAL_REPS,
    N_DRAWS,
    N_SEEDS,
    TOTAL_PULSES,
    ALL_PULSES_SPACE
)

logger = logging.getLogger(__name__)


def fix_keys_for_reps(rng_key):
    rng_keys = jax.random.split(rng_key, num=3)
    rng_keys = list(rng_keys)
    rng_keys = jax.random.split(rng_keys[-1], num=TOTAL_REPS)
    rng_keys = list(rng_keys)
    return rng_keys


def fix_draws_and_seeds(rng_key, max_draws, max_seeds):
    rng_keys = jax.random.split(rng_key, num=2)
    rng_keys = list(rng_keys)
    draws_space = \
        jax.random.choice(
            key=rng_keys[0],
            a=np.arange(0, max_draws, 1),
            shape=(N_DRAWS,),
            replace=False
        ) \
        .tolist()
    seeds_for_generating_subjects = \
        jax.random.choice(
            key=rng_keys[1],
            a=np.arange(0, max_seeds, 1),
            shape=(N_SEEDS,),
            replace=False
        ) \
        .tolist()
    logger.info(f"Draws: {draws_space}")
    logger.info(
        f"Seeds for generating subjects: {seeds_for_generating_subjects}"
    )
    return draws_space, seeds_for_generating_subjects


def fix_nested_pulses(simulator, simulation_df):
    all_pulses = simulation_df[simulator.intensity].unique()
    all_pulses = np.sort(all_pulses)

    assert TOTAL_PULSES == all_pulses.shape[0]
    assert TOTAL_PULSES == ALL_PULSES_SPACE[-1]

    pulses_map = {}
    pulses_map[TOTAL_PULSES] = \
        np.arange(0, TOTAL_PULSES, 1).astype(int).tolist()

    for i in range(len(ALL_PULSES_SPACE) - 2, -1, -1):
        n_pulses = ALL_PULSES_SPACE[i]
        subsample_from = pulses_map[ALL_PULSES_SPACE[i + 1]]
        ind = \
            np.round(np.linspace(0, len(subsample_from) - 1, n_pulses)) \
            .astype(int).tolist()
        pulses_map[n_pulses] = np.array(subsample_from)[ind].tolist()

    for i in range(len(ALL_PULSES_SPACE) - 1, -1, -1):
        n_pulses = ALL_PULSES_SPACE[i]
        pulses_map[n_pulses] = list(
            all_pulses[pulses_map[n_pulses]]
        )
        assert set(pulses_map[n_pulses]) <= set(all_pulses)
        if n_pulses != TOTAL_PULSES:
            assert set(pulses_map[n_pulses]) <= set(
                pulses_map[ALL_PULSES_SPACE[i + 1]]
            )
        logger.info(f"n_pulses: {n_pulses}, pulses: {pulses_map[n_pulses]}")

    return pulses_map
