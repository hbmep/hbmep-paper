import logging
import warnings
from functools import partial

import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import biject_to
from numpyro.util import find_stack_level

from hbmep.nn.functional import EPSILON

from constants import (
    TOTAL_PULSES,
    N_PULSES_SPACE
)

logger = logging.getLogger(__name__)


def generate_nested_pulses(simulator, simulation_df):
    all_pulses = simulation_df[simulator.intensity].unique()
    all_pulses = np.sort(all_pulses)

    assert TOTAL_PULSES == all_pulses.shape[0]
    assert TOTAL_PULSES == N_PULSES_SPACE[-1]

    pulses_map = {}
    pulses_map[TOTAL_PULSES] = \
        np.arange(0, TOTAL_PULSES, 1).astype(int).tolist()

    for i in range(len(N_PULSES_SPACE) - 2, -1, -1):
        n_pulses = N_PULSES_SPACE[i]
        subsample_from = pulses_map[N_PULSES_SPACE[i + 1]]
        ind = \
            np.round(np.linspace(0, len(subsample_from) - 1, n_pulses)) \
            .astype(int).tolist()
        pulses_map[n_pulses] = np.array(subsample_from)[ind].tolist()

    for i in range(len(N_PULSES_SPACE) - 1, -1, -1):
        n_pulses = N_PULSES_SPACE[i]
        pulses_map[n_pulses] = list(
            all_pulses[pulses_map[n_pulses]]
        )
        assert set(pulses_map[n_pulses]) <= set(all_pulses)
        assert len(pulses_map[n_pulses]) == n_pulses
        if n_pulses != TOTAL_PULSES:
            assert set(pulses_map[n_pulses]) <= set(
                pulses_map[N_PULSES_SPACE[i + 1]]
            )

    return pulses_map


def init_to_uniform(site=None, radius=2):
    """
    Initialize to a random point in the area `(-radius, radius)` of unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the unconstrained domain.
    """
    if site is None:
        return partial(init_to_uniform, radius=radius)

    if (
        site["type"] == "sample"
        and not site["is_observed"]
        and not site["fn"].support.is_discrete
    ):
        if site["value"] is not None:
            warnings.warn(
                f"init_to_uniform() skipping initialization of site '{site['name']}'"
                " which already stores a value.",
                stacklevel=find_stack_level(),
            )
            return site["value"]

        # XXX: we import here to avoid circular import
        from numpyro.infer.util import helpful_support_errors

        rng_key = site["kwargs"].get("rng_key")
        sample_shape = site["kwargs"].get("sample_shape")

        with helpful_support_errors(site):
            transform = biject_to(site["fn"].support)
        unconstrained_shape = transform.inverse_shape(site["fn"].shape())
        unconstrained_samples = dist.Uniform(EPSILON, radius)(
            rng_key=rng_key, sample_shape=sample_shape + unconstrained_shape
        )
        return transform(unconstrained_samples)
