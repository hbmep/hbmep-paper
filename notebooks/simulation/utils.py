import numpy as np


def generate_pulses_map(n_pulses_space):
    d = {}
    for n in n_pulses_space:
        temp = np.linspace(0, 100, n)
        temp = np.round(temp * 2) / 2
        temp = sorted(list(set(temp)))
        assert len(temp) == n
        d[n] = temp
    return d
