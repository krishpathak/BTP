import numpy as np
from scipy.signal import hilbert


def tkeo(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    e = np.empty_like(x)
    e[..., 1:-1] = x[..., 1:-1] ** 2 - x[..., 2:] * x[..., :-2]
    e[..., 0], e[..., -1] = e[..., 1], e[..., -2]
    return e


def ago(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    e = np.empty_like(x)
    e[..., 3:] = x[..., 2:-1] * x[..., 1:-2] - x[..., 3:] * x[..., :-3]
    e[..., :3] = e[..., 3, None]
    return e


def edo(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.empty_like(x)
    y[..., 1:-1] = (x[..., 2:] - x[..., :-2]) / 2.0
    y[..., 0], y[..., -1] = y[..., 1], y[..., -2]
    return np.abs(hilbert(y, axis=-1)) ** 2


OPERATORS = {"TKEO": tkeo, "AGO": ago, "EDO": edo}


def improved_energy(modes, operator="TKEO"):
    modes = np.atleast_2d(np.asarray(modes, dtype=float))
    per_mode = OPERATORS[operator](modes)
    return per_mode, per_mode.sum(axis=0)


def conventional_energy(signal, operator="TKEO"):
    e = OPERATORS[operator](np.asarray(signal, dtype=float))
    return e[0] if np.ndim(signal) == 1 else e
