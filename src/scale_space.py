import numpy as np
from scipy.special import ive


def otsu_threshold(values, nbins=64):
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return 0.0
    if np.allclose(v, v[0]):
        return float(v[0])

    hist, edges = np.histogram(v, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(float) / hist.sum()

    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = -np.inf

    return float(centers[int(np.argmax(sigma_b2))])


def _discrete_gaussian_kernel(t, radius=None):
    if radius is None:
        radius = max(1, int(np.ceil(4.0 * np.sqrt(t))))
    n = np.arange(-radius, radius + 1)
    k = ive(np.abs(n), t)
    s = k.sum()
    return k / s if s > 0 else k


def _smooth(sig, t):
    k = _discrete_gaussian_kernel(t)
    r = (k.size - 1) // 2
    padded = np.pad(sig, r, mode="reflect")
    return np.convolve(padded, k, mode="valid")


def _smooth_step(sig, kernel, radius):
    padded = np.pad(sig, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _local_minima(sig):
    if sig.size < 3:
        return np.array([], dtype=int)
    left = sig[1:-1] < sig[:-2]
    right = sig[1:-1] <= sig[2:]
    return np.flatnonzero(left & right) + 1


def scale_space_lengths(spectrum, num_scales=None, scale_step=0.5,
                        track_window=1):
    spec = np.asarray(spectrum, dtype=float).ravel()
    if spec.size < 5:
        return np.array([], dtype=int), np.array([], dtype=int)

    if num_scales is None:
        num_scales = max(20, spec.size // 2)

    seeds = _local_minima(spec)
    if seeds.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    kernel = _discrete_gaussian_kernel(scale_step)
    radius = (kernel.size - 1) // 2

    births = seeds.copy()
    current = seeds.copy()
    lengths = np.ones(seeds.size, dtype=int)
    alive = np.ones(seeds.size, dtype=bool)

    plane = spec
    for _ in range(int(num_scales)):
        if not alive.any():
            break
        plane = _smooth_step(plane, kernel, radius)
        minima = _local_minima(plane)
        if minima.size == 0:
            break
        # follow each minimum to the nearest one in the smoothed spectrum
        for i in np.flatnonzero(alive):
            d = np.abs(minima - current[i])
            j = int(np.argmin(d))
            if d[j] <= track_window:
                current[i] = minima[j]
                lengths[i] += 1
            else:
                alive[i] = False

    return births, lengths


def magnitude_spectrum(signal):
    x = np.asarray(signal, dtype=float).ravel()
    x = x - x.mean()
    mag = np.abs(np.fft.rfft(x))
    m = mag.max()
    return mag / m if m > 0 else mag


def detect_boundaries(signal, num_scales=None, scale_step=0.5):
    spec = magnitude_spectrum(signal)
    pos, lengths = scale_space_lengths(spec, num_scales, scale_step)
    if pos.size == 0:
        return np.array([0, spec.size - 1])

    thr = otsu_threshold(lengths)
    keep = pos[lengths > thr]
    if keep.size == 0:
        keep = pos[[int(np.argmax(lengths))]]

    return np.unique(np.concatenate(([0], np.sort(keep), [spec.size - 1])))


def estimate_nvm(signal, num_scales=None, scale_step=0.5, nvm_min=2, nvm_max=12):
    b = detect_boundaries(signal, num_scales, scale_step)
    return int(np.clip(b.size - 1, nvm_min, nvm_max))
