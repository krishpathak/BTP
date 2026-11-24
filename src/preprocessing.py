import sys
from math import gcd
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C


def pick_lead_ii(record):
    names = [str(s).strip().upper() for s in record.sig_name]
    for target in ("II", "ECG", "MLII", "LEAD II"):
        if target in names:
            return record.p_signal[:, names.index(target)].astype(float)
    return record.p_signal[:, 0].astype(float)


def fill_gaps(x):
    # a few Fantasia records have isolated NaNs; the wavelet transform would
    # spread each one and wipe out the whole record
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    if bad.all():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def wavelet_denoise(x, wavelet=None, level=None, kill_details=None):
    wavelet = wavelet or C.WAVELET
    level = level or C.WAVELET_LEVEL
    kill_details = C.DENOISE_DETAIL_KILL if kill_details is None else kill_details

    x = fill_gaps(x)
    max_level = pywt.dwt_max_level(x.size, pywt.Wavelet(wavelet).dec_len)
    level = int(min(level, max_level))
    if level < 2:
        return x - np.mean(x)

    coeffs = pywt.wavedec(x, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])
    for d in range(1, kill_details + 1):
        coeffs[-d] = np.zeros_like(coeffs[-d])

    y = pywt.waverec(coeffs, wavelet)
    return y[: x.size]


def resample_to(x, fs_in, fs_out=None):
    fs_out = fs_out or C.TARGET_FS
    if int(fs_in) == int(fs_out):
        return np.asarray(x, dtype=float)
    g = gcd(int(fs_out), int(fs_in))
    return resample_poly(np.asarray(x, dtype=float),
                         int(fs_out) // g, int(fs_in) // g).astype(float)


def pan_tompkins(x, fs):
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs

    lo, hi = 5.0 / nyq, min(15.0 / nyq, 0.99)
    b, a = butter(3, [lo, hi], btype="band")
    filtered = filtfilt(b, a, x)

    deriv = np.convolve(filtered, np.array([1, 2, 0, -2, -1]) * fs / 8.0,
                        mode="same")
    squared = deriv ** 2

    win = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    refractory = int(round(0.200 * fs))

    cand = np.flatnonzero((integrated[1:-1] > integrated[:-2]) &
                          (integrated[1:-1] >= integrated[2:])) + 1
    if cand.size == 0:
        return np.array([], dtype=int)

    init = integrated[: min(integrated.size, int(2 * fs))]
    spki = float(np.max(init)) * 0.25 if init.size else float(np.max(integrated))
    npki = float(np.mean(init)) * 0.5 if init.size else 0.0

    peaks, rr = [], []
    last = -refractory
    for p in cand.tolist():
        thr1 = npki + 0.25 * (spki - npki)
        val = integrated[p]

        if val > thr1 and p - last > refractory:
            peaks.append(p)
            if len(peaks) > 1:
                rr.append(peaks[-1] - peaks[-2])
                rr = rr[-8:]
            spki = 0.125 * val + 0.875 * spki
            last = p
            continue

        npki = 0.125 * val + 0.875 * npki

        # missed beat: nothing for 1.66 x mean RR, look back with a lower threshold
        if rr and peaks and (p - peaks[-1]) > 1.66 * np.mean(rr):
            seg_lo, seg_hi = peaks[-1] + refractory, p
            if seg_hi > seg_lo:
                seg = integrated[seg_lo:seg_hi]
                j = int(np.argmax(seg))
                if seg[j] > 0.5 * thr1:
                    q = seg_lo + j
                    peaks.append(q)
                    rr.append(peaks[-1] - peaks[-2])
                    rr = rr[-8:]
                    spki = 0.25 * seg[j] + 0.75 * spki
                    last = q

    if not peaks:
        return np.array([], dtype=int)

    # snap to the largest |ECG| sample within 50 ms; mode="same" already
    # centres the filters so no lag correction is needed
    half = max(1, int(round(0.050 * fs)))
    refined = []
    for p in peaks:
        a0, b0 = max(0, p - half), min(x.size, p + half + 1)
        if b0 > a0:
            refined.append(a0 + int(np.argmax(np.abs(x[a0:b0]))))

    refined = np.unique(refined)
    if refined.size == 0:
        return refined
    keep = np.insert(np.diff(refined) > refractory // 2, 0, True)
    return refined[keep]


def segment_beats(x, rpeaks, pre=None, post=None, normalise=True):
    pre = C.PRE_R if pre is None else pre
    post = C.POST_R if post is None else post
    L = pre + 1 + post

    x = np.asarray(x, dtype=float)
    ok = [int(r) for r in rpeaks if r - pre >= 0 and r + post + 1 <= x.size]
    if not ok:
        return np.empty((0, L), dtype=np.float32)

    beats = np.stack([x[r - pre: r + post + 1] for r in ok])
    beats = beats[np.isfinite(beats).all(axis=1)]
    if beats.size == 0:
        return np.empty((0, L), dtype=np.float32)

    if normalise:
        mu = beats.mean(axis=1, keepdims=True)
        sd = beats.std(axis=1, keepdims=True)
        keep = sd.ravel() > 1e-8
        beats, mu, sd = beats[keep], mu[keep], sd[keep]
        if beats.size == 0:
            return np.empty((0, L), dtype=np.float32)
        beats = (beats - mu) / sd

    return beats.astype(np.float32)


def clip_artifacts(x, k=8.0):
    # only used for peak detection: a saturated spike would otherwise push the
    # Pan-Tompkins threshold too high and hide real beats
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    if sigma <= 0:
        return x
    return np.clip(x, med - k * sigma, med + k * sigma)


def record_to_beats(signal, fs_in):
    y = wavelet_denoise(signal)
    y = resample_to(y, fs_in, C.TARGET_FS)
    r = pan_tompkins(clip_artifacts(y), C.TARGET_FS)
    return segment_beats(y, r), y, r
