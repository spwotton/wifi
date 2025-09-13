from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample_poly


@dataclass
class AMParams:
    band: Tuple[float, float] = (38000.0, 42000.0)
    lpf_hz: float = 8000.0
    out_rate: int = 16000


def _butter_bandpass(lo: float, hi: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return b, a


def _butter_lpf(cut: float, fs: int, order: int = 5):
    nyq = 0.5 * fs
    b, a = butter(order, cut / nyq, btype="low")
    return b, a


def am_demod(x: np.ndarray, sr: int, params: AMParams = AMParams()) -> tuple[np.ndarray, int]:
    lo, hi = params.band
    if hi >= 0.45 * sr:
        raise ValueError("Carrier band too close to Nyquist for given sample rate")
    b, a = _butter_bandpass(lo, hi, sr, order=4)
    xb = filtfilt(b, a, x)
    env = np.abs(hilbert(xb))
    b2, a2 = _butter_lpf(params.lpf_hz, sr, order=5)
    base = filtfilt(b2, a2, env)
    # DC removal
    base = base - np.mean(base)
    # Normalize to -1..1 guard
    if np.max(np.abs(base)) > 1e-6:
        base = base / (np.max(np.abs(base)) + 1e-9)
    # Resample
    if sr != params.out_rate:
        # Use rational approximation
        from math import gcd
        g = gcd(sr, params.out_rate)
        up = params.out_rate // g
        down = sr // g
        y = resample_poly(base, up, down)
        return y.astype(np.float32), params.out_rate
    return base.astype(np.float32), sr
