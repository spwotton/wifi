from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.signal import get_window


@dataclass
class BandEnergy:
    band: Tuple[float, float]
    power_db: float


def _psd_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(2 ** np.ceil(np.log2(len(x))))
    w = get_window("hann", len(x), fftbins=True)
    xw = x * w
    X = np.fft.rfft(xw, n=n)
    psd = (np.abs(X) ** 2) / (np.sum(w ** 2))
    f = np.fft.rfftfreq(n, d=1.0 / sr)
    psd_db = 10 * np.log10(psd + 1e-12)
    return f, psd_db


def band_power_db(x: np.ndarray, sr: int, bands: Iterable[tuple[float, float]]) -> list[BandEnergy]:
    f, p = _psd_db(x, sr)
    out: list[BandEnergy] = []
    for b in bands:
        lo, hi = b
        m = (f >= lo) & (f < hi)
        if not np.any(m):
            out.append(BandEnergy(band=b, power_db=float("-inf")))
        else:
            out.append(BandEnergy(band=b, power_db=float(np.mean(p[m]))))
    return out


def narrowband_peaks(x: np.ndarray, sr: int, min_freq: float = 18000.0, prominence_db: float = 8.0) -> list[tuple[float, float]]:
    f, p = _psd_db(x, sr)
    m = f >= min_freq
    f2, p2 = f[m], p[m]
    if len(p2) < 5:
        return []
    # Simple local maxima with prominence threshold
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(p2) - 1):
        if p2[i] > p2[i - 1] and p2[i] > p2[i + 1]:
            left_min = np.min(p2[max(0, i - 20):i]) if i > 0 else p2[i]
            right_min = np.min(p2[i + 1:i + 21]) if i + 1 < len(p2) else p2[i]
            prom = p2[i] - max(left_min, right_min)
            if prom >= prominence_db:
                peaks.append((float(f2[i]), float(p2[i])))
    return peaks


def estimate_snr_db(x: np.ndarray) -> float:
    # Median absolute deviation as noise proxy
    x = x - np.mean(x)
    noise = np.median(np.abs(x)) + 1e-9
    peak = np.max(np.abs(x)) + 1e-9
    return 20 * np.log10(peak / noise)
