from __future__ import annotations

import contextlib
import dataclasses
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict
from collections import deque

import numpy as np
from scipy.signal import spectrogram, welch, chirp, butter, filtfilt, hilbert, resample_poly
import soundfile as sf


try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclasses.dataclass
class DetectionEvent:
    timestamp: float
    samplerate: int
    prf_hz: float
    band_low_hz: float
    band_high_hz: float
    snr_db: float
    method: str
    notes: str = ""


def _init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sonar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                samplerate INTEGER NOT NULL,
                prf_hz REAL,
                band_low_hz REAL,
                band_high_hz REAL,
                snr_db REAL,
                method TEXT,
                notes TEXT
            );
            """
        )
    return conn


def list_audio_devices():
    if sd is None:
        print("sounddevice not available. Install with `pip install sounddevice`.", file=sys.stderr)
        return
    print("Available audio devices:")
    print(sd.query_devices())


def _capture_audio(duration_sec: float, samplerate: int, channels: int, device: Optional[str]) -> tuple[np.ndarray, int]:
    if sd is None:
        raise RuntimeError("sounddevice not available; cannot capture audio.")
    # Resolve device: accept index, exact name, or case-insensitive substring (e.g., 'audiobox')
    def _resolve_input_device(dev: Optional[str | int]):
        if dev is None:
            return None
        # Already numeric index
        if isinstance(dev, int):
            return dev
        s = str(dev).strip()
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
        # Search devices for input-capable names containing the substring
        try:
            devices = sd.query_devices()
            s_lower = s.lower()
            matches: list[tuple[int, dict]] = []
            for idx, info in enumerate(devices):
                try:
                    name = str(info.get("name", ""))
                    max_in = int(info.get("max_input_channels", 0))
                except Exception:
                    name = str(info)
                    max_in = 0
                if max_in > 0 and s_lower in name.lower():
                    matches.append((idx, info))
            if matches:
                # Prefer WASAPI/DirectSound over MME by simple heuristic in name
                def pref(m):
                    n = str(m[1].get("name", "")).lower()
                    score = 0
                    if "wasapi" in n:
                        score += 3
                    if "directsound" in n:
                        score += 2
                    if "mme" in n:
                        score += 1
                    return -score

                matches.sort(key=pref)
                return matches[0][0]
        except Exception:
            pass
        return None

    dev_arg = _resolve_input_device(device)
    try:
        sd.check_input_settings(device=dev_arg, samplerate=samplerate, channels=channels)
    except Exception as e:
        print(f"Warning: requested settings not supported: {e}. Falling back to default device/rate.")
        dev_arg = None
        samplerate = int(sd.query_devices(kind='input')['default_samplerate'])
        channels = 1

    q: queue.Queue[np.ndarray] = queue.Queue()
    frames = int(duration_sec * samplerate)

    def cb(indata, frames_, time_, status):  # noqa: ANN001
        if status:
            # Avoid raising; log and continue.
            print(f"Audio status: {status}")
        q.put(indata.copy())

    blocksize = 0  # let driver choose
    dtype = 'float32'
    collected = []
    with sd.InputStream(
        device=dev_arg,
        channels=channels,
        samplerate=samplerate,
        callback=cb,
        blocksize=blocksize,
        dtype=dtype,
    ):
        needed = frames
        while needed > 0:
            try:
                chunk = q.get(timeout=1.0)
            except queue.Empty:
                continue
            collected.append(chunk)
            needed -= len(chunk)
    data = np.concatenate(collected, axis=0)
    # Mixdown to mono
    if data.ndim == 2 and data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data.reshape(-1)
    return data, int(samplerate)


def _bandlimit(x: np.ndarray, fs: int, band: Optional[Tuple[float, float]]):
    if not band:
        return x
    lo, hi = band
    # FFT bandpass (lightweight, avoids scipy.filter for now)
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    mask = (freqs >= lo) & (freqs <= hi)
    X[~mask] = 0
    y = np.fft.irfft(X, n=n)
    return y


def _estimate_snr_db(x: np.ndarray) -> float:
    # Simple SNR estimate via robust stats
    p_total = np.mean(x**2) + 1e-12
    # Noise floor ~ median absolute deviation
    mad = np.median(np.abs(x - np.median(x))) + 1e-12
    p_noise = (mad * 1.4826) ** 2 + 1e-12
    snr = 10 * np.log10(p_total / p_noise)
    return float(snr)


def _detect_bursty_prf(x: np.ndarray, fs: int) -> Optional[float]:
    # Envelope via magnitude of STFT or rectified band-limited signal
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=1024, noverlap=768, scaling='spectrum', mode='magnitude')
    env = Sxx.mean(axis=0)  # broadband energy over time
    env = (env - np.mean(env)) / (np.std(env) + 1e-12)
    # Autocorrelation of envelope to find periodicity
    ac = np.correlate(env, env, mode='full')
    ac = ac[ac.size // 2 :]
    # Ignore lag 0, search within reasonable PRF range 1–60 Hz
    lags = np.arange(len(ac))
    lag_times = lags * (t[1] - t[0])
    mask = (lag_times >= 1.0 / 60.0) & (lag_times <= 1.0)
    if not np.any(mask):
        return None
    idx = np.argmax(ac[mask])
    lag = lag_times[mask][idx]
    if lag <= 0:
        return None
    prf = 1.0 / lag
    return float(prf)


def _dominant_frequency_hz(x: np.ndarray, fs: int) -> float:
    # Power spectrum via rFFT
    n = len(x)
    if n == 0:
        return 0.0
    X = np.fft.rfft(x)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    idx = int(np.argmax(P)) if P.size else 0
    return float(freqs[idx] if idx < len(freqs) else 0.0)


def _ultrasonic_energy_ratio(x: np.ndarray, fs: int, threshold_hz: float = 18000.0) -> float:
    # Fraction of spectral energy above threshold_hz
    n = len(x)
    if n == 0:
        return 0.0
    X = np.fft.rfft(x)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    total = float(np.sum(P) + 1e-12)
    mask = freqs >= min(threshold_hz, fs / 2.0)
    ultra = float(np.sum(P[mask]))
    return float(ultra / total)


def _am_demodulate_to_audio(
    x: np.ndarray,
    fs: int,
    carrier_band: tuple[float, float] = (30000.0, 50000.0),
    lpf_hz: float = 8000.0,
    out_sr: int = 16000,
) -> tuple[int, np.ndarray] | None:
    """AM demodulation for ultrasonic carriers (e.g., ~40 kHz) into audible baseband audio.

    Steps:
    - Band-limit around the carrier (e.g., 30–50 kHz)
    - Envelope detect via analytic signal magnitude (Hilbert)
    - Low-pass filter the envelope to audio band
    - Resample to out_sr for saving
    Returns (sr, audio) or None if fs insufficient.
    """
    lo, hi = carrier_band
    if fs < 2 * hi:
        # Insufficient sample rate to cover carrier band by Nyquist
        return None
    # Bandpass around carrier using FFT masking
    x_bp = _bandlimit(x, fs, (lo, hi))
    # Envelope via Hilbert magnitude
    env = np.abs(hilbert(x_bp))
    # Remove DC offset
    env = env - np.mean(env)
    # Low-pass to audio band
    nyq = fs * 0.5
    wc = min(max(lpf_hz / nyq, 1e-4), 0.99)
    b, a = butter(4, wc, btype='low')
    base = filtfilt(b, a, env).astype(np.float32)
    # Normalize to avoid clipping
    peak = float(np.max(np.abs(base)) + 1e-12)
    if peak > 0:
        base = base / peak * 0.9
    # Resample if needed
    if fs != out_sr:
        # Use rational resample
        from math import gcd

        g = gcd(int(fs), int(out_sr))
        up = int(out_sr // g)
        down = int(fs // g)
        base = resample_poly(base, up, down).astype(np.float32)
        fs_out = out_sr
    else:
        fs_out = fs
    return int(fs_out), base


def _matched_chirp_score(x: np.ndarray, fs: int, f0: float, f1: float, dur_s: float) -> float:
    n = int(dur_s * fs)
    if n <= 8:
        return 0.0
    t = np.arange(n) / fs
    tpl = chirp(t, f0=f0, f1=f1, t1=dur_s, method='linear')
    # Normalize
    tpl = tpl / (np.linalg.norm(tpl) + 1e-12)
    x_norm = x / (np.linalg.norm(x) + 1e-12)
    # Correlate via FFT for speed
    N = int(2 ** np.ceil(np.log2(len(x_norm) + len(tpl) - 1)))
    X = np.fft.rfft(x_norm, N)
    T = np.fft.rfft(tpl, N)
    corr = np.fft.irfft(X * np.conj(T), N)
    peak = float(np.max(np.abs(corr)))
    return peak


def run_sonar_scan(
    db_path: Path,
    duration_sec: float = 10.0,
    samplerate: int = 48000,
    channels: int = 1,
    device: Optional[str] = None,
    band_hz: Optional[Tuple[float, float]] = None,
    plot: bool = False,
    chirp: Optional[str] = None,
    ultra_threshold_hz: float = 18000.0,
    save_audio: bool = False,
    demod_out: Optional[Path] = None,
    carrier_band: Optional[Tuple[float, float]] = None,
):
    conn = _init_db(db_path)
    # Capture audio (may adjust samplerate if device cannot support request)
    data, samplerate = _capture_audio(duration_sec, samplerate, channels, device)
    print(f"Captured audio: {duration_sec}s | requested≈{int(duration_sec)}s@req | using {samplerate} Hz, device={device}")
    if band_hz is not None:
        print(f"Applying band limit: {band_hz[0]:.0f}–{band_hz[1]:.0f} Hz")
        data = _bandlimit(data, samplerate, band_hz)

    snr_db = _estimate_snr_db(data)
    prf = _detect_bursty_prf(data, samplerate)
    dom = _dominant_frequency_hz(data, samplerate)
    ultra_ratio = _ultrasonic_energy_ratio(data, samplerate, threshold_hz=ultra_threshold_hz)
    match_score = 0.0
    if chirp:
        try:
            f0_s, f1_s, dur_ms_s = chirp.split(":")
            f0, f1, dur_ms = float(f0_s), float(f1_s), float(dur_ms_s)
            match_score = _matched_chirp_score(data, samplerate, f0, f1, dur_ms / 1000.0)
        except Exception as e:
            print(f"Invalid --chirp spec: {chirp} ({e})")

    if plot and plt is not None:
        f, t, Sxx = spectrogram(data, fs=samplerate, nperseg=1024, noverlap=768, scaling='spectrum')
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-15), shading='gouraud')
        plt.title('Spectrogram (dB)')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.show()
    elif plot and plt is None:
        print("matplotlib not available; install it or run without --plot")

    method = "burst_PRF_autocorr"
    notes = "periodic energy bursts detected" if prf else "no clear periodic bursts"
    if chirp:
        method += "+matched_chirp"
        notes += f" | chirp_match≈{match_score:.2f}"
    notes += f" | dom≈{dom:.0f}Hz ultra≈{ultra_ratio:.3f}"

    # Optional save captured audio for evidence chain
    if save_audio:
        out_dir = db_path.parent / "captures"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_label = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        wav_path = out_dir / f"audio_{ts_label}.wav"
        sf.write(str(wav_path), data.astype(np.float32), samplerate)
        # Compute SHA-256 of the WAV file
        import hashlib

        h = hashlib.sha256()
        with open(wav_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        notes += f" | file={wav_path.name} sha256={h.hexdigest()[:16]}…"

    # Optional AM demodulation if high-rate capture and carrier band provided
    if demod_out is not None and carrier_band is not None:
        try:
            dem = _am_demodulate_to_audio(data, samplerate, carrier_band=carrier_band)
            if dem is None:
                print("Demodulation skipped: insufficient sample rate for carrier band.")
            else:
                dem_sr, dem_audio = dem
                demod_out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(demod_out), dem_audio, dem_sr)
                notes += f" | demod={demod_out.name}@{dem_sr}"
                print(f"Wrote demodulated audio: {demod_out} ({dem_sr} Hz)")
        except Exception as e:
            print(f"Demodulation failed: {e}")

    evt = DetectionEvent(
        timestamp=time.time(),
        samplerate=samplerate,
        prf_hz=prf or 0.0,
        band_low_hz=(band_hz[0] if band_hz else 0.0),
        band_high_hz=(band_hz[1] if band_hz else float(samplerate) / 2.0),
        snr_db=snr_db,
        method=method,
        notes=notes,
    )
    with conn:
        conn.execute(
            "INSERT INTO sonar_events(timestamp, samplerate, prf_hz, band_low_hz, band_high_hz, snr_db, method, notes) VALUES (?,?,?,?,?,?,?,?)",
            (
                evt.timestamp,
                evt.samplerate,
                evt.prf_hz,
                evt.band_low_hz,
                evt.band_high_hz,
                evt.snr_db,
                evt.method,
                evt.notes,
            ),
        )
    print(
        f"Logged event: PRF={evt.prf_hz:.2f} Hz | SNR≈{evt.snr_db:.1f} dB | Band={evt.band_low_hz:.0f}-{evt.band_high_hz:.0f} Hz"
    )


def export_events_csv(db_path: Path, out_csv: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, timestamp, samplerate, prf_hz, band_low_hz, band_high_hz, snr_db, method, notes FROM sonar_events ORDER BY id ASC"
    )
    rows = cur.fetchall()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "timestamp", "samplerate", "prf_hz", "band_low_hz", "band_high_hz", "snr_db", "method", "notes"])
        w.writerows(rows)


def _load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    data, sr = sf.read(str(path), always_2d=False)
    x = data.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    # Normalize if integer encoded
    maxabs = np.max(np.abs(x)) + 1e-12
    if maxabs > 1.0:
        x = x / maxabs
    return int(sr), x


def _spectral_subtraction(Sxx: np.ndarray, percentile: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    # Estimate noise floor per frequency bin as low percentile across time
    noise = np.percentile(Sxx, q=np.clip(percentile, 0, 100), axis=1)
    S_denoised = np.clip(Sxx - noise[:, None], a_min=0.0, a_max=None)
    return S_denoised, noise


def _avg_psd_from_sxx(Sxx: np.ndarray) -> np.ndarray:
    return np.mean(Sxx, axis=1)


def analyze_wav_file(
    wav_path: Path,
    db_path: Path,
    band_hz: Optional[Tuple[float, float]] = None,
    plot: bool = False,
    chirp: Optional[str] = None,
    denoise: str = "specsub",
    noise_percentile: float = 20.0,
    psd_out: Optional[Path] = None,
    demod_out: Optional[Path] = None,
    carrier_band: Optional[Tuple[float, float]] = None,
):
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    fs, x = _load_wav_mono(wav_path)
    x_orig = x.copy()
    if band_hz is not None:
        x = _bandlimit(x, fs, band_hz)
    # Spectrogram for detection and denoising
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=2048, noverlap=1536, scaling='spectrum', mode='magnitude')
    if band_hz is not None:
        mask = (f >= band_hz[0]) & (f <= band_hz[1])
        f = f[mask]
        Sxx = Sxx[mask, :]

    S_proc = Sxx
    noise_floor = np.zeros_like(f)
    method = "wav"
    if denoise == "specsub":
        S_proc, noise_floor = _spectral_subtraction(Sxx, percentile=noise_percentile)
        method += "+specsub"
    else:
        method += "+none"

    # Envelope and PRF on denoised energy
    env = S_proc.mean(axis=0)
    env = (env - np.mean(env)) / (np.std(env) + 1e-12)
    ac = np.correlate(env, env, mode='full')
    ac = ac[ac.size // 2 :]
    lags = np.arange(len(ac))
    lag_times = lags * (t[1] - t[0])
    mask_lag = (lag_times >= 1.0 / 60.0) & (lag_times <= 1.0)
    prf = None
    if np.any(mask_lag):
        idx = np.argmax(ac[mask_lag])
        lag = lag_times[mask_lag][idx]
        if lag > 0:
            prf = 1.0 / lag

    # PSD profile from processed spectrogram
    avg_psd = _avg_psd_from_sxx(S_proc)

    # Features
    power = np.sum(avg_psd) + 1e-12
    centroid = float(np.sum(f * avg_psd) / power)
    bandwidth = float(np.sqrt(np.sum(((f - centroid) ** 2) * avg_psd) / power))
    flatness = float(np.exp(np.mean(np.log(avg_psd + 1e-15))) / (np.mean(avg_psd) + 1e-15))
    snr_db = _estimate_snr_db(x)
    dom = _dominant_frequency_hz(x, fs)
    ultra_ratio = _ultrasonic_energy_ratio(x, fs, threshold_hz=18000.0)

    match_score = 0.0
    if chirp:
        try:
            f0_s, f1_s, dur_ms_s = chirp.split(":")
            f0, f1, dur_ms = float(f0_s), float(f1_s), float(dur_ms_s)
            x_bl = _bandlimit(x_orig, fs, (min(f0, f1) * 0.8, max(f0, f1) * 1.2))
            match_score = _matched_chirp_score(x_bl, fs, f0, f1, dur_ms / 1000.0)
            method += "+matched_chirp"
        except Exception as e:
            print(f"Invalid --chirp spec: {chirp} ({e})")

    # Log summary event
    conn = _init_db(db_path)
    with conn:
        conn.execute(
            "INSERT INTO sonar_events(timestamp, samplerate, prf_hz, band_low_hz, band_high_hz, snr_db, method, notes) VALUES (?,?,?,?,?,?,?,?)",
            (
                time.time(),
                fs,
                float(prf or 0.0),
                float(band_hz[0] if band_hz else 0.0),
                float(band_hz[1] if band_hz else fs / 2.0),
                float(snr_db),
                method,
                f"file={wav_path.name} centroid={centroid:.1f}Hz bw={bandwidth:.1f}Hz flat={flatness:.3f} dom={dom:.0f}Hz ultra={ultra_ratio:.3f} match={match_score:.2f}",
            ),
        )

    # Optional plotting
    if plot and plt is not None:
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, 10 * np.log10(S_proc + 1e-15), shading='gouraud')
        plt.title('Spectrogram (dB) - processed')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.tight_layout()
        plt.show()
    elif plot and plt is None:
        print("matplotlib not available; install it or run without --plot")

    # Optional PSD profile CSV
    if psd_out is not None:
        psd_out.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with open(psd_out, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["frequency_hz", "avg_power"])
            for fi, pi in zip(f, avg_psd):
                w.writerow([float(fi), float(pi)])
        print(f"Wrote PSD profile: {psd_out}")

    print(
        f"Analyzed {wav_path.name}: PRF={float(prf or 0.0):.2f} Hz | centroid≈{centroid:.1f} Hz | bw≈{bandwidth:.1f} Hz | flatness≈{flatness:.3f} | dom≈{dom:.0f} Hz | ultra≈{ultra_ratio:.3f}"
    )

    # Optional demodulation to baseband audio
    if demod_out is not None and carrier_band is not None:
        try:
            dem = _am_demodulate_to_audio(x_orig, fs, carrier_band=carrier_band)
            if dem is None:
                print("Demodulation skipped: WAV sample rate insufficient for carrier band.")
            else:
                dem_sr, dem_audio = dem
                demod_out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(demod_out), dem_audio, dem_sr)
                print(f"Wrote demodulated audio: {demod_out} ({dem_sr} Hz)")
        except Exception as e:
            print(f"Demodulation failed: {e}")


def watch_sonar(
    db_path: Path,
    samplerate: int = 48000,
    device: Optional[str] = None,
    band_hz: Optional[Tuple[float, float]] = None,
    window_sec: float = 3.0,
    check_interval_sec: float = 1.0,
    snr_min_db: float = 3.0,
    ultra_threshold_hz: float = 18000.0,
    ultra_ratio_min: float = 0.03,
    chirp: Optional[str] = None,
    prf_min: float = 1.0,
    prf_max: float = 60.0,
    cooldown_sec: float = 30.0,
    on_trigger: Optional[Callable[[Dict[str, float]], None]] = None,
):
    """Continuously monitor audio and log events when thresholds are exceeded.

    On trigger, inserts an event into sonar_events and optionally calls on_trigger with metrics.
    """
    if sd is None:
        raise RuntimeError("sounddevice not available; cannot watch audio.")
    conn = _init_db(db_path)
    buf = deque()  # list of np.ndarray blocks
    blocksize = 0
    dtype = 'float32'
    last_trigger_ts = 0.0
    frames_per_check = int(check_interval_sec * samplerate)
    target_len = int(window_sec * samplerate)

    # resolve device like _capture_audio does
    dev_arg = device
    try:
        sd.check_input_settings(device=dev_arg, samplerate=samplerate, channels=1)
    except Exception:
        dev_arg = None
        samplerate = int(sd.query_devices(kind='input')['default_samplerate'])

    q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames_, time_, status):  # noqa: ANN001
        if status:
            print(f"Audio status: {status}")
        q.put(indata.copy())

    print(f"Watching audio @ {samplerate} Hz; window={window_sec}s check={check_interval_sec}s thresholds: SNR>={snr_min_db}dB, ultra>={ultra_ratio_min}")
    with sd.InputStream(device=dev_arg, channels=1, samplerate=samplerate, callback=cb, blocksize=blocksize, dtype=dtype):
        pending = 0
        while True:
            try:
                chunk = q.get(timeout=1.0)
            except queue.Empty:
                continue
            buf.append(chunk.reshape(-1))
            pending += len(chunk)
            # Trim buffer to target_len * 2 (some slack)
            cur = np.concatenate(list(buf), axis=0) if buf else np.zeros(0, dtype=np.float32)
            if len(cur) > target_len * 2:
                # drop older samples
                drop = len(cur) - target_len * 2
                # efficiently drop from left by re-building deque
                acc = 0
                newbuf = deque()
                for b in buf:
                    if acc + len(b) <= drop:
                        acc += len(b)
                        continue
                    if drop > acc:
                        b = b[(drop - acc):]
                    newbuf.append(b)
                buf = newbuf
                cur = np.concatenate(list(buf), axis=0)

            if pending < frames_per_check:
                continue
            pending = 0
            if len(cur) < target_len:
                continue
            x = cur[-target_len:]
            if band_hz is not None:
                x = _bandlimit(x, samplerate, band_hz)
            snr_db = _estimate_snr_db(x)
            ultra_ratio = _ultrasonic_energy_ratio(x, samplerate, threshold_hz=ultra_threshold_hz)
            prf = _detect_bursty_prf(x, samplerate)
            dom = _dominant_frequency_hz(x, samplerate)
            match_score = 0.0
            if chirp:
                try:
                    f0_s, f1_s, dur_ms_s = chirp.split(":")
                    f0, f1, dur_ms = float(f0_s), float(f1_s), float(dur_ms_s)
                    match_score = _matched_chirp_score(x, samplerate, f0, f1, dur_ms / 1000.0)
                except Exception:
                    pass
            prf_ok = prf is not None and prf_min <= prf <= prf_max
            cond = (snr_db >= snr_min_db) or (ultra_ratio >= ultra_ratio_min) or prf_ok or (match_score >= 0.5)
            now_ts = time.time()
            if cond and (now_ts - last_trigger_ts >= cooldown_sec):
                last_trigger_ts = now_ts
                # log event
                method = "watch"
                if chirp:
                    method += "+matched_chirp"
                notes = f"dom≈{dom:.0f}Hz ultra≈{ultra_ratio:.3f}"
                if prf:
                    notes = f"PRF={prf:.2f}Hz | " + notes
                with conn:
                    conn.execute(
                        "INSERT INTO sonar_events(timestamp, samplerate, prf_hz, band_low_hz, band_high_hz, snr_db, method, notes) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            now_ts,
                            samplerate,
                            float(prf or 0.0),
                            float(band_hz[0] if band_hz else 0.0),
                            float(band_hz[1] if band_hz else samplerate / 2.0),
                            float(snr_db),
                            method,
                            notes,
                        ),
                    )
                info = {
                    "timestamp": now_ts,
                    "snr_db": float(snr_db),
                    "ultra_ratio": float(ultra_ratio),
                    "prf": float(prf or 0.0),
                    "dom": float(dom),
                    "match": float(match_score),
                }
                print(f"[watch] trigger: SNR={snr_db:.1f}dB ultra={ultra_ratio:.3f} prf={float(prf or 0.0):.2f} dom={dom:.0f}Hz")
                if on_trigger:
                    try:
                        on_trigger(info)
                    except Exception as e:
                        print(f"on_trigger failed: {e}")
