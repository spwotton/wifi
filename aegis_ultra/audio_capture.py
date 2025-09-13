from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except Exception as e:  # pragma: no cover - runtime import guard
    sd = None  # type: ignore
    sf = None  # type: ignore

from .config import UltraConfig, ensure_dirs


@dataclass
class DeviceInfo:
    index: int
    name: str
    samplerate: int
    channels: int


def probe_devices(cfg: UltraConfig, prefer_contains: tuple[str, ...] = ("Audiobox", "USB96")) -> Optional[DeviceInfo]:
    if sd is None:
        return None
    devices = sd.query_devices()
    candidates: list[tuple[int, dict]] = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = d.get("name", "")
        # Prefer specific names
        score = 0
        lname = name.lower()
        for tag in prefer_contains:
            if tag.lower() in lname:
                score += 10
        # Prefer high rates
        for r in cfg.preferred_rates:
            try:
                sd.check_input_settings(device=idx, samplerate=r, channels=cfg.channels)
                score += 1 if r == 48000 else (2 if r == 96000 else 3)
            except Exception:
                pass
        candidates.append((score, {"index": idx, **d}))
    if not candidates:
        return None
    best = sorted(candidates, key=lambda t: t[0], reverse=True)[0][1]
    # Pick best achievable rate from preferred list
    chosen_rate = None
    for r in cfg.preferred_rates:
        try:
            sd.check_input_settings(device=best["index"], samplerate=r, channels=cfg.channels)
            chosen_rate = r
            break
        except Exception:
            continue
    if chosen_rate is None:
        # fallback to default samplerate if available
        chosen_rate = int(sd.query_devices(best["index"], "input")["default_samplerate"]) or 48000
    return DeviceInfo(index=int(best["index"]), name=best.get("name", "?"), samplerate=int(chosen_rate), channels=cfg.channels)


class ChunkWriter:
    def __init__(self, cfg: UltraConfig, base_dir: Path, samplerate: int, channels: int):
        self.cfg = cfg
        self.base_dir = base_dir
        self.samplerate = samplerate
        self.channels = channels
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        ensure_dirs(cfg)
        base_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        self._thr.start()

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=2)

    def submit(self, frames: np.ndarray):
        try:
            self.q.put_nowait(frames.copy())
        except queue.Full:
            _ = self.q.get_nowait()
            self.q.put_nowait(frames.copy())

    def _run(self):
        if sf is None:
            return
        chunk_len = int(self.samplerate * self.cfg.chunk_sec)
        buf: list[np.ndarray] = []
        total = 0
        cur_start = time.time()
        while not self._stop.is_set():
            try:
                data = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            buf.append(data)
            total += data.shape[0]
            if total >= chunk_len:
                merged = np.concatenate(buf, axis=0)
                frames = merged[:chunk_len]
                buf = [merged[chunk_len:]] if merged.shape[0] > chunk_len else []
                total = sum(x.shape[0] for x in buf)
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(cur_start))
                fname = f"audio_{ts}.wav"
                path = self.base_dir / fname
                sf.write(str(path), frames, self.samplerate, subtype="PCM_16")
                cur_start = time.time()


def stream_to_chunks(cfg: UltraConfig, device: Optional[DeviceInfo] = None):
    if sd is None:
        raise RuntimeError("sounddevice not available; install sounddevice and soundfile")
    ensure_dirs(cfg)
    if device is None:
        device = probe_devices(cfg)
    if device is None:
        raise RuntimeError("No audio input device available")
    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    writer = ChunkWriter(cfg, cfg.audio_log_dir, device.samplerate, device.channels)
    writer.start()

    def callback(indata, frames, time_info, status):  # type: ignore
        if status:
            pass
        q.put(indata.copy())

    with sd.InputStream(
        device=device.index,
        channels=device.channels,
        samplerate=device.samplerate,
        dtype="int16" if cfg.bit_depth == 16 else "float32",
        callback=callback,
    ):
        try:
            while True:
                try:
                    data = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                # Convert to mono float32
                if data.dtype != np.float32:
                    data = data.astype(np.float32) / 32768.0
                if data.ndim == 2 and data.shape[1] > 1:
                    data = data.mean(axis=1, keepdims=False)
                writer.submit(data)
        except KeyboardInterrupt:
            pass
        finally:
            writer.stop()
