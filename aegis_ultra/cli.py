from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import UltraConfig, ensure_dirs
from .audio_capture import probe_devices, stream_to_chunks
from .spectral import band_power_db, narrowband_peaks, estimate_snr_db
from .demod import am_demod, AMParams
from .manifest import ManifestEntry, append_manifest, file_sha256, verify_manifest


def cmd_probe():
    cfg = UltraConfig()
    dev = probe_devices(cfg)
    if dev is None:
        print("No input devices available")
        return 1
    print(f"Device: {dev.name} @ {dev.samplerate} Hz, {dev.channels}ch")
    return 0


def cmd_stream():
    cfg = UltraConfig()
    ensure_dirs(cfg)
    print(f"Writing chunks to {cfg.audio_log_dir} (Ctrl+C to stop)")
    try:
        stream_to_chunks(cfg)
    except KeyboardInterrupt:
        pass
    return 0


def _load_wav(p: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(p), always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if np.max(np.abs(x)) > 2:
        # likely int16
        x = x / 32768.0
    return x, int(sr)


def cmd_analyze(wav_path: str):
    cfg = UltraConfig()
    p = Path(wav_path)
    if not p.exists():
        print(f"No such file: {p}")
        return 2
    x, sr = _load_wav(p)
    bands = [cfg.near_ultra_band, cfg.mid_ultra_band, cfg.high_ultra_band]
    be = band_power_db(x, sr, bands)
    peaks = narrowband_peaks(x, sr, min_freq=cfg.near_ultra_band[0])
    snr = estimate_snr_db(x)
    print("Bands (dB):", {str(b.band): round(b.power_db, 1) for b in be})
    if peaks:
        print("Peaks:", [(round(f, 1), round(p, 1)) for f, p in peaks[:10]])
    print(f"SNR: {snr:.1f} dB @ {sr} Hz")
    # Demod if possible
    try:
        y, yr = am_demod(x, sr, AMParams())
        out = p.with_name(p.stem + "_demod.wav")
        import soundfile as sf

        sf.write(str(out), y, yr, subtype="PCM_16")
        print(f"Demod written: {out}")
    except Exception as e:
        print(f"Demod skipped: {e}")
    return 0


def cmd_manifest_add(path: str):
    cfg = UltraConfig()
    ensure_dirs(cfg)
    p = Path(path)
    sha = file_sha256(p)
    mpath = cfg.manifest_dir / "ultra_manifest.jsonl"
    entry = ManifestEntry(ts=time.time(), kind="audio", path=str(p), sha256=sha, meta={})
    h = append_manifest(mpath, entry)
    print(f"Manifest updated: {mpath} (hash {h[:8]}…)")
    return 0


def cmd_manifest_verify():
    cfg = UltraConfig()
    mpath = cfg.manifest_dir / "ultra_manifest.jsonl"
    if not mpath.exists():
        print("No manifest yet")
        return 1
    ok = verify_manifest(mpath)
    print("Manifest OK" if ok else "Manifest CORRUPT")
    return 0 if ok else 3


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="aegis-ultra")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe")
    sub.add_parser("stream")
    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("wav")
    ap_ma = sub.add_parser("manifest-add")
    ap_ma.add_argument("path")
    sub.add_parser("manifest-verify")

    ns = ap.parse_args(argv)
    if ns.cmd == "probe":
        return cmd_probe()
    if ns.cmd == "stream":
        return cmd_stream()
    if ns.cmd == "analyze":
        return cmd_analyze(ns.wav)
    if ns.cmd == "manifest-add":
        return cmd_manifest_add(ns.path)
    if ns.cmd == "manifest-verify":
        return cmd_manifest_verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
