from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class UltraConfig:
    # Use workspace-local paths by default for portability
    out_root: Path = Path("data") / "ultra"
    audio_log_dir: Path = out_root / "captures"
    evidence_dir: Path = out_root / "evidence"
    manifest_dir: Path = out_root / "manifests"
    chunk_sec: float = 5.0
    channels: int = 1
    bit_depth: int = 16
    preferred_rates: tuple[int, ...] = (192000, 96000, 48000)
    near_ultra_band: tuple[float, float] = (18000.0, 22000.0)
    mid_ultra_band: tuple[float, float] = (22000.0, 30000.0)
    high_ultra_band: tuple[float, float] = (30000.0, 50000.0)
    proxy_snr_db: float = 12.0
    ultra_snr_db: float = 10.0
    peak_persist_frames: int = 3
    corr_tolerance: float = 2.0


def ensure_dirs(cfg: UltraConfig) -> None:
    for d in (cfg.audio_log_dir, cfg.evidence_dir, cfg.manifest_dir):
        d.mkdir(parents=True, exist_ok=True)
