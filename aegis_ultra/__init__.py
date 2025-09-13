"""AEGIS.UltraDetect - Directional Ultrasonic Speaker Detection

High-level package providing audio capture, spectral analysis, AM demodulation,
event creation, manifest chaining, and cross-source correlation utilities.
"""

from .config import UltraConfig
from .audio_capture import probe_devices, stream_to_chunks
from .spectral import band_power_db, narrowband_peaks
from .demod import am_demod, AMParams
from .manifest import ManifestEntry, append_manifest, verify_manifest

__all__ = [
    "UltraConfig",
    "probe_devices",
    "stream_to_chunks",
    "band_power_db",
    "narrowband_peaks",
    "am_demod",
    "AMParams",
    "ManifestEntry",
    "append_manifest",
    "verify_manifest",
]
