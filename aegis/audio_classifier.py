from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

import os


@dataclass
class AudioClassifyConfig:
    model: str = "superb/wav2vec2-base-superb-ks"  # keyword spotting as a small default
    top_k: int = 5
    backend: str = "auto"  # auto|hf|local
    timeout: int = 30
    retries: int = 2


def _env(key: str) -> Optional[str]:
    try:
        load_dotenv(dotenv_path=".env.local", override=False)
    except Exception:
        pass
    try:
        load_dotenv(override=False)
    except Exception:
        pass
    return os.environ.get(key)


def _hf_client():
    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except Exception:
        return None
    token = _env("HF_TOKEN") or _env("HF_API_TOKEN")
    if not token:
        return None
    try:
        return InferenceClient(provider="hf-inference", api_key=token)
    except Exception:
        return None


def _hf_audio_classify(wav_path: Path, model: str, top_k: int = 5, timeout: int = 30, retries: int = 2) -> List[Dict]:
    # Prefer huggingface_hub InferenceClient if available
    client = _hf_client()
    if client is not None:
        try:
            audio_bytes = wav_path.read_bytes()
            data = client.audio_classification(audio_bytes, model=model)
            preds = data if isinstance(data, list) else [data]
            preds = sorted(preds, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            return preds[:top_k]
        except Exception:
            pass
    token = _env("HF_TOKEN") or _env("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HF_API_TOKEN not set")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    audio_bytes = wav_path.read_bytes()
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, data=audio_bytes, timeout=timeout)
            if r.status_code == 503:
                time.sleep(min(2 ** i, 5))
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"HF error: {data['error']}")
            if not isinstance(data, list):
                if isinstance(data, dict) and "scores" in data:
                    data = data["scores"]
                else:
                    return [{"label": "raw", "score": 0.0, "raw": json.dumps(data)[:1000]}]
            preds = sorted(
                [p for p in data if isinstance(p, dict) and "label" in p],
                key=lambda x: float(x.get("score", 0.0)),
                reverse=True,
            )
            return preds[:top_k]
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** i, 5))
    raise RuntimeError(f"HF inference failed: {last_err}")


def _local_audio_classify(wav_path: Path, model: str, top_k: int = 5) -> List[Dict]:
    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers not installed for local backend") from e
    pipe = pipeline(task="audio-classification", model=model)
    result = pipe(str(wav_path))
    # transformers returns list of dicts with 'label' and 'score'
    if isinstance(result, list):
        preds = sorted(result, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return preds[:top_k]
    return [result]  # type: ignore


def classify_audio_file(wav_path: Path, cfg: AudioClassifyConfig = AudioClassifyConfig()) -> List[Dict]:
    if not wav_path.exists():
        raise FileNotFoundError(str(wav_path))
    backend = cfg.backend.lower()
    if backend == "hf":
        return _hf_audio_classify(wav_path, cfg.model, top_k=cfg.top_k, timeout=cfg.timeout, retries=cfg.retries)
    if backend == "local":
        return _local_audio_classify(wav_path, cfg.model, top_k=cfg.top_k)
    # auto: prefer HF if token set, else local if transformers available, else error
    # Accept either HF_API_TOKEN or HF_TOKEN for convenience
    if _env("HF_API_TOKEN") or _env("HF_TOKEN"):
        return _hf_audio_classify(wav_path, cfg.model, top_k=cfg.top_k, timeout=cfg.timeout, retries=cfg.retries)
    try:
        return _local_audio_classify(wav_path, cfg.model, top_k=cfg.top_k)
    except Exception:
        raise RuntimeError("No backend available: set HF_API_TOKEN for remote or install transformers for local")
