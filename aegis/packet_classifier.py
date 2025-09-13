from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

import os


@dataclass
class PacketClassifyConfig:
    model: str = "rdpahalavan/bert-network-packet-flow-header-payload"
    top_k: int = 5
    timeout: int = 30
    retries: int = 2
    backend: str = "auto"  # auto|hf|local
    # tshark options
    tshark_filter: str = ""
    max_packets: int = 200


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


def _hf_text_classify(text: str, model: str, top_k: int = 5, timeout: int = 30, retries: int = 2) -> List[Dict]:
    # Prefer huggingface_hub InferenceClient if available
    client = _hf_client()
    if client is not None:
        try:
            # Some client versions support top_k; if not, slice below
            data = client.text_classification(text, model=model)
            preds = data if isinstance(data, list) else [data]
            preds = sorted(
                [p for p in preds if isinstance(p, dict) and "label" in p],
                key=lambda x: float(x.get("score", 0.0)),
                reverse=True,
            )
            return preds[:top_k]
        except Exception:
            pass
    # Fallback to raw requests API
    token = _env("HF_TOKEN") or _env("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HF_API_TOKEN not set")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"inputs": text}
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 503:
                import time as _t
                _t.sleep(min(2 ** i, 5))
                continue
            r.raise_for_status()
            data = r.json()
            preds = []
            if isinstance(data, list):
                if data and isinstance(data[0], list):
                    preds = data[0]
                else:
                    preds = data
            elif isinstance(data, dict) and "labels" in data and "scores" in data:
                preds = [{"label": l, "score": s} for l, s in zip(data["labels"], data["scores"])]
            else:
                return [{"label": "raw", "score": 0.0, "raw": json.dumps(data)[:1000]}]
            preds = sorted(
                [p for p in preds if isinstance(p, dict) and "label" in p],
                key=lambda x: float(x.get("score", 0.0)),
                reverse=True,
            )
            return preds[:top_k]
        except Exception as e:
            last_err = e
            import time as _t
            _t.sleep(min(2 ** i, 5))
    raise RuntimeError(f"HF inference failed: {last_err}")


def _has_tshark() -> bool:
    return shutil.which("tshark") is not None


def _extract_packets_hex(pcap: Path, tshark_filter: str = "", max_packets: int = 200) -> List[Tuple[str, str]]:
    """Extract hex of headers+payloads using tshark; returns list of (summary, hex_payload)."""
    if not _has_tshark():
        raise RuntimeError("tshark not found in PATH. Install Wireshark/tshark and retry.")
    # Use tshark to output PDML (XML) or json; here we use -T fields with data.data for raw bytes when available
    # Limit packets with -c
    cmd = [
        "tshark",
        "-r",
        str(pcap),
        "-T",
        "fields",
        "-e",
        "frame.number",
        "-e",
        "frame.protocols",
        "-e",
        "data.data",
        "-c",
        str(int(max_packets)),
    ]
    if tshark_filter:
        cmd.extend(["-Y", tshark_filter])
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    lines = out.decode("utf-8", errors="ignore").splitlines()
    results: List[Tuple[str, str]] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        frame_no, protos, hexdata = parts[0], parts[1], parts[2]
        if not hexdata:
            continue
        summary = f"frame={frame_no} protos={protos}"
        results.append((summary, hexdata))
    return results


def _hex_to_tokens(hexdata: str, max_bytes: int = 512) -> str:
    """Convert hex string to a compact textual representation suitable for a BERT-like classifier.

    We prefix with a small header summary and truncate to max_bytes to keep inputs reasonable.
    """
    # Normalize: remove colons if any
    hexdata = hexdata.replace(":", "").strip()
    if len(hexdata) % 2 == 1:
        hexdata = hexdata[:-1]
    # Truncate
    hexdata = hexdata[: max_bytes * 2]
    # Group bytes
    bytes_list = [hexdata[i : i + 2] for i in range(0, len(hexdata), 2)]
    # Space-separated tokens
    return " ".join(bytes_list)


def classify_pcap_packets(
    pcap_path: Path,
    cfg: PacketClassifyConfig = PacketClassifyConfig(),
) -> List[Dict]:
    pairs = _extract_packets_hex(pcap_path, tshark_filter=cfg.tshark_filter, max_packets=int(cfg.max_packets))
    outputs: List[Dict] = []
    for summary, hexdata in pairs:
        tokens = _hex_to_tokens(hexdata)
        text = f"header_payload: {tokens}"
        # Backend selection
        backend = (cfg.backend or "auto").lower()
        preds: List[Dict]
        if backend == "hf":
            preds = _hf_text_classify(text, cfg.model, top_k=cfg.top_k, timeout=cfg.timeout, retries=cfg.retries)
        elif backend == "local":
            preds = _local_text_classify(text, cfg.model, top_k=cfg.top_k)
        else:
            # auto
            if _env("HF_TOKEN") or _env("HF_API_TOKEN") or _hf_client() is not None:
                try:
                    preds = _hf_text_classify(text, cfg.model, top_k=cfg.top_k, timeout=cfg.timeout, retries=cfg.retries)
                except Exception:
                    preds = _local_text_classify(text, cfg.model, top_k=cfg.top_k)
            else:
                preds = _local_text_classify(text, cfg.model, top_k=cfg.top_k)
        outputs.append({"summary": summary, "predictions": preds})
    return outputs


def _local_text_classify(text: str, model: str, top_k: int = 5) -> List[Dict]:
    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers not installed for local packet classification") from e
    pipe = pipeline(task="text-classification", model=model, top_k=top_k)
    result = pipe(text)
    # Pipeline returns list of dicts or list[list[dict]] when top_k>1
    if isinstance(result, list):
        if result and isinstance(result[0], list):
            preds = result[0]
        else:
            preds = result
    else:
        preds = [result]  # type: ignore
    preds = sorted(preds, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return preds[:top_k]
