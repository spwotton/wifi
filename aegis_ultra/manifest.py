from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ManifestEntry:
    ts: float
    kind: str
    path: str
    sha256: str
    meta: dict
    prev: Optional[str] = None


def file_sha256(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def append_manifest(manifest_path: Path, entry: ManifestEntry) -> str:
    prev_hash = None
    if manifest_path.exists():
        with manifest_path.open("rb") as f:
            last = None
            for line in f:
                last = line
            if last:
                try:
                    obj = json.loads(last.decode("utf-8"))
                    prev_hash = obj.get("entry_hash")
                except Exception:
                    prev_hash = None
    entry.prev = prev_hash
    payload = asdict(entry)
    enc = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    chain_hash = hashlib.sha256(enc).hexdigest()
    with manifest_path.open("ab") as f:
        line = json.dumps({"entry": payload, "entry_hash": chain_hash}, separators=(",", ":")).encode("utf-8")
        f.write(line + b"\n")
    return chain_hash


def verify_manifest(manifest_path: Path) -> bool:
    prev = None
    with manifest_path.open("rb") as f:
        for line in f:
            obj = json.loads(line.decode("utf-8"))
            entry = obj["entry"]
            entry_prev = entry.get("prev")
            if entry_prev != prev:
                return False
            enc = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if hashlib.sha256(enc).hexdigest() != obj.get("entry_hash"):
                return False
            prev = obj.get("entry_hash")
    return True
