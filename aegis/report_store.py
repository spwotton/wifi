from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from aegis_ultra.manifest import ManifestEntry, append_manifest


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_ai_tables(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                kind TEXT NOT NULL,            -- 'base' | 'summary'
                provider TEXT,                 -- e.g., 'gemini' | 'openai' | 'hf'
                model TEXT,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                content TEXT                   -- optional copy of the text
            );
            """
        )


def store_ai_report(
    db_path: Path,
    kind: str,
    path: Path,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    store_content: bool = True,
) -> int:
    ensure_ai_tables(db_path)
    text = path.read_text(encoding="utf-8") if store_content else ""
    sha = _sha256_file(path)
    sz = os.path.getsize(path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_reports (timestamp, kind, provider, model, path, sha256, size_bytes, content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (time.time(), kind, provider, model, str(path), sha, int(sz), text),
        )
        conn.commit()
        return int(cur.lastrowid)


def append_reports_manifest(out_root: Path, kind: str, path: Path, sha256: str, meta: dict | None = None) -> str:
    man_dir = out_root / "reports"
    man_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = man_dir / "manifest.jsonl"
    entry = ManifestEntry(ts=time.time(), kind=kind, path=str(path), sha256=sha256, meta=meta or {})
    return append_manifest(manifest_path, entry)


def llm_call_log(out_root: Path, payload: dict) -> None:
    log_dir = out_root / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "llm_calls.jsonl"
    payload = {"ts": time.time(), **payload}
    with log_path.open("ab") as f:
        f.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
