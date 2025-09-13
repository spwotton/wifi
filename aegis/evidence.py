from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TableSpec:
    name: str
    query: str
    headers: list[str]


TABLES: list[TableSpec] = [
    TableSpec(
        name="sonar_events",
        query="SELECT id, timestamp, samplerate, prf_hz, band_low_hz, band_high_hz, snr_db, method, notes FROM sonar_events ORDER BY id ASC",
        headers=[
            "id",
            "timestamp",
            "samplerate",
            "prf_hz",
            "band_low_hz",
            "band_high_hz",
            "snr_db",
            "method",
            "notes",
        ],
    ),
    TableSpec(
        name="wifi_frames",
        query="SELECT ts, subtype, bssid, src, dst, channel, rssi, reason, notes FROM wifi_frames ORDER BY ts ASC",
        headers=["ts", "subtype", "bssid", "src", "dst", "channel", "rssi", "reason", "notes"],
    ),
    TableSpec(
        name="rtl433_messages",
        query="SELECT ts, model, freq, data FROM rtl433_messages ORDER BY ts ASC",
        headers=["ts", "model", "freq", "data_json"],
    ),
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _time_range(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [float(min(values)), float(max(values))]


def package_evidence(db_path: Path, out_dir: Path) -> Path:
    """Export CSVs for key tables and write a manifest.json with hashes and ranges.

    Returns path to manifest.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    manifest = {
        "package_id": time.strftime("%Y%m%d_%H%M%S", time.gmtime()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tables": [],
    }
    for spec in TABLES:
        cur = conn.cursor()
        try:
            cur.execute(spec.query)
        except sqlite3.OperationalError:
            # Table may not exist; skip
            continue
        rows = cur.fetchall()
        csv_path = out_dir / f"{spec.name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(spec.headers)
            w.writerows(rows)
        row_count = len(rows)
        # Collect time ranges if a timestamp column exists in typical position
        ts_values: list[float] = []
        # Heuristic: for sonar_events, index 1; others index 0
        ts_index = 1 if spec.name == "sonar_events" else 0
        for r in rows:
            try:
                ts_values.append(float(r[ts_index]))
            except Exception:
                pass
        entry = {
            "name": spec.name,
            "csv": csv_path.name,
            "rows": row_count,
            "sha256": _sha256_file(csv_path),
            "time_range": _time_range(ts_values),
        }
        manifest["tables"].append(entry)
    # Top-level hash over concatenation of table hashes
    top = hashlib.sha256()
    for t in manifest["tables"]:
        top.update(bytes.fromhex(t["sha256"]))
    manifest["package_sha256"] = top.hexdigest()
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest_path
