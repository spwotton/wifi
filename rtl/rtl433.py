from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional


def _has_rtl433() -> bool:
    return shutil.which("rtl_433") is not None


def _init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rtl433_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                model TEXT,
                freq REAL,
                data TEXT
            );
            """
        )
    return conn


def run_rtl433(duration: float, db_path: Path, device_index: Optional[int] = None, extra_args: Optional[list[str]] = None) -> int:
    if not _has_rtl433():
        raise RuntimeError("rtl_433 not found in PATH. Install rtl_433 and retry.")
    conn = _init_db(db_path)
    cmd = ["rtl_433", "-F", "json", "-A", "-T", str(int(duration))]
    if device_index is not None:
        cmd += ["-d", str(device_index)]
    if extra_args:
        cmd += list(extra_args)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    inserted = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = obj.get("time") or obj.get("timestamp") or time.time()
        try:
            ts_val = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")) if isinstance(ts, str) else float(ts)
        except Exception:
            ts_val = float(time.time())
        model = obj.get("model") or obj.get("protocol") or "unknown"
        freq = obj.get("freq") or obj.get("frequency")
        try:
            freq_val = float(freq) if freq is not None else None
        except Exception:
            freq_val = None
        with conn:
            conn.execute(
                "INSERT INTO rtl433_messages(ts, model, freq, data) VALUES (?,?,?,?)",
                (ts_val, str(model), freq_val, json.dumps(obj)),
            )
        inserted += 1
    proc.wait()
    return inserted


def export_rtl433(db_path: Path, out_csv: Path, start_ts: Optional[float] = None, end_ts: Optional[float] = None) -> int:
    import csv
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    q = "SELECT ts, model, freq, data FROM rtl433_messages"
    params: list = []
    if start_ts is not None or end_ts is not None:
        clauses = []
        if start_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(start_ts))
        if end_ts is not None:
            clauses.append("ts <= ?")
            params.append(float(end_ts))
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY ts ASC"
    cur.execute(q, params)
    rows = cur.fetchall()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "model", "freq", "data_json"])
        w.writerows(rows)
    return len(rows)
