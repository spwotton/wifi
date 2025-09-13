from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple


def _rec_hash(rec: Dict) -> str:
    """Stable hash of a record using canonicalized JSON (sorted keys)."""
    data = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _connect(dsn: str):
    try:
        import psycopg2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("psycopg2-binary not installed. pip install psycopg2-binary") from e
    return psycopg2.connect(dsn)


def _ensure_tables(conn) -> None:
    cur = conn.cursor()
    # sonar_events
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sonar_events (
            id BIGINT,
            timestamp DOUBLE PRECISION,
            samplerate INTEGER,
            prf_hz DOUBLE PRECISION,
            band_low_hz DOUBLE PRECISION,
            band_high_hz DOUBLE PRECISION,
            snr_db DOUBLE PRECISION,
            method TEXT,
            notes TEXT,
            rec_hash TEXT UNIQUE,
            inserted_at TIMESTAMPTZ DEFAULT now()
        );
        """
    )
    # wifi_frames
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wifi_frames (
            ts DOUBLE PRECISION,
            subtype TEXT,
            bssid TEXT,
            src TEXT,
            dst TEXT,
            channel INTEGER,
            rssi DOUBLE PRECISION,
            reason TEXT,
            notes TEXT,
            rec_hash TEXT UNIQUE,
            inserted_at TIMESTAMPTZ DEFAULT now()
        );
        """
    )
    # rtl433_messages
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rtl433_messages (
            ts DOUBLE PRECISION,
            model TEXT,
            freq DOUBLE PRECISION,
            data_json TEXT,
            rec_hash TEXT UNIQUE,
            inserted_at TIMESTAMPTZ DEFAULT now()
        );
        """
    )
    # ai_reports
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_reports (
            timestamp DOUBLE PRECISION,
            kind TEXT,
            provider TEXT,
            model TEXT,
            path TEXT,
            sha256 TEXT,
            size_bytes BIGINT,
            rec_hash TEXT UNIQUE,
            inserted_at TIMESTAMPTZ DEFAULT now()
        );
        """
    )
    conn.commit()


def _upsert_batch(conn, table: str, cols: List[str], rows: List[Tuple]):
    if not rows:
        return 0
    placeholders = ",".join(["%s"] * (len(cols) + 1))  # +1 for rec_hash
    col_list = ",".join(cols + ["rec_hash"])
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT (rec_hash) DO NOTHING"
    cur = conn.cursor()
    cur.executemany(sql, rows)
    # rows inserted = rowcount but for ON CONFLICT DO NOTHING rowcount is total attempted; can't rely.
    # We'll just commit and return 0 (or estimate). For visibility, caller can count JSONL lines.
    conn.commit()
    return cur.rowcount


def offload_postgres(from_dir: Path, dsn: str, create_tables: bool = True) -> Dict[str, int]:
    """Offload append-only JSONL exports into Postgres with idempotent upserts.

    Returns a dict of attempted per-table counts.
    """
    from_dir = Path(from_dir)
    conn = _connect(dsn)
    if create_tables:
        _ensure_tables(conn)

    stats: Dict[str, int] = {}

    # sonar_events
    se_path = from_dir / "sonar_events.jsonl"
    if se_path.exists():
        cols = ["id", "timestamp", "samplerate", "prf_hz", "band_low_hz", "band_high_hz", "snr_db", "method", "notes"]
        rows: List[Tuple] = []
        n = 0
        for rec in _load_jsonl(se_path):
            n += 1
            rec_hash = _rec_hash(rec)
            rows.append(tuple(rec.get(c) for c in cols) + (rec_hash,))
            if len(rows) >= 500:
                _upsert_batch(conn, "sonar_events", cols, rows)
                rows = []
        if rows:
            _upsert_batch(conn, "sonar_events", cols, rows)
        stats["sonar_events"] = n

    # wifi_frames
    wf_path = from_dir / "wifi_frames.jsonl"
    if wf_path.exists():
        cols = ["ts", "subtype", "bssid", "src", "dst", "channel", "rssi", "reason", "notes"]
        rows = []
        n = 0
        for rec in _load_jsonl(wf_path):
            n += 1
            rec_hash = _rec_hash(rec)
            rows.append(tuple(rec.get(c) for c in cols) + (rec_hash,))
            if len(rows) >= 1000:
                _upsert_batch(conn, "wifi_frames", cols, rows)
                rows = []
        if rows:
            _upsert_batch(conn, "wifi_frames", cols, rows)
        stats["wifi_frames"] = n

    # rtl433_messages
    rtl_path = from_dir / "rtl433_messages.jsonl"
    if rtl_path.exists():
        cols = ["ts", "model", "freq", "data_json"]
        rows = []
        n = 0
        for rec in _load_jsonl(rtl_path):
            n += 1
            # normalize data field name variations
            if "data_json" not in rec and "data" in rec:
                rec["data_json"] = json.dumps(rec["data"], separators=(",", ":")) if not isinstance(rec["data"], str) else rec["data"]
            rec_hash = _rec_hash(rec)
            rows.append(tuple(rec.get(c) for c in cols) + (rec_hash,))
            if len(rows) >= 1000:
                _upsert_batch(conn, "rtl433_messages", cols, rows)
                rows = []
        if rows:
            _upsert_batch(conn, "rtl433_messages", cols, rows)
        stats["rtl433_messages"] = n

    # ai_reports
    ai_path = from_dir / "ai_reports.jsonl"
    if ai_path.exists():
        cols = ["timestamp", "kind", "provider", "model", "path", "sha256", "size_bytes"]
        rows = []
        n = 0
        for rec in _load_jsonl(ai_path):
            n += 1
            rec_hash = _rec_hash(rec)
            rows.append(tuple(rec.get(c) for c in cols) + (rec_hash,))
            if len(rows) >= 500:
                _upsert_batch(conn, "ai_reports", cols, rows)
                rows = []
        if rows:
            _upsert_batch(conn, "ai_reports", cols, rows)
        stats["ai_reports"] = n

    conn.close()
    return stats
