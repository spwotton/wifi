from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from wlan.analyzer import compute_anomalies


@dataclass
class Event:
    ts: float
    source: str
    subtype: str
    info: str


def _fetch_events(
    db_path: Path,
    *,
    include_raw_wifi: bool = False,
    wifi_window_sec: int = 60,
    wifi_deauth_thresh: int = 5,
    wifi_disassoc_thresh: int = 5,
    wifi_min_count: int = 3,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> list[Event]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    out: list[Event] = []
    # Sonar
    try:
        cur.execute("SELECT timestamp, method, notes FROM sonar_events ORDER BY timestamp ASC")
        for ts, method, notes in cur.fetchall():
            out.append(Event(float(ts), "sonar", str(method), str(notes)))
    except sqlite3.OperationalError:
        pass
    # Wi‑Fi raw frames (optional)
    if include_raw_wifi:
        try:
            q = "SELECT ts, subtype, bssid FROM wifi_frames ORDER BY ts ASC"
            if start_ts is not None or end_ts is not None:
                clauses = []
                params = []
                if start_ts is not None:
                    clauses.append("ts >= ?")
                    params.append(float(start_ts))
                if end_ts is not None:
                    clauses.append("ts <= ?")
                    params.append(float(end_ts))
                q = q.replace(" ORDER BY", " WHERE " + " AND ".join(clauses) + " ORDER BY")
                cur.execute(q, params)
            else:
                cur.execute(q)
            for ts, subtype, bssid in cur.fetchall():
                out.append(Event(float(ts), "wifi", str(subtype), str(bssid)))
        except sqlite3.OperationalError:
            pass
    # Wi‑Fi anomaly windows as events
    try:
        anomalies = compute_anomalies(
            db_path,
            window_sec=wifi_window_sec,
            deauth_thresh=wifi_deauth_thresh,
            disassoc_thresh=wifi_disassoc_thresh,
            min_count=wifi_min_count,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        for rec in anomalies:
            flags = []
            if rec.get("flag_deauth"):
                flags.append("DEAUTH")
            if rec.get("flag_disassoc"):
                flags.append("DISASSOC")
            info = (
                f"bssid={rec['bssid']} deauth={rec['deauth']} disassoc={rec['disassoc']} "
                f"score={rec['score']:.2f} flags={'+'.join(flags) if flags else '-'}"
            )
            out.append(Event(float(rec["window_start"]), "wifi-anom", "anomaly", info))
    except Exception:
        pass
    # rtl_433
    try:
        cur.execute("SELECT ts, model, freq FROM rtl433_messages ORDER BY ts ASC")
        for ts, model, freq in cur.fetchall():
            out.append(Event(float(ts), "rtl433", str(model), f"freq={freq}"))
    except sqlite3.OperationalError:
        pass
    return out


def correlate(
    db_path: Path,
    tolerance: float = 2.0,
    *,
    include_raw_wifi: bool = False,
    wifi_window_sec: int = 60,
    wifi_deauth_thresh: int = 5,
    wifi_disassoc_thresh: int = 5,
    wifi_min_count: int = 3,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> list[dict]:
    """Find temporal clusters where events from 2+ sources occur within tolerance seconds.

    Includes Wi‑Fi anomaly windows (computed on the fly) to correlate with sonar/rtl_433 events.
    Set include_raw_wifi=True to also include raw Wi‑Fi frames in the clustering.
    """
    events = sorted(
        _fetch_events(
            db_path,
            include_raw_wifi=include_raw_wifi,
            wifi_window_sec=wifi_window_sec,
            wifi_deauth_thresh=wifi_deauth_thresh,
            wifi_disassoc_thresh=wifi_disassoc_thresh,
            wifi_min_count=wifi_min_count,
            start_ts=start_ts,
            end_ts=end_ts,
        ),
        key=lambda e: e.ts,
    )
    clusters: list[list[Event]] = []
    current: list[Event] = []
    for e in events:
        if not current:
            current = [e]
            continue
        if e.ts - current[0].ts <= tolerance:
            current.append(e)
        else:
            clusters.append(current)
            current = [e]
    if current:
        clusters.append(current)

    results: list[dict] = []
    for cluster in clusters:
        sources = {ev.source for ev in cluster}
        if len(sources) < 2:
            continue
        results.append(
            {
                "start_ts": cluster[0].ts,
                "end_ts": cluster[-1].ts,
                "sources": sorted(list(sources)),
                "count": len(cluster),
                "items": [
                    {"ts": ev.ts, "source": ev.source, "subtype": ev.subtype, "info": ev.info}
                    for ev in cluster
                ],
            }
        )
    return results
