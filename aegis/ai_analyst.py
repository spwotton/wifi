from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .correlate import correlate
from wlan.analyzer import compute_anomalies


@dataclass
class AnalystConfig:
    tolerance: float = 2.0
    include_raw_wifi: bool = False
    wifi_window_sec: int = 60
    wifi_deauth_thresh: int = 5
    wifi_disassoc_thresh: int = 5
    wifi_min_count: int = 3


def _query_last_sonar_events(db: Path, start_ts: Optional[float], end_ts: Optional[float], max_rows: int = 50):
    q = "SELECT timestamp, samplerate, prf_hz, band_low_hz, band_high_hz, snr_db, method, notes FROM sonar_events"
    cond = []
    args: list[float] = []
    if start_ts is not None:
        cond.append("timestamp >= ?")
        args.append(float(start_ts))
    if end_ts is not None:
        cond.append("timestamp <= ?")
        args.append(float(end_ts))
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY timestamp DESC LIMIT ?"
    args.append(int(max_rows))
    with sqlite3.connect(db) as conn:
        return list(conn.execute(q, tuple(args)))


def _rtl433_summary(db: Path, start_ts: Optional[float], end_ts: Optional[float]) -> int:
    try:
        with sqlite3.connect(db) as conn:
            cur = conn.cursor()
            if start_ts is None and end_ts is None:
                q = "SELECT COUNT(*) FROM rtl433_messages"
                return int(cur.execute(q).fetchone()[0])
            cond = []
            args: list[float] = []
            if start_ts is not None:
                cond.append("timestamp >= ?")
                args.append(float(start_ts))
            if end_ts is not None:
                cond.append("timestamp <= ?")
                args.append(float(end_ts))
            q = "SELECT COUNT(*) FROM rtl433_messages WHERE " + " AND ".join(cond)
            return int(cur.execute(q, tuple(args)).fetchone()[0])
    except Exception:
        return 0


def generate_ai_incident_report(
    db_path: Path,
    out_path: Path,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    cfg: AnalystConfig = AnalystConfig(),
) -> Path:
    # Gather data
    clusters = correlate(
        db_path,
        tolerance=float(cfg.tolerance),
        include_raw_wifi=bool(cfg.include_raw_wifi),
        wifi_window_sec=int(cfg.wifi_window_sec),
        wifi_deauth_thresh=int(cfg.wifi_deauth_thresh),
        wifi_disassoc_thresh=int(cfg.wifi_disassoc_thresh),
        wifi_min_count=int(cfg.wifi_min_count),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    anomalies = compute_anomalies(
        db_path,
        window_sec=int(cfg.wifi_window_sec),
        deauth_thresh=int(cfg.wifi_deauth_thresh),
        disassoc_thresh=int(cfg.wifi_disassoc_thresh),
        min_count=int(cfg.wifi_min_count),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    sonar_rows = _query_last_sonar_events(db_path, start_ts, end_ts, max_rows=200)
    rtl_count = _rtl433_summary(db_path, start_ts, end_ts)

    # Heuristic reasoning
    now = time.time()
    time_window = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_ts))} .. {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_ts))}" if (start_ts and end_ts) else "recent period"
    critical_findings: list[str] = []
    insights: list[str] = []
    recommendations: list[str] = []

    # Elevate if clusters show multi-modal coupling
    if clusters:
        multimodal = [c for c in clusters if len(c.get("sources", [])) >= 2]
        if multimodal:
            best = max(multimodal, key=lambda c: c.get("count", 0))
            sources = ",".join(sorted(best.get("sources", [])))
            critical_findings.append(
                f"Detected temporally coupled anomalies across {sources} around {time.strftime('%H:%M:%S', time.localtime(best['start_ts']))}–{time.strftime('%H:%M:%S', time.localtime(best['end_ts']))} (count={best.get('count', 0)})."
            )
            # Confidence by count + source diversity
            conf = min(0.5 + 0.1 * best.get("count", 0) + 0.2 * (len(best.get("sources", [])) - 1), 0.95)
            insights.append(f"Confidence this was coordinated: {conf:.0%} (heuristic)")
            # Scenario mapping
            if "sonar" in sources and "wifi" in sources:
                insights.append("Pattern matches an acoustic trigger coincident with Wi‑Fi disruption (e.g., deauth spike).")
                recommendations.append("Isolate critical cameras/APs; review Wi‑Fi logs near these timestamps; if possible, capture high-rate (≥96 kHz) audio for demodulation.")
            if "rtl433" in sources:
                insights.append("Co-occurring ISM band activity was observed.")
    else:
        insights.append("No cross-source clusters exceeded thresholds in this period.")

    # Wi‑Fi anomaly posture
    if anomalies:
        worst = max(anomalies, key=lambda a: a.get("score", 0))
        insights.append(
            f"Wi‑Fi anomalies observed: top window at {worst['window_start']} with score {worst['score']:.1f} (deauth={worst['deauth']}, disassoc={worst['disassoc']})."
        )
        recommendations.append("Ensure management frame protection (802.11w) where supported; rotate PSKs; review AP placement to reduce exterior leakage.")

    # Sonar posture
    if sonar_rows:
        recent = sonar_rows[0]
        notes = str(recent[7])
        if "demod=" in notes or "ultra" in notes:
            insights.append("Ultrasonic energy detected recently; consider targeted high-rate capture for demodulation.")
        recommendations.append("Disable or restrict always-on microphones in non-essential devices; prefer push-to-talk modes.")

    # rtl_433
    if rtl_count > 0:
        insights.append(f"rtl_433 decoded {rtl_count} ISM/OOK messages in the period (likely ambient devices).")

    # Baseline recommendations
    if not recommendations:
        recommendations.extend([
            "Maintain current posture; continue passive monitoring.",
            "Schedule periodic reviews of anomaly thresholds based on environment.",
        ])

    # Compose Markdown report
    md = []
    md.append(f"# AI Security Analyst Report\n")
    md.append(f"Period: {time_window}\n\n")
    if critical_findings:
        md.append("## Critical findings\n")
        for c in critical_findings:
            md.append(f"- {c}\n")
        md.append("\n")
    md.append("## Insights\n")
    for i in insights:
        md.append(f"- {i}\n")
    md.append("\n")
    md.append("## Recommendations\n")
    for r in recommendations:
        md.append(f"- {r}\n")
    md.append("\n")
    # Timeline highlights (clusters)
    if clusters:
        md.append("## Timeline highlights\n")
        for i, c in enumerate(clusters[:10], 1):
            md.append(
                f"- [{i}] {time.strftime('%H:%M:%S', time.localtime(c['start_ts']))}–{time.strftime('%H:%M:%S', time.localtime(c['end_ts']))} sources={','.join(c['sources'])} count={c['count']}\n"
            )
        md.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")
    return out_path
