from __future__ import annotations

import base64
import html
import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from wlan.analyzer import compute_anomalies
from aegis.correlate import correlate

try:  # optional plotting
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def _q_scalar(conn: sqlite3.Connection, q: str, params: list | tuple = ()) -> int:
    try:
        cur = conn.execute(q, params)
        row = cur.fetchone()
        return int(row[0] if row and row[0] is not None else 0)
    except sqlite3.OperationalError:
        return 0


def _fmt_ts(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return str(ts)


def generate_html_report(
    db_path: Path,
    out_html: Path,
    *,
    wifi_window_sec: int = 60,
    wifi_deauth_thresh: int = 5,
    wifi_disassoc_thresh: int = 5,
    wifi_min_count: int = 3,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    corr_tolerance: float = 2.0,
    corr_include_raw_wifi: bool = False,
) -> Path:
    conn = sqlite3.connect(db_path)
    # Basic counts
    sonar_count = _q_scalar(conn, "SELECT COUNT(*) FROM sonar_events")
    wifi_count = _q_scalar(conn, "SELECT COUNT(*) FROM wifi_frames")
    rtl_count = _q_scalar(conn, "SELECT COUNT(*) FROM rtl433_messages")

    # Recent sonar sample
    sonar_rows = []
    try:
        cur = conn.execute(
            "SELECT timestamp, samplerate, prf_hz, snr_db, method, notes FROM sonar_events ORDER BY timestamp DESC LIMIT 10"
        )
        sonar_rows = cur.fetchall()
    except sqlite3.OperationalError:
        sonar_rows = []

    # Wi‑Fi anomalies
    anomalies = []
    if wifi_count > 0:
        anomalies = compute_anomalies(
            db_path,
            window_sec=wifi_window_sec,
            deauth_thresh=wifi_deauth_thresh,
            disassoc_thresh=wifi_disassoc_thresh,
            min_count=wifi_min_count,
            start_ts=start_ts,
            end_ts=end_ts,
        )

    # Correlation clusters
    try:
        clusters = correlate(
            db_path,
            tolerance=corr_tolerance,
            include_raw_wifi=corr_include_raw_wifi,
            wifi_window_sec=wifi_window_sec,
            wifi_deauth_thresh=wifi_deauth_thresh,
            wifi_disassoc_thresh=wifi_disassoc_thresh,
            wifi_min_count=wifi_min_count,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    except Exception:
        clusters = []

    # Prepare charts
    def _fig_to_data_uri() -> str:
        if plt is None:
            return ""
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120)
        plt.close()
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    wifi_ts_uri = ""
    if plt is not None and anomalies:
        # Aggregate anomaly score per window_start
        agg = {}
        for rec in anomalies:
            ts = float(rec["window_start"])  # already epoch seconds
            agg[ts] = agg.get(ts, 0.0) + float(rec.get("score", 0.0))
        xs = sorted(agg.keys())
        ys = [agg[t] for t in xs]
        if xs:
            plt.figure(figsize=(6, 2.2))
            plt.plot([time.strftime("%H:%M:%S", time.localtime(x)) for x in xs], ys, marker="o")
            plt.title("Wi‑Fi anomaly score over time")
            plt.ylabel("score")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)
            wifi_ts_uri = _fig_to_data_uri()

    rtl_bar_uri = ""
    if plt is not None and rtl_count > 0:
        try:
            cur = conn.execute(
                "SELECT model, COUNT(*) AS c FROM rtl433_messages GROUP BY model ORDER BY c DESC LIMIT 10"
            )
            rows = cur.fetchall()
            if rows:
                labels = [str(r[0]) for r in rows]
                counts = [int(r[1]) for r in rows]
                plt.figure(figsize=(6, 2.2))
                plt.bar(labels, counts)
                plt.title("rtl_433 top models")
                plt.ylabel("count")
                plt.xticks(rotation=30, ha="right")
                rtl_bar_uri = _fig_to_data_uri()
        except sqlite3.OperationalError:
            pass

    sonar_hist_uri = ""
    if plt is not None and sonar_count > 0:
        try:
            cur = conn.execute("SELECT prf_hz FROM sonar_events WHERE prf_hz IS NOT NULL AND prf_hz > 0")
            vals = [float(r[0]) for r in cur.fetchall() if r and r[0] is not None]
            if vals:
                plt.figure(figsize=(6, 2.2))
                plt.hist(vals, bins=min(20, max(5, int(len(vals) ** 0.5))), color="#4a90e2")
                plt.title("Sonar PRF histogram (Hz)")
                plt.xlabel("PRF (Hz)")
                plt.ylabel("count")
                sonar_hist_uri = _fig_to_data_uri()
        except sqlite3.OperationalError:
            pass

    # Score clusters to help rank significance
    def _score_cluster(c: dict) -> float:
        base = 0.0
        # Favor presence of wifi anomalies and multiple sources
        srcs = set(c.get("sources", []))
        base += 1.5 if "wifi-anom" in srcs else 0.0
        base += 1.0 if "sonar" in srcs else 0.0
        base += 0.5 if "rtl433" in srcs else 0.0
        base += 0.1 * max(0, int(c.get("count", 0)) - 1)
        # If items include explicit scores from wifi-anom info, add a bit
        try:
            for it in c.get("items", []) or []:
                if it.get("source") == "wifi-anom":
                    # parse score=... from info string
                    info = str(it.get("info", ""))
                    if "score=" in info:
                        s = info.split("score=", 1)[1]
                        s = s.split()[0].strip()
                        base += min(2.0, float(s)) * 0.2
        except Exception:
            pass
        return float(base)

    clusters_sorted = sorted(clusters, key=_score_cluster, reverse=True)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("""<!doctype html><html><head><meta charset='utf-8'>
<title>RF/Wi‑Fi/Acoustic Report</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;}
header{margin-bottom:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}
table{border-collapse:collapse;width:100%;}
th,td{border-bottom:1px solid #eee;padding:6px 8px;text-align:left;font-size:14px}
th{background:#f6f6f6}
.flag{display:inline-block;padding:2px 6px;border-radius:4px;background:#e00;color:#fff;margin-right:6px;font-size:12px}
.muted{color:#666}
code{background:#f4f4f4;padding:1px 4px;border-radius:4px}
</style></head><body>""")
        f.write("<header><h1>RF/Wi‑Fi/Acoustic Daily Report</h1>")
        f.write(f"<div class='muted'>Generated: {_fmt_ts(time.time())}</div>")
        f.write("</header>")

        # Summary cards
        f.write("<section class='cards'>")
        f.write(f"<div class='card'><h3>Sonar events</h3><div><b>{sonar_count}</b></div></div>")
        f.write(f"<div class='card'><h3>Wi‑Fi frames</h3><div><b>{wifi_count}</b></div></div>")
        f.write(f"<div class='card'><h3>rtl_433 msgs</h3><div><b>{rtl_count}</b></div></div>")
        f.write(f"<div class='card'><h3>Clusters</h3><div><b>{len(clusters)}</b></div></div>")
        f.write("</section>")

    # Recent sonar
        f.write("<h2>Recent sonar events</h2>")
        if not sonar_rows:
            f.write("<div class='muted'>No sonar events logged.</div>")
        else:
            f.write("<table><thead><tr><th>Time</th><th>PRF (Hz)</th><th>SNR (dB)</th><th>Method</th><th>Notes</th></tr></thead><tbody>")
            for ts, sr, prf_hz, snr_db, method, notes in sonar_rows:
                f.write(
                    f"<tr><td>{_fmt_ts(ts)}</td><td>{prf_hz:.2f}</td><td>{snr_db:.1f}</td><td>{html.escape(method)}</td><td>{html.escape(notes or '')}</td></tr>"
                )
            f.write("</tbody></table>")

    # Wi‑Fi anomalies
        f.write("<h2>Wi‑Fi anomalies</h2>")
        f.write(
            f"<div class='muted'>window={wifi_window_sec}s deauth_thresh={wifi_deauth_thresh} disassoc_thresh={wifi_disassoc_thresh} min_count={wifi_min_count}</div>"
        )
        if wifi_ts_uri:
            f.write(f"<div><img alt='Wi‑Fi anomaly series' src='{wifi_ts_uri}'/></div>")
        if not anomalies:
            f.write("<div class='muted'>No anomalies at current thresholds.</div>")
        else:
            f.write("<table><thead><tr><th>Window start</th><th>BSSID</th><th>deauth</th><th>disassoc</th><th>score</th><th>flags</th></tr></thead><tbody>")
            for rec in anomalies:
                flags = []
                if rec.get("flag_deauth"):
                    flags.append("DEAUTH")
                if rec.get("flag_disassoc"):
                    flags.append("DISASSOC")
                f.write(
                    f"<tr><td>{_fmt_ts(rec['window_start'])}</td><td><code>{html.escape(rec['bssid'])}</code></td><td>{rec['deauth']}</td><td>{rec['disassoc']}</td><td>{rec['score']:.2f}</td><td>{' '.join(f'<span class=flag>{html.escape(fl)}</span>' for fl in flags) if flags else '-'}</td></tr>"
                )
            f.write("</tbody></table>")

        # rtl_433 top models
        f.write("<h2>rtl_433 top models</h2>")
        if rtl_bar_uri:
            f.write(f"<div><img alt='rtl_433 top models' src='{rtl_bar_uri}'/></div>")
        elif rtl_count == 0:
            f.write("<div class='muted'>No rtl_433 data.</div>")

        # Sonar PRF histogram
        f.write("<h2>Sonar PRF histogram</h2>")
        if sonar_hist_uri:
            f.write(f"<div><img alt='Sonar PRF histogram' src='{sonar_hist_uri}'/></div>")
        elif sonar_count == 0:
            f.write("<div class='muted'>No sonar events.</div>")

        # Correlation clusters
        f.write("<h2>Cross-source clusters</h2>")
        if not clusters_sorted:
            f.write("<div class='muted'>No clusters found.</div>")
        else:
            f.write("<table><thead><tr><th>#</th><th>Score</th><th>Start</th><th>End</th><th>Sources</th><th>Count</th><th>Details</th></tr></thead><tbody>")
            for i, c in enumerate(clusters_sorted, 1):
                sc = _score_cluster(c)
                sources = html.escape(','.join(sorted(set(c.get('sources', [])))))
                # Compact details: up to 3 items summarized
                details = []
                for it in (c.get('items') or [])[:3]:
                    details.append(html.escape(f"{it.get('source')}:{it.get('subtype')}"))
                detail_str = ', '.join(details) if details else '-'
                f.write(
                    f"<tr><td>{i}</td><td>{sc:.2f}</td><td>{_fmt_ts(c['start_ts'])}</td><td>{_fmt_ts(c['end_ts'])}</td><td>{sources}</td><td>{c.get('count', 0)}</td><td>{detail_str}</td></tr>"
                )
            f.write("</tbody></table>")

        f.write("</body></html>")

    return out_html
