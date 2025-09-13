from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import time


def _has_tshark() -> bool:
    return shutil.which("tshark") is not None


@dataclass
class WiFiFrameSummary:
    ts: float
    subtype: str
    bssid: str
    src: str
    dst: str
    channel: Optional[int]
    rssi: Optional[float]
    reason: Optional[int]


def _init_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wifi_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                subtype TEXT,
                bssid TEXT,
                src TEXT,
                dst TEXT,
                channel INTEGER,
                rssi REAL,
                reason INTEGER,
                notes TEXT
            );
            """
        )
    return conn


def tshark_json_iter(pcap: Path) -> Iterable[WiFiFrameSummary]:
    """Yield simplified Wi-Fi management frames from a pcap/pcapng using tshark.

    Requires tshark with support for 802.11 fields.
    """
    if not _has_tshark():
        raise RuntimeError("tshark not found in PATH. Install Wireshark/tshark and retry.")
    # Fields to extract
    fields = [
        "frame.time_epoch",
        "wlan.fc.type_subtype",
        "wlan.bssid",
        "wlan.sa",
        "wlan.da",
        "radiotap.dbm_antsignal",
        "radiotap.channel.freq",
        "wlan.fixed.reason_code",
    ]
    cmd = [
        "tshark",
        "-r",
        str(pcap),
        "-Y",
        "wlan && (wlan.fc.type_subtype == 0x000c || wlan.fc.type_subtype == 0x000a || wlan.fc.type_subtype == 0x0008 || wlan.fc.type == 0)",
        "-T",
        "json",
        "-e",
    ]
    # tshark -T json ignores -e; json includes all fields, so filter in Python
    out = subprocess.check_output(["tshark", "-r", str(pcap), "-T", "json"], stderr=subprocess.DEVNULL)
    packets = json.loads(out.decode("utf-8", errors="ignore"))
    for pkt in packets:
        layers = pkt.get("_source", {}).get("layers", {})
        wlan = layers.get("wlan", {})
        rtap = layers.get("radiotap", {})
        frame = layers.get("frame", {})
        subtype = wlan.get("wlan.fc.type_subtype", [None])
        subtype = subtype[0] if isinstance(subtype, list) else subtype
        # Map hex subtype to human label when possible
        subtype_label = {
            "0x0008": "beacon",
            "0x000c": "deauth",
            "0x000a": "disassoc",
        }.get(subtype, str(subtype))
        ts = frame.get("frame.time_epoch", [None])
        ts = float(ts[0]) if ts and isinstance(ts, list) else None
        bssid = wlan.get("wlan.bssid", [""])
        bssid = bssid[0] if isinstance(bssid, list) else bssid
        src = wlan.get("wlan.sa", [""])
        src = src[0] if isinstance(src, list) else src
        dst = wlan.get("wlan.da", [""])
        dst = dst[0] if isinstance(dst, list) else dst
        rssi = rtap.get("radiotap.dbm_antsignal", [None])
        rssi = float(rssi[0]) if rssi and isinstance(rssi, list) else None
        chf = rtap.get("radiotap.channel.freq", [None])
        channel = None
        try:
            if chf and isinstance(chf, list) and chf[0] is not None:
                fmhz = float(chf[0]) / 1e6
                # Basic mapping: 2.4 GHz channels
                if 2412 <= float(chf[0]) <= 2472:
                    channel = int(round((float(chf[0]) - 2407) / 5.0))
        except Exception:
            channel = None
        reason = wlan.get("wlan.fixed.reason_code", [None])
        reason = int(reason[0]) if reason and isinstance(reason, list) and reason[0] is not None else None

        if subtype_label in {"beacon", "deauth", "disassoc"}:
            yield WiFiFrameSummary(
                ts=ts or 0.0,
                subtype=subtype_label,
                bssid=bssid or "",
                src=src or "",
                dst=dst or "",
                channel=channel,
                rssi=rssi,
                reason=reason,
            )


def analyze_pcap_to_db(pcap: Path, db_path: Path) -> dict:
    """Parse pcap and insert Wi-Fi frames into SQLite; return quick stats."""
    conn = _init_db(db_path)
    counts = {"beacon": 0, "deauth": 0, "disassoc": 0}
    with conn:
        for rec in tshark_json_iter(pcap):
            conn.execute(
                "INSERT INTO wifi_frames(ts, subtype, bssid, src, dst, channel, rssi, reason, notes) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    rec.ts,
                    rec.subtype,
                    rec.bssid,
                    rec.src,
                    rec.dst,
                    rec.channel,
                    rec.rssi,
                    rec.reason,
                    "pcap",
                ),
            )
            if rec.subtype in counts:
                counts[rec.subtype] += 1
    return counts


def live_capture_and_analyze(interface: str, duration: float, db_path: Path) -> dict:
    """Start a short live capture to temp pcap, then analyze it. Requires monitor mode capable OS/adapter."""
    if not _has_tshark():
        raise RuntimeError("tshark not found in PATH.")
    with tempfile.TemporaryDirectory() as td:
        pcap = Path(td) / "live.pcapng"
        cmd = [
            "tshark",
            "-I",
            "-i",
            interface,
            "-a",
            f"duration:{int(duration)}",
            "-w",
            str(pcap),
        ]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"tshark live capture failed: {e}")
        return analyze_pcap_to_db(pcap, db_path)


def export_wifi_frames(db_path: Path, out_csv: Path, start_ts: Optional[float] = None, end_ts: Optional[float] = None) -> int:
    """Export wifi_frames to CSV; returns row count."""
    import csv

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    q = "SELECT ts, subtype, bssid, src, dst, channel, rssi, reason, notes FROM wifi_frames"
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
        w.writerow(["ts", "subtype", "bssid", "src", "dst", "channel", "rssi", "reason", "notes"])
        w.writerows(rows)
    return len(rows)


def seed_demo_wifi(db_path: Path) -> dict:
    """Insert synthetic Wi‑Fi management frames to demonstrate anomaly detection.

    Creates a few 60s windows around 'now' for two BSSIDs, with one clear deauth burst.
    Returns counts per subtype.
    """
    conn = _init_db(db_path)
    now = time.time()
    win = int(now // 60 * 60)
    b1 = "AA:BB:CC:DD:EE:FF"
    b2 = "11:22:33:44:55:66"
    counts = {"beacon": 0, "deauth": 0, "disassoc": 0}

    def insert(ts, subtype, bssid, src="", dst="", reason=None, note="seed"):
        nonlocal counts
        with conn:
            conn.execute(
                "INSERT INTO wifi_frames(ts, subtype, bssid, src, dst, channel, rssi, reason, notes) VALUES (?,?,?,?,?,?,?,?,?)",
                (float(ts), subtype, bssid, src, dst, None, None, reason, note),
            )
        if subtype in counts:
            counts[subtype] += 1

    # Window -120s (win-120)
    for i in range(2):
        insert(win - 120 + i * 10, "deauth", b1)
    insert(win - 115, "disassoc", b1)

    # Window -60s (moderate activity)
    for i in range(3):
        insert(win - 60 + i * 10, "deauth", b1)
    insert(win - 45, "disassoc", b1)

    # Window 0s (burst)
    for i in range(7):
        insert(win + i * 5, "deauth", b1)
    for i in range(2):
        insert(win + 5 + i * 15, "disassoc", b1)

    # Second BSSID low-level noise across windows
    for wstart in (win - 120, win - 60, win):
        insert(wstart + 12, "deauth", b2)
        insert(wstart + 25, "disassoc", b2)

    return counts


def compute_anomalies(
    db_path: Path,
    window_sec: int = 60,
    deauth_thresh: int = 5,
    disassoc_thresh: int = 5,
    min_count: int = 3,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
):
    """Compute per-BSSID deauth/disassoc anomalies in sliding windows.

    Returns list of dicts with keys: bssid, window_start, deauth, disassoc, total, flags, score.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    q = "SELECT ts, subtype, bssid FROM wifi_frames WHERE subtype IN ('deauth','disassoc')"
    params: list = []
    if start_ts is not None:
        q += " AND ts >= ?"
        params.append(float(start_ts))
    if end_ts is not None:
        q += " AND ts <= ?"
        params.append(float(end_ts))
    q += " ORDER BY ts ASC"
    cur.execute(q, params)
    rows = cur.fetchall()
    buckets: dict[tuple[str, int], dict[str, int]] = {}
    for ts, subtype, bssid in rows:
        bssid = bssid or "unknown"
        win = int(float(ts) // window_sec * window_sec)
        key = (bssid, win)
        agg = buckets.setdefault(key, {"deauth": 0, "disassoc": 0})
        if subtype == "deauth":
            agg["deauth"] += 1
        else:
            agg["disassoc"] += 1
    out = []
    for (bssid, win), counts in sorted(buckets.items(), key=lambda kv: kv[0][1]):
        de = counts.get("deauth", 0)
        di = counts.get("disassoc", 0)
        total = de + di
        if total < min_count:
            continue
        flag_de = de >= deauth_thresh
        flag_di = di >= disassoc_thresh
        score = (de / max(deauth_thresh, 1)) + (di / max(disassoc_thresh, 1))
        out.append(
            {
                "bssid": bssid,
                "window_start": win,
                "deauth": de,
                "disassoc": di,
                "total": total,
                "flag_deauth": flag_de,
                "flag_disassoc": flag_di,
                "score": round(score, 2),
            }
        )
    return out


def kismet_ndjson_import(ndjson_path: Path, db_path: Path) -> int:
    """Best-effort import of Kismet NDJSON alert lines mapping to wifi_frames.

    Looks for keys indicating DEAUTH/DISASSOC and inserts with minimal fields.
    """
    conn = _init_db(db_path)
    inserted = 0
    with open(ndjson_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Common patterns (may vary by Kismet version/fieldset)
            alert = obj.get("kismet.alert") or obj.get("alert") or {}
            atype = (
                alert.get("kismet.alert.type")
                or alert.get("type")
                or obj.get("kismet.alert.type")
                or obj.get("type")
            )
            if not atype:
                continue
            atype_up = str(atype).upper()
            subtype = None
            if "DEAUTH" in atype_up:
                subtype = "deauth"
            elif "DISASSOC" in atype_up:
                subtype = "disassoc"
            else:
                continue
            ts = (
                obj.get("kismet.timestamp")
                or alert.get("kismet.alert.timestamp")
                or obj.get("ts")
                or obj.get("timestamp")
            )
            try:
                ts_val = float(ts)
            except Exception:
                ts_val = 0.0
            bssid = (
                alert.get("kismet.alert.bssid")
                or obj.get("bssid")
                or obj.get("dot11.device")
                or ""
            )
            src = alert.get("source") or obj.get("source") or ""
            dst = alert.get("dest") or obj.get("dest") or ""
            reason = alert.get("reason_code") or obj.get("reason_code")
            try:
                reason_val = int(reason) if reason is not None else None
            except Exception:
                reason_val = None
            with conn:
                conn.execute(
                    "INSERT INTO wifi_frames(ts, subtype, bssid, src, dst, channel, rssi, reason, notes) VALUES (?,?,?,?,?,?,?,?,?)",
                    (ts_val, subtype, str(bssid), str(src), str(dst), None, None, reason_val, "kismet"),
                )
            inserted += 1
    return inserted
