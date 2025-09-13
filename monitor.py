import argparse
import os
import time
from pathlib import Path
from typing import Optional

from sonar.detector import (
    list_audio_devices,
    run_sonar_scan,
    export_events_csv,
    analyze_wav_file,
    watch_sonar,
)
import shutil
from wlan.analyzer import analyze_pcap_to_db, live_capture_and_analyze, compute_anomalies, export_wifi_frames, kismet_ndjson_import, seed_demo_wifi
from rtl.rtl433 import run_rtl433, export_rtl433
from aegis.evidence import package_evidence
from aegis.correlate import correlate
from aegis.report import generate_html_report
from aegis.ai_analyst import generate_ai_incident_report, AnalystConfig
from aegis.llm import summarize_markdown, LLMConfig, quick_test as llm_quick_test, prefer_and_summarize
from aegis.chat import answer_question, ChatConfig
from aegis.report_store import llm_call_log as _llm_log
from aegis.report_store import store_ai_report, append_reports_manifest, llm_call_log
from aegis.audio_classifier import classify_audio_file, AudioClassifyConfig
from aegis.packet_classifier import classify_pcap_packets, PacketClassifyConfig
import sqlite3
from aegis.offload import offload_postgres
def _auto_pick_audiobox() -> str | None:
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        return None
    try:
        devices = sd.query_devices()
        picks = []
        for idx, info in enumerate(devices):
            name = str(info.get("name", ""))
            max_in = int(info.get("max_input_channels", 0))
            if max_in > 0 and ("audiobox" in name.lower() or "usb96" in name.lower()):
                picks.append((idx, name))
        if picks:
            # Return index as string for compatibility
            return str(picks[0][0])
    except Exception:
        return None
    return None
import hashlib
import glob



def ensure_data_dir(db_path: Path):
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)


def cmd_sonar(args: argparse.Namespace):
    if args.list:
        list_audio_devices()
        return
    db = Path(args.db)
    ensure_data_dir(db)
    band = None
    if args.band:
        try:
            lo, hi = args.band.split(":")
            band = (float(lo), float(hi))
        except Exception:
            raise SystemExit("--band must be like low:high in Hz, e.g. 5000:20000")
    # Prefer explicit device; otherwise try to auto-pick Audiobox/USB96
    dev = args.device or _auto_pick_audiobox()
    run_sonar_scan(
        db_path=db,
        duration_sec=args.duration,
        samplerate=args.rate,
        channels=args.channels,
        device=dev,
        band_hz=band,
        plot=args.plot,
        chirp=args.chirp,
        ultra_threshold_hz=args.ultra_thresh,
        save_audio=args.save_audio,
        demod_out=Path(args.demod_out) if args.demod_out else None,
        carrier_band=(tuple(map(float, args.carrier_band.split(":"))) if args.carrier_band else None),
    )


def cmd_export(args: argparse.Namespace):
    export_events_csv(Path(args.db), Path(args.out))
    print(f"Exported to {args.out}")


def cmd_wav(args: argparse.Namespace):
    db = Path(args.db)
    ensure_data_dir(db)
    band = None
    if args.band:
        try:
            lo, hi = args.band.split(":")
            band = (float(lo), float(hi))
        except Exception:
            raise SystemExit("--band must be like low:high in Hz, e.g. 5000:20000")
    analyze_wav_file(
        wav_path=Path(args.in_file),
        db_path=db,
        band_hz=band,
        plot=args.plot,
        chirp=args.chirp,
        denoise=args.denoise,
        noise_percentile=args.noise_percentile,
        psd_out=Path(args.psd_out) if args.psd_out else None,
        demod_out=Path(args.demod_out) if args.demod_out else None,
        carrier_band=(tuple(map(float, args.carrier_band.split(":"))) if args.carrier_band else None),
    )


def cmd_wifi(args: argparse.Namespace):
    db = Path(args.db)
    ensure_data_dir(db)
    if shutil.which("tshark") is None:
        raise SystemExit("tshark not found. Install Wireshark/tshark and ensure it's on PATH.")
    if getattr(args, "pcap", None):
        counts = analyze_pcap_to_db(Path(args.pcap), db)
    else:
        counts = live_capture_and_analyze(args.iface, float(args.duration), db)
    print(f"Wi-Fi summary: beacons={counts.get('beacon',0)} deauth={counts.get('deauth',0)} disassoc={counts.get('disassoc',0)}")


def cmd_rtl433(args: argparse.Namespace):
    db = Path(args.db)
    ensure_data_dir(db)
    if shutil.which("rtl_433") is None:
        raise SystemExit("rtl_433 not found on PATH. Install rtl_433 to use this command.")
    n = run_rtl433(float(args.duration), db, device_index=args.device)
    print(f"rtl_433: inserted {n} messages")


def cmd_wifi_report(args: argparse.Namespace):
    findings = compute_anomalies(
        Path(args.db),
        window_sec=int(args.window),
        deauth_thresh=int(args.deauth_thresh),
        disassoc_thresh=int(args.disassoc_thresh),
        min_count=int(args.min_count),
        start_ts=float(args.start_ts) if args.start_ts else None,
        end_ts=float(args.end_ts) if args.end_ts else None,
    )
    if not findings:
        print("No anomalies detected with current thresholds.")
        return
    print("Anomaly report (per BSSID per window):")
    for rec in findings:
        flags = []
        if rec["flag_deauth"]:
            flags.append("DEAUTH")
        if rec["flag_disassoc"]:
            flags.append("DISASSOC")
        print(
            f"{rec['bssid']} @ {rec['window_start']}: deauth={rec['deauth']} disassoc={rec['disassoc']} total={rec['total']} score={rec['score']} flags={'+'.join(flags) if flags else '-'}"
        )


def cmd_export_wifi(args: argparse.Namespace):
    n = export_wifi_frames(
        Path(args.db),
        Path(args.out),
        start_ts=float(args.start_ts) if args.start_ts else None,
        end_ts=float(args.end_ts) if args.end_ts else None,
    )
    print(f"Exported {n} Wi‑Fi rows to {args.out}")


def cmd_export_rtl(args: argparse.Namespace):
    n = export_rtl433(
        Path(args.db),
        Path(args.out),
        start_ts=float(args.start_ts) if args.start_ts else None,
        end_ts=float(args.end_ts) if args.end_ts else None,
    )
    print(f"Exported {n} rtl_433 rows to {args.out}")


def cmd_audio_classify(args: argparse.Namespace):
    wav = Path(args.wav)
    cfg = AudioClassifyConfig(model=args.model, top_k=int(args.top_k), backend=args.backend, timeout=int(args.timeout))
    try:
        preds = classify_audio_file(wav, cfg)
    except Exception as e:
        msg = str(e)
        print("Audio classification failed:", msg)
        print("Tips: set HF_API_TOKEN or HF_TOKEN in your environment for Hugging Face Inference, or install 'transformers' for local backend.")
        return
    # Print concise table
    for i, p in enumerate(preds, 1):
        label = str(p.get("label", "?"))
        score = float(p.get("score", 0.0))
        print(f"[{i}] {label:30} {score:0.3f}")
    # Optional JSON output
    if getattr(args, "out", None):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        out.write_text(_json.dumps(preds, indent=2), encoding="utf-8")
        # Append to reports manifest for traceability
        try:
            from hashlib import sha256 as _sha
            append_reports_manifest(Path("data"), "audio-classify", out, _sha(out.read_bytes()).hexdigest(), meta={"wav": str(wav), "model": args.model, "backend": args.backend})
        except Exception:
            pass
        print(f"Saved predictions to {out}")


def cmd_packet_classify(args: argparse.Namespace):
    pcap = Path(args.pcap)
    cfg = PacketClassifyConfig(model=args.model, top_k=int(args.top_k), timeout=int(args.timeout), retries=int(args.retries), tshark_filter=args.filter or "", max_packets=int(args.max_packets), backend=getattr(args, "backend", "auto"))
    outputs = classify_pcap_packets(pcap, cfg)
    # Print concise per-packet summary with top-1 label
    for item in outputs:
        preds = item.get("predictions", [])
        top = preds[0] if preds else {"label": "?", "score": 0.0}
        print(f"{item.get('summary','?')} -> {top.get('label')} ({float(top.get('score',0.0)):.3f})")
    if getattr(args, "out", None):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        out.write_text(_json.dumps(outputs, indent=2), encoding="utf-8")
        try:
            from hashlib import sha256 as _sha
            append_reports_manifest(Path("data"), "packet-classify", out, _sha(out.read_bytes()).hexdigest(), meta={"pcap": str(pcap), "model": args.model})
        except Exception:
            pass
        print(f"Saved packet predictions to {out}")


def cmd_wifi_kismet_import(args: argparse.Namespace):
    n = kismet_ndjson_import(Path(args.in_file), Path(args.db))
    print(f"Imported {n} events from Kismet NDJSON")


def cmd_wifi_seed(args: argparse.Namespace):
    db = Path(args.db)
    ensure_data_dir(db)
    counts = seed_demo_wifi(db)
    print(
        f"Seeded demo Wi‑Fi data: beacons={counts.get('beacon',0)} deauth={counts.get('deauth',0)} disassoc={counts.get('disassoc',0)}"
    )


def cmd_package_evidence(args: argparse.Namespace):
    manifest = package_evidence(Path(args.db), Path(args.out_dir))
    print(f"Wrote manifest: {manifest}")


def cmd_correlate(args: argparse.Namespace):
    items = correlate(
        Path(args.db),
        tolerance=float(args.tolerance),
        include_raw_wifi=bool(getattr(args, "include_raw_wifi", False)),
        wifi_window_sec=int(getattr(args, "window", 60)),
        wifi_deauth_thresh=int(getattr(args, "deauth_thresh", 5)),
        wifi_disassoc_thresh=int(getattr(args, "disassoc_thresh", 5)),
        wifi_min_count=int(getattr(args, "min_count", 3)),
        start_ts=float(args.start_ts) if getattr(args, "start_ts", None) else None,
        end_ts=float(args.end_ts) if getattr(args, "end_ts", None) else None,
    )
    if not items:
        print("No cross-source clusters found with given tolerance.")
        return
    print("Cross-source clusters:")
    for i, c in enumerate(items, 1):
        print(f"[{i}] {c['start_ts']}..{c['end_ts']} sources={','.join(c['sources'])} count={c['count']}")


def cmd_report(args: argparse.Namespace):
    out = generate_html_report(
        Path(args.db),
        Path(args.out),
        wifi_window_sec=int(args.window),
        wifi_deauth_thresh=int(args.deauth_thresh),
        wifi_disassoc_thresh=int(args.disassoc_thresh),
        wifi_min_count=int(args.min_count),
        start_ts=float(args.start_ts) if args.start_ts else None,
        end_ts=float(args.end_ts) if args.end_ts else None,
        corr_tolerance=float(args.tolerance),
        corr_include_raw_wifi=bool(getattr(args, "corr_include_raw_wifi", False)),
    )
    print(f"Wrote report: {out}")
    # Also archive a timestamped copy under data/reports/YYYYMMDD/
    try:
        tsd = time.strftime("%Y%m%d")
        tsf = time.strftime("%H%M%S")
        arch_dir = Path("data") / "reports" / tsd
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch_path = arch_dir / f"report_{tsf}.html"
        from shutil import copyfile as _copy
        _copy(out, arch_path)
        from hashlib import sha256 as _sha
        from aegis.report_store import append_reports_manifest
        append_reports_manifest(Path("data"), "report", arch_path, _sha(arch_path.read_bytes()).hexdigest(), meta={"source": str(out)})
    except Exception:
        pass


def cmd_ai_report(args: argparse.Namespace):
    db = Path(args.db)
    ensure_data_dir(db)
    out = Path(args.out)
    cfg = AnalystConfig(
        tolerance=float(args.tolerance),
        include_raw_wifi=bool(getattr(args, "include_raw_wifi", False)),
        wifi_window_sec=int(args.window),
        wifi_deauth_thresh=int(args.deauth_thresh),
        wifi_disassoc_thresh=int(args.disassoc_thresh),
        wifi_min_count=int(args.min_count),
    )
    start_ts = float(args.start_ts) if args.start_ts else None
    end_ts = float(args.end_ts) if args.end_ts else None
    path = generate_ai_incident_report(db, out, start_ts=start_ts, end_ts=end_ts, cfg=cfg)
    print(f"Wrote AI analyst report: {path}")
    # Store base report in DB and manifest
    try:
        row_id = store_ai_report(db, kind="base", path=path)
        from hashlib import sha256
        digest = sha256(Path(path).read_bytes()).hexdigest()
        append_reports_manifest(Path("data"), "base", path, digest, meta={"row_id": row_id})
    except Exception as e:
        print(f"[ai-report] warning: failed to store base report: {e}")

    # Persist correlation clusters and Wi‑Fi anomalies snapshots for the same window
    try:
        from aegis.correlate import correlate
        from wlan.analyzer import compute_anomalies
        ts_label = time.strftime("%Y%m%d_%H%M%S")
        clusters = correlate(
            Path(args.db),
            tolerance=float(args.tolerance),
            include_raw_wifi=bool(getattr(args, "include_raw_wifi", False)),
            wifi_window_sec=int(args.window),
            wifi_deauth_thresh=int(args.deauth_thresh),
            wifi_disassoc_thresh=int(args.disassoc_thresh),
            wifi_min_count=int(args.min_count),
            start_ts=start_ts,
            end_ts=end_ts,
        )
        anomalies = compute_anomalies(
            Path(args.db),
            window_sec=int(args.window),
            deauth_thresh=int(args.deauth_thresh),
            disassoc_thresh=int(args.disassoc_thresh),
            min_count=int(args.min_count),
            start_ts=start_ts,
            end_ts=end_ts,
        )
        snap_dir = Path("data") / "reports"
        snap_dir.mkdir(parents=True, exist_ok=True)
        import json
        cl_file = snap_dir / f"clusters_{ts_label}.json"
        an_file = snap_dir / f"anomalies_{ts_label}.json"
        cl_file.write_text(json.dumps(clusters, separators=(",", ":")), encoding="utf-8")
        an_file.write_text(json.dumps(anomalies, separators=(",", ":")), encoding="utf-8")
        from hashlib import sha256 as _sha
        append_reports_manifest(Path("data"), "clusters", cl_file, _sha(cl_file.read_bytes()).hexdigest(), meta={"count": len(clusters)})
        append_reports_manifest(Path("data"), "anomalies", an_file, _sha(an_file.read_bytes()).hexdigest(), meta={"count": len(anomalies)})
    except Exception as e:
        print(f"[ai-report] warning: failed to persist snapshots: {e}")
    # Archive base report with timestamped copy to avoid overwrites
    try:
        tsd = time.strftime("%Y%m%d")
        tsf = time.strftime("%H%M%S")
        arch_dir = Path("data") / "reports" / tsd
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch_path = arch_dir / f"ai_report_{tsf}.md"
        from shutil import copyfile as _copy
        _copy(path, arch_path)
        from hashlib import sha256 as _sha
        append_reports_manifest(Path("data"), "base-archived", arch_path, _sha(arch_path.read_bytes()).hexdigest(), meta={"source": str(path)})
    except Exception:
        pass

    # Optional LLM post-processing
    if getattr(args, "use_llm", False):
        md = Path(path).read_text(encoding="utf-8")
        prov = str(getattr(args, "llm_provider", "auto"))
        model = str(getattr(args, "llm_model", ""))
        # Decide summary text
        if prov == "auto":
            out_md, used_provider, used_model = prefer_and_summarize(md)
            llm_call_log(Path("data"), {"provider": used_provider, "model": used_model, "bytes": len(md)})
        else:
            out_md = summarize_markdown(md, LLMConfig(provider=prov, model=model or ("gemini-2.0-pro" if prov=="gemini" else "gpt-4.1")))
            used_provider, used_model = prov, (model or ("gemini-2.0-pro" if prov=="gemini" else "gpt-4.1"))
            llm_call_log(Path("data"), {"provider": used_provider, "model": used_model, "bytes": len(md)})
        # Write separate summary file next to the base report
        summary_path = Path(path).with_name(Path(path).stem + "_summary.md")
        summary_path.write_text(out_md, encoding="utf-8")
        print(f"Refined via LLM provider={used_provider} model={used_model} -> {summary_path.name}")
        # Store summary in DB and manifest
        try:
            row_id = store_ai_report(db, kind="summary", path=summary_path, provider=used_provider, model=used_model)
            from hashlib import sha256
            digest = sha256(summary_path.read_bytes()).hexdigest()
            append_reports_manifest(Path("data"), "summary", summary_path, digest, meta={"row_id": row_id, "provider": used_provider, "model": used_model})
        except Exception as e:
            print(f"[ai-report] warning: failed to store summary report: {e}")
        # Archive summary with timestamped copy
        try:
            tsd = time.strftime("%Y%m%d")
            tsf = time.strftime("%H%M%S")
            arch_dir = Path("data") / "reports" / tsd
            arch_dir.mkdir(parents=True, exist_ok=True)
            arch_path = arch_dir / f"ai_report_summary_{tsf}.md"
            from shutil import copyfile as _copy
            _copy(summary_path, arch_path)
            from hashlib import sha256 as _sha
            append_reports_manifest(Path("data"), "summary-archived", arch_path, _sha(arch_path.read_bytes()).hexdigest(), meta={"source": str(summary_path), "provider": used_provider, "model": used_model})
        except Exception:
            pass


def cmd_watch(args: argparse.Namespace):
    """Continuous audio watch that triggers Wi‑Fi ingest and report/evidence updates."""
    db = Path(args.db)
    ensure_data_dir(db)
    pcap_dir = Path(args.pcap_dir) if args.pcap_dir else None
    out_root = Path(args.out_root) if args.out_root else db.parent
    reports_dir = out_root / "reports" / time.strftime("%Y%m%d")
    evidence_dir = out_root / "evidence"
    reports_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def on_trigger(_info: dict):
        # Ingest any pcaps once per trigger
        if pcap_dir and pcap_dir.exists():
            _ingest_pcaps_once(db, pcap_dir, out_root / "state" / "processed_pcaps.txt", out_root / "archive" / "pcaps" / time.strftime("%Y%m%d"))
        # Write a quick report and evidence package
        out = reports_dir / f"report_watch_{time.strftime('%H%M%S')}.html"
        generate_html_report(db, out)
        package_evidence(db, evidence_dir)
        print(f"[watch] updated report {out} and evidence package")

    # Prefer explicit device; otherwise auto-pick Audiobox if present
    watch_device = args.device or _auto_pick_audiobox()
    watch_sonar(
        db_path=db,
        samplerate=int(args.rate),
        device=watch_device,
        band_hz=(tuple(map(float, args.band.split(":"))) if args.band else None),
        window_sec=float(args.window),
        check_interval_sec=float(args.check),
        snr_min_db=float(args.snr_min),
        ultra_threshold_hz=float(args.ultra_thresh),
        ultra_ratio_min=float(args.ultra_min),
        chirp=args.chirp,
        cooldown_sec=float(args.cooldown),
        on_trigger=on_trigger,
    )


def cmd_health(args: argparse.Namespace):
    print("Environment health check")
    ok = True
    # Python deps
    deps = [
        ("numpy", "import numpy as _"),
        ("scipy", "import scipy as _"),
        ("soundfile", "import soundfile as _"),
        ("sounddevice", "import sounddevice as _"),
        ("matplotlib", "import matplotlib as _"),
    ]
    import subprocess, sys
    for name, code in deps:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True)
        status = "OK" if r.returncode == 0 else "MISSING"
        ok = ok and (r.returncode == 0)
        print(f"  {name:12} {status}")
    # External tools
    print("External tools:")
    print(f"  tshark       {'OK' if shutil.which('tshark') else 'MISSING'}")
    print(f"  rtl_433      {'OK' if shutil.which('rtl_433') else 'MISSING'}")
    # Audio devices
    try:
        list_audio_devices()
    except Exception as e:
        print(f"Audio device query failed: {e}")
        ok = False
    print("Result:", "OK" if ok else "Issues detected")


def cmd_chat(args: argparse.Namespace):
    db = Path(args.db)
    ensure_data_dir(db)
    start_ts = float(args.start_ts) if args.start_ts else None
    end_ts = float(args.end_ts) if args.end_ts else None
    cfg = ChatConfig(
        max_context_chars=int(args.max_context),
        wifi_window_sec=int(args.window),
        wifi_deauth_thresh=int(args.deauth_thresh),
        wifi_disassoc_thresh=int(args.disassoc_thresh),
        wifi_min_count=int(args.min_count),
        corr_tolerance=float(args.tolerance),
        include_raw_wifi=bool(getattr(args, "include_raw_wifi", False)),
    )
    # Streaming path (only for direct LLM chat, not the context-aware mode)
    if getattr(args, "stream", False):
        # Build plain prompt via aegis.chat context for consistency
        from aegis.chat import build_context  # type: ignore
        context = build_context(db, start_ts=start_ts, end_ts=end_ts, cfg=cfg)
        prompt = context + "\n\nUser: " + args.question + "\nAssistant:"
        from aegis.llm import chat_text_stream
        used_provider = args.provider
        used_model = args.model or ("gpt-4o-mini" if used_provider == "openai" else "")
        try:
            for chunk in chat_text_stream(prompt, LLMConfig(provider=used_provider, model=used_model, max_tokens=512)):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"[stream] error: {e}")
        return
    # Default non-stream context-aware answer
    ans, used_provider, used_model, context = answer_question(
        db, args.question, provider=args.provider, model=args.model, start_ts=start_ts, end_ts=end_ts, cfg=cfg
    )
    print(ans)
    # Persist transcript if requested
    if getattr(args, "out", None):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ans, encoding="utf-8")
        print(f"Saved answer to {out}")
    # Log LLM usage if applicable
    if used_provider not in (None, "", "local"):
        try:
            _llm_log(Path("data"), {"provider": used_provider, "model": used_model, "bytes": len(context) + len(args.question)})
        except Exception:
            pass


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ingest_pcaps_once(db: Path, pcap_dir: Path, registry: Path, archive_dir: Path | None = None) -> int:
    """Ingest all pcap/pcapng files that have not been processed before.

    Tracks by sha256 in a registry file under `registry`. Optionally moves successfully ingested pcaps
    into `archive_dir` (keeps filename) to avoid reprocessing and to preserve originals.
    """
    reg = registry
    seen: set[str] = set()
    if reg.exists():
        for line in reg.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(" ", 1)
            if parts and parts[0]:
                seen.add(parts[0])
    inserted_total = 0
    pcaps = [
        Path(p)
        for ext in ("*.pcap", "*.pcapng")
        for p in glob.glob(str(pcap_dir / ext))
    ]
    for p in pcaps:
        try:
            digest = _sha256_file(p)
        except Exception:
            continue
        if digest in seen:
            continue
        try:
            counts = analyze_pcap_to_db(p, db)
            inserted = counts.get("beacon", 0) + counts.get("deauth", 0) + counts.get("disassoc", 0)
            inserted_total += inserted
            with open(reg, "a", encoding="utf-8") as rf:
                rf.write(f"{digest} {p}\n")
            print(f"Ingested pcap {p.name}: {counts}")
            if archive_dir is not None:
                try:
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    dest = archive_dir / p.name
                    # Avoid overwriting existing
                    if not dest.exists():
                        shutil.move(str(p), str(dest))
                except Exception as e:
                    print(f"Failed to archive {p}: {e}")
        except Exception as e:
            print(f"Failed to ingest {p}: {e}")
    return inserted_total


def cmd_export_jsonl(args: argparse.Namespace):
    """Export append-only JSONL streams for tables with watermark to avoid duplicates.

    Writes per-table JSONL files under --out-dir, and a state file with last exported timestamp per table.
    """
    db = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "export_state.json"
    import json as _json
    state = {}
    if state_path.exists():
        try:
            state = _json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    # Table configs: (table, ts_column, select_sql)
    tables = [
        ("sonar_events", "timestamp", "SELECT id, timestamp, samplerate, prf_hz, band_low_hz, band_high_hz, snr_db, method, notes FROM sonar_events WHERE timestamp > ? ORDER BY timestamp ASC"),
        ("wifi_frames", "ts", "SELECT ts, subtype, bssid, src, dst, channel, rssi, reason, notes FROM wifi_frames WHERE ts > ? ORDER BY ts ASC"),
        ("rtl433_messages", "ts", "SELECT ts, model, freq, data FROM rtl433_messages WHERE ts > ? ORDER BY ts ASC"),
        ("ai_reports", "timestamp", "SELECT timestamp, kind, provider, model, path, sha256, size_bytes FROM ai_reports WHERE timestamp > ? ORDER BY timestamp ASC"),
    ]
    with sqlite3.connect(db) as conn:
        for name, ts_col, sql in tables:
            last_ts = float(state.get(name, 0))
            try:
                cur = conn.execute(sql, (last_ts,))
            except sqlite3.OperationalError:
                continue
            out_path = out_dir / f"{name}.jsonl"
            max_ts = last_ts
            with out_path.open("ab") as f:
                for row in cur:
                    rec = {k: v for k, v in zip([c[0] for c in cur.description], row)}
                    # Serialize JSON-safe
                    line = _json.dumps(rec, separators=(",", ":"))
                    f.write(line.encode("utf-8") + b"\n")
                    tsv = float(rec.get(ts_col, 0) or 0)
                    if tsv > max_ts:
                        max_ts = tsv
            if max_ts > last_ts:
                state[name] = max_ts
    state_path.write_text(_json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Exported JSONL to {out_dir} (state updated)")


def cmd_collect(args: argparse.Namespace):
    db = Path(args.db)
    ensure_data_dir(db)
    end_ts = time.time() + max(0, int(args.minutes)) * 60
    iter_no = 0
    did_rtl = False
    # Organization roots
    out_root = Path(args.out_root) if args.out_root else db.parent
    run_id = time.strftime("%Y%m%d_%H%M%S")
    reports_dir = (out_root / "reports" / time.strftime("%Y%m%d")) if args.organize else None
    state_dir = (out_root / "state") if args.organize else None
    logs_dir = (out_root / "logs") if args.organize else None
    evidence_run_dir = (out_root / "evidence" / f"run_{run_id}") if args.organize else None
    if args.organize:
        # Ensure dirs exist
        for d in [reports_dir, state_dir, logs_dir, evidence_run_dir]:
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)
    # Registry for processed pcaps
    pcap_registry = (state_dir / "processed_pcaps.txt") if args.organize else (db.parent / "processed_pcaps.txt")
    # Setup log file
    log_fh = None
    if logs_dir is not None:
        try:
            log_fh = open(logs_dir / f"collect_{run_id}.log", "a", encoding="utf-8")
        except Exception:
            log_fh = None
    while True:
        iter_no += 1
        print(f"[collect] iteration {iter_no}")
        if log_fh:
            log_fh.write(f"iter={iter_no} ts={time.time():.0f} start\n")
        # Sonar
        if not args.no_sonar:
            band = None
            if args.sonar_band:
                try:
                    lo, hi = args.sonar_band.split(":")
                    band = (float(lo), float(hi))
                except Exception:
                    print("--sonar-band must be low:high Hz (e.g., 5000:20000)")
            # Prefer explicit device; otherwise auto-pick Audiobox if present
            sonar_dev = args.sonar_device or _auto_pick_audiobox()
            run_sonar_scan(
                db_path=db,
                duration_sec=float(args.sonar_duration),
                samplerate=int(args.sonar_rate),
                channels=1,
                device=sonar_dev,
                band_hz=band,
                plot=False,
                chirp=args.sonar_chirp,
                ultra_threshold_hz=float(args.ultra_thresh),
                save_audio=bool(args.save_audio),
            )
        # PCAP ingest
        if args.pcap_dir:
            pdir = Path(args.pcap_dir)
            if pdir.exists():
                # Optional archive directory by date
                arch_dir = None
                if args.archive_pcaps and args.organize:
                    arch_dir = out_root / "archive" / "pcaps" / time.strftime("%Y%m%d")
                added = _ingest_pcaps_once(db, pdir, pcap_registry, arch_dir)
                if log_fh:
                    log_fh.write(f"iter={iter_no} pcaps_ingested={added}\n")
        # rtl_433 (once per session)
        if (not did_rtl) and args.rtl_duration and shutil.which("rtl_433"):
            try:
                n = run_rtl433(float(args.rtl_duration), db)
                print(f"rtl_433 captured {n} messages")
            except Exception as e:
                print(f"rtl_433 failed: {e}")
            did_rtl = True
        # On-iteration reporting/packaging if requested
        if args.periodic_pack:
            out_report = (
                (reports_dir / f"report_{iter_no:03d}_{time.strftime('%H%M%S')}.html")
                if args.organize and reports_dir is not None
                else Path(args.report_out)
            )
            generate_html_report(
                db,
                out_report,
                wifi_window_sec=int(args.window),
                wifi_deauth_thresh=int(args.deauth_thresh),
                wifi_disassoc_thresh=int(args.disassoc_thresh),
                wifi_min_count=int(args.min_count),
                corr_tolerance=float(args.tolerance),
                corr_include_raw_wifi=bool(getattr(args, "corr_include_raw_wifi", False)),
            )
            print(f"[collect] wrote report: {out_report}")
            pack_dir = (
                evidence_run_dir / f"iter_{iter_no:03d}"
                if args.organize and evidence_run_dir is not None
                else Path(args.evidence_dir)
            )
            manifest = package_evidence(db, pack_dir)
            print(f"[collect] wrote evidence manifest: {manifest}")
            if log_fh:
                log_fh.write(f"iter={iter_no} report={out_report} manifest={manifest}\n")

        # Loop control
        if time.time() >= end_ts or int(args.minutes) == 0:
            break
        # Sleep until next minute tick
        time.sleep(max(1, int(args.interval)))
    # After collection: final report + evidence (if not already done periodically)
    if not args.periodic_pack:
        out_report = (
            (reports_dir / f"report_final_{time.strftime('%H%M%S')}.html")
            if args.organize and reports_dir is not None
            else Path(args.report_out)
        )
        generate_html_report(
            db,
            out_report,
            wifi_window_sec=int(args.window),
            wifi_deauth_thresh=int(args.deauth_thresh),
            wifi_disassoc_thresh=int(args.disassoc_thresh),
            wifi_min_count=int(args.min_count),
            corr_tolerance=float(args.tolerance),
            corr_include_raw_wifi=bool(getattr(args, "corr_include_raw_wifi", False)),
        )
        print(f"Wrote report: {out_report}")
        pack_dir = (
            evidence_run_dir / "final"
            if args.organize and evidence_run_dir is not None
            else Path(args.evidence_dir)
        )
        manifest = package_evidence(db, pack_dir)
        print(f"Wrote evidence manifest: {manifest}")
    if log_fh:
        log_fh.write("done\n")
        log_fh.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="monitor", description="Passive sonar detection (safe & compliant)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sonar", help="Run a sonar scan")
    ps.add_argument("--list", action="store_true", help="List audio devices and exit")
    ps.add_argument("--db", default=str(Path("data") / "monitor.db"))
    ps.add_argument("--duration", type=float, default=10.0, help="Capture duration in seconds")
    ps.add_argument("--rate", type=int, default=48000, help="Sample rate (Hz)")
    ps.add_argument("--channels", type=int, default=1, help="Num channels (1=mono)")
    ps.add_argument("--device", type=str, default=None, help="Sound device index or name (optional)")
    ps.add_argument("--band", type=str, default=None, help="Analysis band low:high in Hz (optional)")
    ps.add_argument("--plot", action="store_true", help="Show a quick spectrogram window")
    ps.add_argument(
        "--chirp",
        type=str,
        default=None,
        help="Optional matched linear chirp template f0:f1:dur_ms (e.g., 5000:15000:5)",
    )
    ps.add_argument("--ultra-thresh", type=float, default=18000.0, help="Ultrasonic threshold in Hz for energy ratio")
    ps.add_argument("--save-audio", action="store_true", help="Save captured audio WAV and include SHA-256 in event notes")
    ps.add_argument("--demod-out", type=str, default=None, help="If set and sampling rate is sufficient, write demodulated baseband audio WAV here")
    ps.add_argument("--carrier-band", type=str, default=None, help="Carrier band low:high Hz for demodulation (e.g., 30000:50000 for ~40 kHz)")
    ps.set_defaults(func=cmd_sonar)

    pe = sub.add_parser("export", help="Export logged events to CSV")
    pe.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pe.add_argument("--out", required=True, help="Output CSV path")
    pe.set_defaults(func=cmd_export)

    pw = sub.add_parser("wav", help="Analyze a WAV file offline")
    pw.add_argument("--in", dest="in_file", required=True, help="Input WAV file path")
    pw.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pw.add_argument("--band", type=str, default=None, help="Analysis band low:high in Hz (optional)")
    pw.add_argument("--plot", action="store_true", help="Show spectrogram")
    pw.add_argument(
        "--chirp",
        type=str,
        default=None,
        help="Optional matched linear chirp template f0:f1:dur_ms",
    )
    pw.add_argument(
        "--denoise",
        type=str,
        choices=["none", "specsub"],
        default="specsub",
        help="Noise reduction method",
    )
    pw.add_argument(
        "--noise-percentile",
        type=float,
        default=20.0,
        help="Percentile for noise floor estimate (0-100)",
    )
    pw.add_argument(
        "--psd-out",
        type=str,
        default=None,
        help="Optional CSV path to write average PSD profile",
    )
    pw.add_argument("--demod-out", type=str, default=None, help="If set and WAV sample rate is sufficient, write demodulated baseband audio here")
    pw.add_argument("--carrier-band", type=str, default=None, help="Carrier band low:high Hz for demodulation (e.g., 30000:50000)")
    pw.set_defaults(func=cmd_wav)

    # Wi-Fi via tshark
    pwifi = sub.add_parser("wifi", help="Analyze Wi-Fi pcap or run short live capture (requires tshark)")
    pwifi.add_argument("--db", default=str(Path("data") / "monitor.db"))
    src = pwifi.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", type=str, help="Path to pcap/pcapng to analyze")
    src.add_argument("--iface", type=str, help="Interface for live capture (monitor mode)")
    pwifi.add_argument("--duration", type=float, default=10.0, help="Live capture duration (s)")
    pwifi.set_defaults(func=cmd_wifi)

    # Wi‑Fi anomaly report
    preport = sub.add_parser("wifi-report", help="Compute Wi‑Fi deauth/disassoc anomalies per BSSID/time window")
    preport.add_argument("--db", default=str(Path("data") / "monitor.db"))
    preport.add_argument("--window", type=int, default=60, help="Window size seconds")
    preport.add_argument("--deauth-thresh", type=int, default=5, help="Deauth threshold per window")
    preport.add_argument("--disassoc-thresh", type=int, default=5, help="Disassoc threshold per window")
    preport.add_argument("--min-count", type=int, default=3, help="Minimum total frames to consider a window")
    preport.add_argument("--start-ts", type=float, default=None)
    preport.add_argument("--end-ts", type=float, default=None)
    preport.set_defaults(func=cmd_wifi_report)

    # Exports
    pew = sub.add_parser("export-wifi", help="Export wifi_frames to CSV")
    pew.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pew.add_argument("--out", required=True)
    pew.add_argument("--start-ts", type=float, default=None)
    pew.add_argument("--end-ts", type=float, default=None)
    pew.set_defaults(func=cmd_export_wifi)

    per = sub.add_parser("export-rtl433", help="Export rtl_433 rows to CSV")
    per.add_argument("--db", default=str(Path("data") / "monitor.db"))
    per.add_argument("--out", required=True)
    per.add_argument("--start-ts", type=float, default=None)
    per.add_argument("--end-ts", type=float, default=None)
    per.set_defaults(func=cmd_export_rtl)

    # Audio classification (Hugging Face / local transformers)
    pac = sub.add_parser("audio-classify", help="Classify an audio WAV using Hugging Face or local transformers")
    pac.add_argument("--wav", required=True, help="Path to WAV file")
    pac.add_argument("--model", default="superb/wav2vec2-base-superb-ks", help="HF model id or local model path")
    pac.add_argument("--backend", default="auto", choices=["auto", "hf", "local"], help="Select backend")
    pac.add_argument("--top-k", type=int, default=5)
    pac.add_argument("--timeout", type=int, default=30)
    pac.add_argument("--out", type=str, default="", help="Optional JSON output path for predictions")
    pac.set_defaults(func=cmd_audio_classify)

    # Packet classification (Hugging Face text classifier on header/payload tokens)
    ppc = sub.add_parser("packet-classify", help="Classify packets from pcap using a HF packet model")
    ppc.add_argument("--pcap", required=True, help="Path to pcap/pcapng")
    ppc.add_argument("--model", default="rdpahalavan/bert-network-packet-flow-header-payload")
    ppc.add_argument("--backend", default="auto", choices=["auto", "hf", "local"], help="Select HF remote or local transformers backend")
    ppc.add_argument("--top-k", type=int, default=5)
    ppc.add_argument("--timeout", type=int, default=30)
    ppc.add_argument("--retries", type=int, default=2)
    ppc.add_argument("--max-packets", type=int, default=200, help="Max packets to process")
    ppc.add_argument("--filter", type=str, default="", help="Optional tshark display filter (-Y)")
    ppc.add_argument("--out", type=str, default="", help="Optional JSON output path for predictions")
    ppc.set_defaults(func=cmd_packet_classify)

    # Kismet import
    pk = sub.add_parser("wifi-kismet-import", help="Import Kismet NDJSON alerts into wifi_frames")
    pk.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pk.add_argument("--in", dest="in_file", required=True)
    pk.set_defaults(func=cmd_wifi_kismet_import)

    # Wi‑Fi seed demo data
    psd = sub.add_parser("wifi-seed", help="Insert synthetic Wi‑Fi frames for testing (no tshark required)")
    psd.add_argument("--db", default=str(Path("data") / "monitor.db"))
    psd.set_defaults(func=cmd_wifi_seed)

    # Evidence packaging
    pe = sub.add_parser("package-evidence", help="Export CSVs and manifest for chain-of-custody")
    pe.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pe.add_argument("--out-dir", default=str(Path("data") / "evidence"))
    pe.set_defaults(func=cmd_package_evidence)

    # Correlation
    pc = sub.add_parser("correlate", help="Find temporal clusters across sources (sonar/wifi/rtl433)")
    pc.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pc.add_argument("--tolerance", type=float, default=2.0, help="Seconds tolerance between events")
    pc.add_argument("--include-raw-wifi", action="store_true", help="Include raw Wi‑Fi frames (not just anomaly windows)")
    pc.add_argument("--window", type=int, default=60, help="Wi‑Fi anomaly window for correlation")
    pc.add_argument("--deauth-thresh", type=int, default=5)
    pc.add_argument("--disassoc-thresh", type=int, default=5)
    pc.add_argument("--min-count", type=int, default=3)
    pc.add_argument("--start-ts", type=float, default=None)
    pc.add_argument("--end-ts", type=float, default=None)
    pc.set_defaults(func=cmd_correlate)

    # rtl_433
    prtl = sub.add_parser("rtl433", help="Capture rtl_433 JSON for a duration (requires rtl_433)")
    prtl.add_argument("--db", default=str(Path("data") / "monitor.db"))
    prtl.add_argument("--duration", type=float, default=30.0, help="Duration seconds")
    prtl.add_argument("--device", type=int, default=None, help="RTL-SDR device index")
    prtl.set_defaults(func=cmd_rtl433)

    # HTML report
    prep = sub.add_parser("report", help="Generate HTML summary report")
    prep.add_argument("--db", default=str(Path("data") / "monitor.db"))
    prep.add_argument("--out", default=str(Path("data") / "report.html"))
    prep.add_argument("--window", type=int, default=60)
    prep.add_argument("--deauth-thresh", type=int, default=5)
    prep.add_argument("--disassoc-thresh", type=int, default=5)
    prep.add_argument("--min-count", type=int, default=3)
    prep.add_argument("--start-ts", type=float, default=None)
    prep.add_argument("--end-ts", type=float, default=None)
    prep.add_argument("--tolerance", type=float, default=2.0)
    prep.add_argument("--corr-include-raw-wifi", action="store_true", help="Include raw Wi‑Fi frames in correlation clusters")
    prep.set_defaults(func=cmd_report)

    # Chat about data
    pchat = sub.add_parser("chat", help="Ask a question about local data; answers use context from DB")
    pchat.add_argument("question", type=str, help="Your question")
    pchat.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pchat.add_argument("--provider", type=str, default="auto", choices=["auto", "gemini", "openai", "hf", "llamacpp", "local"], help="LLM provider (or 'local' for heuristic answer)")
    pchat.add_argument("--model", type=str, default="", help="Override model for provider")
    pchat.add_argument("--stream", action="store_true", help="Stream the model output (openai/llamacpp)")
    pchat.add_argument("--out", type=str, default="", help="Optional path to save answer text")
    pchat.add_argument("--start-ts", type=float, default=None)
    pchat.add_argument("--end-ts", type=float, default=None)
    pchat.add_argument("--max-context", type=int, default=8000)
    pchat.add_argument("--window", type=int, default=60)
    pchat.add_argument("--deauth-thresh", type=int, default=5)
    pchat.add_argument("--disassoc-thresh", type=int, default=5)
    pchat.add_argument("--min-count", type=int, default=3)
    pchat.add_argument("--tolerance", type=float, default=2.0)
    pchat.add_argument("--include-raw-wifi", action="store_true")
    pchat.set_defaults(func=cmd_chat)

    # AI Analyst Markdown report
    pai = sub.add_parser("ai-report", help="Generate a natural-language incident report (Markdown)")
    pai.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pai.add_argument("--out", default=str(Path("data") / "reports" / "ai_report.md"))
    pai.add_argument("--window", type=int, default=60)
    pai.add_argument("--deauth-thresh", type=int, default=5)
    pai.add_argument("--disassoc-thresh", type=int, default=5)
    pai.add_argument("--min-count", type=int, default=3)
    pai.add_argument("--tolerance", type=float, default=2.0)
    pai.add_argument("--include-raw-wifi", action="store_true")
    pai.add_argument("--start-ts", type=float, default=None)
    pai.add_argument("--end-ts", type=float, default=None)
    pai.add_argument("--use-llm", action="store_true", help="Post-process with an LLM (requires API key)")
    pai.add_argument("--llm-provider", type=str, default="auto", choices=["auto", "openai", "gemini", "hf", "llamacpp"], help="LLM provider")
    pai.add_argument("--llm-model", type=str, default="", help="Provider model (optional)")
    pai.set_defaults(func=cmd_ai_report)

    # LLM quick test
    pllmt = sub.add_parser("llm-test", help="Attempt a tiny request to any LLM providers with keys set in env")
    pllmt.set_defaults(func=lambda _a: print(llm_quick_test()))

    # Watch mode
    pwatch = sub.add_parser("watch", help="Continuous audio watch; on triggers, ingest pcaps and update report/evidence")
    pwatch.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pwatch.add_argument("--rate", type=int, default=48000)
    pwatch.add_argument("--device", type=str, default=None)
    pwatch.add_argument("--band", type=str, default=None)
    pwatch.add_argument("--window", type=float, default=3.0)
    pwatch.add_argument("--check", type=float, default=1.0)
    pwatch.add_argument("--snr-min", dest="snr_min", type=float, default=3.0)
    pwatch.add_argument("--ultra-thresh", type=float, default=18000.0)
    pwatch.add_argument("--ultra-min", type=float, default=0.03)
    pwatch.add_argument("--chirp", type=str, default=None)
    pwatch.add_argument("--cooldown", type=float, default=30.0)
    pwatch.add_argument("--pcap-dir", type=str, default=str(Path("data") / "pcaps"))
    pwatch.add_argument("--out-root", type=str, default=str(Path("data")))
    pwatch.set_defaults(func=cmd_watch)

    # Health check
    ph = sub.add_parser("health", help="Check Python deps, external tools, and audio devices")
    ph.set_defaults(func=cmd_health)

    # Collect all systems
    pcoll = sub.add_parser("collect", help="Run multi-source evidence collection and produce report + package")
    pcoll.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pcoll.add_argument("--minutes", type=int, default=1, help="Run for this many minutes (0=single pass)")
    pcoll.add_argument("--interval", type=int, default=60, help="Seconds to wait between iterations")
    # Sonar options
    pcoll.add_argument("--no-sonar", action="store_true", help="Skip sonar scanning")
    pcoll.add_argument("--sonar-rate", type=int, default=48000)
    pcoll.add_argument("--sonar-duration", type=float, default=5.0)
    pcoll.add_argument("--sonar-device", type=str, default=None)
    pcoll.add_argument("--sonar-band", type=str, default=None)
    pcoll.add_argument("--sonar-chirp", type=str, default=None)
    pcoll.add_argument("--ultra-thresh", type=float, default=18000.0)
    pcoll.add_argument("--save-audio", action="store_true")
    # Wi‑Fi ingest
    pcoll.add_argument("--pcap-dir", type=str, default=str(Path("data") / "pcaps"), help="Directory with .pcap/.pcapng to ingest")
    # rtl_433
    pcoll.add_argument("--rtl-duration", type=float, default=0.0, help="If >0 and rtl_433 exists, capture for this many seconds once")
    # Reporting
    pcoll.add_argument("--report-out", type=str, default=str(Path("data") / "report.html"))
    pcoll.add_argument("--evidence-dir", type=str, default=str(Path("data") / "evidence"))
    pcoll.add_argument("--window", type=int, default=60)
    pcoll.add_argument("--deauth-thresh", type=int, default=5)
    pcoll.add_argument("--disassoc-thresh", type=int, default=5)
    pcoll.add_argument("--min-count", type=int, default=3)
    pcoll.add_argument("--tolerance", type=float, default=2.0)
    pcoll.add_argument("--corr-include-raw-wifi", action="store_true")
    pcoll.add_argument("--periodic-pack", action="store_true", help="Generate report and package evidence at each iteration")
    pcoll.add_argument("--organize", action="store_true", help="Organize outputs into structured directories under --out-root")
    pcoll.add_argument("--out-root", type=str, default=str(Path("data")))
    pcoll.add_argument("--archive-pcaps", action="store_true", help="Move ingested pcaps into an archive by date under --out-root")
    pcoll.set_defaults(func=cmd_collect)

    # Export append-only JSONL (for cloud ingest)
    pjsonl = sub.add_parser("export-jsonl", help="Export append-only JSONL per table with timestamps (for cloud ingest)")
    pjsonl.add_argument("--db", default=str(Path("data") / "monitor.db"))
    pjsonl.add_argument("--out-dir", default=str(Path("data") / "exports"))
    pjsonl.set_defaults(func=cmd_export_jsonl)

    # Offload to Postgres
    po = sub.add_parser("offload-postgres", help="Offload data/exports JSONL to Postgres (idempotent upserts)")
    po.add_argument("--from-dir", default=str(Path("data") / "exports"))
    po.add_argument("--dsn", required=True, help="Postgres DSN, e.g., postgres://user:pass@host:5432/dbname")
    po.set_defaults(func=lambda a: print(offload_postgres(Path(a.from_dir), a.dsn)))

    return p


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
