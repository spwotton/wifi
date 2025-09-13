from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .correlate import correlate
from wlan.analyzer import compute_anomalies
from .llm import LLMConfig, chat_text, prefer_and_chat
from .report_store import ensure_ai_tables


@dataclass
class ChatConfig:
    max_context_chars: int = 8000
    wifi_window_sec: int = 60
    wifi_deauth_thresh: int = 5
    wifi_disassoc_thresh: int = 5
    wifi_min_count: int = 3
    corr_tolerance: float = 2.0
    include_raw_wifi: bool = False


def _query_recent_sonar(db: Path, start_ts: Optional[float], end_ts: Optional[float], limit: int = 10):
    q = "SELECT timestamp, prf_hz, band_low_hz, band_high_hz, snr_db, method, notes FROM sonar_events"
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
    args.append(int(limit))
    with sqlite3.connect(db) as conn:
        return list(conn.execute(q, tuple(args)))


def _rtl_count(db: Path, start_ts: Optional[float], end_ts: Optional[float]) -> int:
    try:
        with sqlite3.connect(db) as conn:
            cur = conn.cursor()
            if start_ts is None and end_ts is None:
                return int(cur.execute("SELECT COUNT(*) FROM rtl433_messages").fetchone()[0])
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


def _latest_ai_report_snippet(db: Path, max_chars: int = 1200) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, path) of the latest base AI report if available."""
    try:
        ensure_ai_tables(db)
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT content, path FROM ai_reports WHERE kind='base' ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None, None
            text = row[0] or ""
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... [truncated]"
            return text, row[1]
    except Exception:
        return None, None


def build_context(
    db_path: Path,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    cfg: ChatConfig = ChatConfig(),
) -> str:
    """Assemble a concise, structured context from local data for Q&A."""
    ts_range = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_ts))} .. {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_ts))}"
        if (start_ts and end_ts)
        else "recent period"
    )

    # Correlation clusters
    clusters = correlate(
        db_path,
        tolerance=float(cfg.corr_tolerance),
        include_raw_wifi=bool(cfg.include_raw_wifi),
        wifi_window_sec=int(cfg.wifi_window_sec),
        wifi_deauth_thresh=int(cfg.wifi_deauth_thresh),
        wifi_disassoc_thresh=int(cfg.wifi_disassoc_thresh),
        wifi_min_count=int(cfg.wifi_min_count),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    # Wi‑Fi anomalies
    anomalies = compute_anomalies(
        db_path,
        window_sec=int(cfg.wifi_window_sec),
        deauth_thresh=int(cfg.wifi_deauth_thresh),
        disassoc_thresh=int(cfg.wifi_disassoc_thresh),
        min_count=int(cfg.wifi_min_count),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    # Sonar + rtl433
    sonar_rows = _query_recent_sonar(db_path, start_ts, end_ts, limit=8)
    rtl_total = _rtl_count(db_path, start_ts, end_ts)
    # Last AI report
    ai_text, ai_path = _latest_ai_report_snippet(db_path)

    lines: list[str] = []
    lines.append(f"Context window: {ts_range}")
    if clusters:
        lines.append("Clusters (top 5):")
        for c in clusters[:5]:
            lines.append(
                f"- {time.strftime('%H:%M:%S', time.localtime(c['start_ts']))}–{time.strftime('%H:%M:%S', time.localtime(c['end_ts']))} src={','.join(c['sources'])} count={c['count']}"
            )
    else:
        lines.append("Clusters: none above thresholds")
    if anomalies:
        worst = max(anomalies, key=lambda a: a.get('score', 0))
        lines.append(
            f"Wi‑Fi anomalies: top window @ {worst['window_start']} score={worst['score']:.1f} deauth={worst['deauth']} disassoc={worst['disassoc']}"
        )
    else:
        lines.append("Wi‑Fi anomalies: none flagged")
    if sonar_rows:
        s = sonar_rows[0]
        lines.append(
            f"Recent sonar: ts={time.strftime('%H:%M:%S', time.localtime(s[0]))} band={int(s[2])}-{int(s[3])}Hz snr={s[4]:.1f}dB method={s[5]}"
        )
    else:
        lines.append("Recent sonar: none recorded")
    lines.append(f"rtl_433 messages: {rtl_total}")
    if ai_text:
        lines.append("\nExcerpt from last AI report:")
        lines.append(ai_text)
        lines.append(f"[source: {ai_path}]")

    ctx = "\n".join(lines)
    if len(ctx) > int(cfg.max_context_chars):
        ctx = ctx[: int(cfg.max_context_chars)] + "\n... [context truncated]"
    return ctx


def local_answer(context: str, question: str) -> str:
    """Heuristic local answer without external LLMs."""
    parts = [
        "Local Q&A (no LLM available)",
        "",
        f"Question: {question}",
        "",
        "Based on the current context:",
        context,
        "",
        "Answer:",
        "- This toolkit found the above events in the selected period.",
        "- For deeper natural-language analysis, enable --provider auto to use a local key (Gemini/OpenAI).",
    ]
    return "\n".join(parts)


def answer_question(
    db_path: Path,
    question: str,
    provider: str = "auto",
    model: str = "",
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    cfg: ChatConfig = ChatConfig(),
) -> tuple[str, str, str, str]:
    """Return (answer, used_provider, used_model, context)."""
    context = build_context(db_path, start_ts=start_ts, end_ts=end_ts, cfg=cfg)
    sys_prompt = (
        "You are a meticulous security analyst. Answer ONLY using the provided context from local data."
        " If information is insufficient, say so explicitly. Be concise and cite times/sources from context."
    )
    user_prompt = f"Question: {question}\n\nContext:\n{context}"
    used_provider = provider
    used_model = model
    if provider == "local":
        return local_answer(context, question), used_provider, used_model, context
    if provider == "auto":
        try:
            ans, used_provider, used_model = prefer_and_chat(user_prompt, system_prompt=sys_prompt)
            return ans, used_provider, used_model, context
        except Exception:
            return local_answer(context, question), "local", "", context
    # Direct provider path
    try:
        ans = chat_text(user_prompt, LLMConfig(provider=provider, model=model or ""), system_prompt=sys_prompt)
        return ans, provider, (model or ""), context
    except Exception:
        return local_answer(context, question), "local", "", context
