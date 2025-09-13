RF/Wi‑Fi/Acoustic Monitoring (Safe & Compliant)

This project offers passive monitoring across three domains:
- Ultrasonic/acoustic detection via microphone (PRF detection, chirp matching)
- Wi‑Fi management frame analysis from pcap or short live capture (tshark)
- ISM/OOK device decoding via rtl_433

All outputs log to a single SQLite database at `data/monitor.db` by default.

Scope and intent
- Passive only: recording, parsing, and analysis; no active emissions or evasion.
- Designed for situational awareness and research, not for offensive use.

Hardware requirements
- Microphone/ADC capable of desired sample rate (e.g., 48 kHz).
- Optional Wi‑Fi adapter that supports monitor mode (Linux/macOS) for live captures.
- Optional RTL‑SDR dongle for rtl_433.

Components
- monitor.py: Unified CLI.
- sonar/detector.py: Audio capture and detection + WAV analysis.
- wlan/analyzer.py: tshark-based Wi‑Fi pcap/live analyzer.
- rtl/rtl433.py: rtl_433 runner and SQLite logger.
 - aegis/evidence.py: Evidence packaging (CSV export + manifest with per-file hashes and package hash).
 - aegis/correlate.py: Cross-source temporal correlation across sonar/wifi/rtl433 events.
 - aegis/report.py: HTML report generator with embedded charts.
 - docs: `sonar/ultrasonicspeaker.md` guidance on detecting directional ultrasonic speakers (parametric arrays)

AEGIS.UltraDetect (new)
- A focused ultrasonic detection module under `aegis_ultra/` with its own minimal CLI.
- Defaults write under `data/ultra/`:
  - Captures: `data/ultra/captures/`
  - Manifests: `data/ultra/manifests/ultra_manifest.jsonl`
  - Evidence: `data/ultra/evidence/`

Quick start (UltraDetect)
- Probe audio device and rates:
  python -m aegis_ultra.cli probe
- Continuous chunked capture (5 s WAV chunks):
  python -m aegis_ultra.cli stream
- Analyze a high-rate WAV, report bands/peaks, attempt AM demod to `*_demod.wav`:
  python -m aegis_ultra.cli analyze data/captures/highrate.wav
- Append a file to the append-only manifest:
  python -m aegis_ultra.cli manifest-add data/ultra/captures/audio_20250101_120000.wav
- Verify the manifest chain:
  python -m aegis_ultra.cli manifest-verify

Install
1) Python deps:
  pip install -r requirements.txt
2) Optional external tools:
  - tshark (Wireshark CLI)
  - rtl_433 (for ISM decoding)

Quick start
1) List audio devices:
  python monitor.py sonar --list
2) Run an audio scan (10 s, 48 kHz):
  python monitor.py sonar --rate 48000 --duration 10 --band 5000:20000 --plot --ultra-thresh 18000 --save-audio
3) Wi‑Fi pcap analysis:
  python monitor.py wifi --pcap path/to/capture.pcapng
4) Wi‑Fi live capture (monitor mode required):
  python monitor.py wifi --iface wlan0 --duration 15
5) rtl_433 (30 s):
  python monitor.py rtl433 --duration 30
6) Wi‑Fi anomaly report (per BSSID per minute, default thresholds 5/5):
  python monitor.py wifi-report --window 60 --deauth-thresh 5 --disassoc-thresh 5
7) Export rows:
  python monitor.py export-wifi --out wifi.csv
  python monitor.py export-rtl433 --out rtl.csv
8) Import Kismet NDJSON alerts:
  python monitor.py wifi-kismet-import --in kismet_alerts.ndjson
9) Package evidence (CSV + manifest.json with hashes):
  python monitor.py package-evidence --out-dir data/evidence
10) Cross-source correlation (sonar/wifi/rtl433):
  python monitor.py correlate --tolerance 2.0
11) Generate HTML report (with charts if matplotlib available):
  python monitor.py report --out data/report.html
11b) Generate AI-style incident report (Markdown):
  python monitor.py ai-report --out data/reports/ai_report.md
12) Health check (deps, tools, audio devices):
  python monitor.py health
13) Continuous audio watch (auto-ingest pcaps on triggers):
  python monitor.py watch --rate 48000 --band 5000:20000 --snr-min 3 --ultra-thresh 18000 --pcap-dir data/pcaps --out-root data

Ultrasonic speaker (parametric array) workflow
- Read: sonar/ultrasonicspeaker.md for background and strategies.
- Capture strategy:
  - Ideally use hardware capable of ≥96 kHz sampling to cover ~40 kHz carriers; consumer 48 kHz will not capture 40 kHz directly.
  - If you have a high-rate interface, record at 96/192 kHz and demodulate:
    python monitor.py sonar --rate 192000 --duration 5 --carrier-band 30000:50000 --demod-out data/captures/demod.wav --save-audio
  - For offline WAVs recorded at high rate:
    python monitor.py wav --in highrate.wav --carrier-band 30000:50000 --demod-out data/captures/demod.wav --plot
  - Demod notes: We envelope-detect (Hilbert), low-pass to audio band, normalize, and resample to 16 kHz. If the sample rate can’t cover the carrier band, demod is skipped with a note.
- Presence-only detection at 48 kHz:
  - Use --band 5000:20000 and monitor ultrasonic energy ratio (--ultra-thresh) and SNR. This won’t see a 40 kHz carrier; it’s still useful for near-ultrasonic beacons (18–20 kHz).
- Watch-driven triage:
  - Leave watch running to capture triggers and auto-ingest pcaps; when access to high-rate hardware is available, perform targeted high-rate captures for demodulation.

Notes on watch mode
- The watch command continuously monitors the microphone and logs events when thresholds are exceeded (SNR/ultrasonic ratio/PRF/chirp match).
- On each trigger, it ingests any new .pcap/.pcapng files from --pcap-dir, updates the HTML report under data/reports/YYYYMMDD/, and refreshes the evidence package under data/evidence/.
- If --device is not provided, it tries to auto-pick an Audiobox/USB96 microphone when present. Otherwise, default input is used.

Windows workflow (no monitor mode)
- Live Wi‑Fi monitor mode is typically unavailable on Windows. You have three practical paths:
  1) Analyze pcaps created elsewhere (Linux/macOS or specialized tools):
    - Place .pcap/.pcapng files under data/pcaps
    - Run: python monitor.py collect --minutes 0 --no-sonar --pcap-dir data/pcaps
    - Or directly: python monitor.py wifi --pcap path\\to\\capture.pcapng
  2) Import Kismet NDJSON alerts produced elsewhere:
    - python monitor.py wifi-kismet-import --in path\\to\\alerts.ndjson
  3) Seed demo Wi‑Fi data to exercise the pipeline:
    - python monitor.py wifi-seed
    - Then run reports and exports as usual.

All-in-one collection and reporting
- Single pass (ingest pcaps, optional sonar/rtl, generate report, package evidence):
  python monitor.py collect --minutes 0 --no-sonar --pcap-dir data/pcaps --report-out data/report.html --evidence-dir data/evidence

Example outputs
- wifi-report (defaults: window=60s, thresholds 5/5):
  Anomaly report (per BSSID per window):
  AA:BB:CC:DD:EE:FF @ 1726160400: deauth=7 disassoc=1 total=8 score=2.4 flags=DEAUTH

- package-evidence:
  data/evidence/
    sonar_events.csv  (sha256: <hash>)
    wifi_frames.csv   (sha256: <hash>)
    rtl433_messages.csv (sha256: <hash>)
    manifest.json     (contains per-file sha256 and package_sha256)

- correlate (tolerance 2.0s):
  [1] 1726160501.2..1726160502.7 sources=sonar,wifi count=4

Legal/privacy scope
- Passive only. Do not perform jamming or unauthorized interception.
- Respect local laws on recording; configure modules to avoid capturing 3rd‑party content where restricted.
- Evidence package includes hashes to aid chain‑of‑custody. Add your own digital signing if needed for court-grade integrity.

Offline WAV analysis
- Denoise and PSD export:
  python monitor.py wav --in input.wav --band 5000:20000 --denoise specsub --psd-out psd.csv --plot

Sonar metrics explained
- PRF: Estimated pulse repetition frequency (Hz) of bursty patterns (if present).
- SNR: Heuristic signal-to-noise ratio estimate in dB.
- Dominant frequency (dom): Peak frequency from the power spectrum.
- Ultrasonic energy ratio (ultra): Fraction of spectral energy above the threshold (default 18 kHz). Use --ultra-thresh to adjust.
- Chirp match: If --chirp is provided (f0:f1:dur_ms), a simple matched-filter score is included.
- When --save-audio is used, a WAV is written under data/captures with a truncated SHA‑256 fingerprint included in the event notes.

HTML report contents
- Summary cards for sonar events, Wi‑Fi frames, rtl_433 messages, and cluster count.
- Recent sonar events table.
- Wi‑Fi anomalies table plus a time-series chart of anomaly scores (if matplotlib is installed).
- rtl_433 top models bar chart and a Sonar PRF histogram (if matplotlib is installed).

AI analyst report (Markdown)
- Synthesizes a human-readable incident summary across sonar/wifi/rtl_433, highlights likely coordinated activity, and suggests next steps.
- Output default: `data/reports/ai_report.md`.

Data
- `sonar_events`, `wifi_frames`, and `rtl433_messages` tables in `data/monitor.db`.
- Export sonar events:
  python monitor.py export --db data/monitor.db --out events.csv
 - Export Wi‑Fi/rtl_433 tables as shown above.

Notes
- Installing tshark/rtl_433 and enabling monitor mode varies by OS. On Windows, install Wireshark with TShark; rtl_433 requires RTL‑SDR drivers. On Linux, `apt install tshark rtl-433` then enable monitor mode on a supported adapter.
- Stay within legal and ethical boundaries; do not transmit or jam signals without authorization.
