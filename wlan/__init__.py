"""Wi-Fi analysis utilities using tshark.

This package provides lightweight wrappers around tshark for:
- Listing interfaces
- Parsing PCAP/PCAPNG files for 802.11 management anomalies (deauth/disassoc)
- Optional short live captures (if supported) which are then parsed

No external Python dependencies required; requires tshark installed on the system.
"""
