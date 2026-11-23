#!/usr/bin/env python3
"""External SIP wire-capture audit of TICD's *observable* ACK invariant.

This is NOT a full TICD validation: an ordinary PCAP contains no independently
verified slave report, zonal peers, or ground-truth attack labels. Never fill
these missing fields with invented values.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ["frame.time_epoch", "sip.Call-ID", "sip.CSeq.seq",
          "sip.CSeq.method", "sip.Method", "sip.Status-Code"]


def decode(path):
    # Work around tshark AppArmor restrictions on GitHub hosted runners.
    with tempfile.TemporaryDirectory(prefix="sip-external-", dir="/tmp") as td:
        temp = Path(td) / ("capture" + path.suffix)
        shutil.copyfile(path, temp)
        command = ["tshark", "-r", str(temp), "-Y", "sip", "-T", "fields",
                   "-E", "occurrence=f", "-E", "separator=\t"]
        for field in FIELDS:
            command += ["-e", field]
        p = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = []
    for line in p.stdout.splitlines():
        vals = line.split("\t")
        vals += [""] * (len(FIELDS) - len(vals))
        row = dict(zip(FIELDS, vals[:len(FIELDS)]))
        if row["sip.Call-ID"]:
            rows.append(row)
    return rows


def analyze(path):
    raw = path.read_bytes()
    records = decode(path)
    if not records:
        raise ValueError(f"{path}: zero decoded SIP messages")
    counts = Counter()
    tx = defaultdict(lambda: {"invite": 0, "ok": 0, "ack": 0})
    calls = set()
    for row in records:
        call = row["sip.Call-ID"]
        calls.add(call)
        seq = row["sip.CSeq.seq"].split(",")[0].strip()
        method = row["sip.Method"].split(",")[0].strip().upper()
        cseq_method = row["sip.CSeq.method"].split(",")[0].strip().upper()
        status = row["sip.Status-Code"].split(",")[0].strip()
        counts["sip_messages"] += 1
        if method:
            counts["method_" + method] += 1
        if status:
            counts["status_" + status] += 1
        if not seq:
            continue
        key = (call, seq)
        if method == "INVITE":
            tx[key]["invite"] += 1
        if status == "200" and cseq_method == "INVITE":
            tx[key]["ok"] += 1
        if method == "ACK":
            tx[key]["ack"] += 1

    answered = {k for k, v in tx.items() if v["ok"]}
    acked = {k for k, v in tx.items() if v["ack"]}
    invited = {k for k, v in tx.items() if v["invite"]}
    # ACK for 2xx is matched by Call-ID and CSeq; these are logical
    # transactions, not packet counts (retransmissions are deduplicated).
    matched = answered & acked
    missing = answered - acked
    ratio = len(matched) / len(answered) if answered else None
    return {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "sip_messages": counts["sip_messages"],
        "unique_call_ids": len(calls),
        "unique_invite_keys": len(invited),
        "unique_invite_200_keys": len(answered),
        "unique_ack_keys": len(acked),
        "matched_200_ack_keys": len(matched),
        "missing_ack_after_200_keys": len(missing),
        "physical_ack_closure_ratio": ratio,
        "method_counts": {k[7:]: v for k, v in counts.items() if k.startswith("method_")},
        "response_counts": {k[7:]: v for k, v in counts.items() if k.startswith("status_")},
        "interpretation": ("No 200 OK to INVITE: ACK closure not assessable" if ratio is None
                           else "Network-observed ACK closure; not TICD reported-versus-physical mismatch"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcaps", nargs="+", type=Path)
    ap.add_argument("--out-dir", default="results_external_sip")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = [analyze(p) for p in args.pcaps]
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    with (out / "capture_metrics.csv").open("w", newline="") as f:
        fields = ["file", "sha256", "bytes", "sip_messages", "unique_call_ids",
                  "unique_invite_keys", "unique_invite_200_keys", "unique_ack_keys",
                  "matched_200_ack_keys", "missing_ack_after_200_keys",
                  "physical_ack_closure_ratio"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fields})
    summary = {
        "validation_target": "SIP/TICD, RFC3261 physical ACK observable sub-invariant only",
        "validation_level": "external_trace_protocol_audit_not_full_detector_validation",
        "external_sources": "Wireshark SampleCaptures mirrored by NormB/sipnab",
        "non_simulator_data": True,
        "production_traffic_verified": False,
        "attack_ground_truth_present": False,
        "independent_slave_reports_present": False,
        "zonal_telemetry_present": False,
        "full_ticd_executed_on_external_data": False,
        "auc_roc": None,
        "recall": None,
        "false_positive_rate": None,
        "warning": "PCAP only cannot validate TICD's mismatch between slave reports and independent transit observation; no attacks are labeled.",
        "total_decoded_sip_messages": sum(x["sip_messages"] for x in results),
        "total_unique_200_keys": sum(x["unique_invite_200_keys"] for x in results),
        "total_matched_ack_keys": sum(x["matched_200_ack_keys"] for x in results),
        "captures": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if not any(r["unique_invite_200_keys"] for r in results):
        raise SystemExit("No 200 OK to INVITE observed in any capture")


if __name__ == "__main__":
    main()
