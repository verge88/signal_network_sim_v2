"""Collect a local Diameter/CU testbed and create offline observation fixtures."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import hmac
import json
import math
from pathlib import Path
import random
import socket
import struct
import subprocess
from time import perf_counter_ns, sleep, time_ns
import shutil

import numpy as np
import pandas as pd
from scapy.contrib.diameter import AVP, DiamAns, DiamReq
from scapy.utils import RawPcapReader

from diameter_external_cascade import WINDOW_SECONDS, write_json
from diameter_report_recurrence import PROFILE_COLUMNS
from diameter_sim_v1 import DiameterSimulator, add_zone_deviation_features


HOST = "127.0.0.1"
WINDOWS = 96
WARMUP = 20
ROLES = ("MME", "HSS", "PCRF")
MODES = ("healthy", "frozen_joint", "frozen_s6a")
ROLE_MIX = {
    "MME": (6, 6, 2, 2, 2, 3, 1, 2, 2, 2, 2, 1),
    "HSS": (6, 7, 3, 6, 3, 2, 1, 4, 2, 1, 1, 2),
    "PCRF": (1, 2, 1, 1, 1, 1, 1, 1, 7, 8, 7, 4),
}
COMMANDS = (
    (16777251, 318, 0), (16777251, 316, 0), (16777251, 317, 0),
    (16777251, 319, 0), (16777251, 320, 0), (16777251, 321, 0),
    (16777251, 322, 0), (16777251, 323, 0), (16777238, 272, 1),
    (16777238, 272, 2), (16777238, 272, 3), (16777238, 258, 0),
)
Z_FEATURES = (
    "obs_total_messages", "obs_s6a_count", "obs_s6a_dominance_ratio",
    "obs_ratio_s6a_ulr", "obs_ratio_s6a_idr",
)
TEMPORAL_INPUTS = (
    *Z_FEATURES[:1], "obs_entropy", "obs_inbound_outbound_ratio",
    "obs_s6a_dominance_ratio", "obs_n_unique_peers", "integrity_check",
)
TEMPLATE = {}


def templates():
    for index, (app, code, sub) in enumerate(COMMANDS):
        for answer, constructor in ((False, DiamReq), (True, DiamAns)):
            avps = [AVP(264, val="testbed.invalid"), AVP(296, val="testbed.invalid")]
            if answer:
                avps.append(AVP(268, val=2001))
            if sub:
                avps.append(AVP(416, val=sub))
            TEMPLATE[index, answer] = bytes(constructor(code, drAppId=app, avpList=avps))


def dial(kind, identity, answer=False):
    template = TEMPLATE[kind, answer]
    return template[:12] + struct.pack("!II", identity, identity) + template[20:]


def receive_exact(connection, length):
    data = bytearray()
    while len(data) < length:
        chunk = connection.recv(length - len(data))
        if not chunk:
            raise EOFError("Diameter socket closed mid-exchange")
        data.extend(chunk)
    return bytes(data)


def receive_diameter(connection, count, answer):
    messages = []
    for _ in range(count):
        header = receive_exact(connection, 20)
        size = int.from_bytes(header[1:4], "big")
        app, hop, end = struct.unpack("!III", header[8:])
        code = int.from_bytes(header[5:8], "big")
        if header[0] != 1 or bool(header[4] & 128) == answer or hop != end or not 20 <= size <= 8192:
            raise ValueError("Invalid Diameter frame on the testbed")
        payload = receive_exact(connection, size - 20)
        sub = 0
        if code == 272:
            offset = 0
            while offset + 8 <= len(payload):
                avp = int.from_bytes(payload[offset:offset + 4], "big")
                length = int.from_bytes(payload[offset + 5:offset + 8], "big")
                if length < 8 or offset + length > len(payload):
                    raise ValueError("Invalid Diameter AVP")
                if avp == 416:
                    sub = int.from_bytes(payload[offset + 8:offset + 12], "big")
                    break
                offset += (length + 3) & ~3
        messages.append((COMMANDS.index((app, code, sub)), hop))
    return messages


def listener(port):
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(15)
    server.bind((HOST, port))
    server.listen()
    return server


def exchange(client, peer, wanted, next_id, answer_pool, from_peer=False):
    requester, responder = (peer, client) if from_peer else (client, peer)
    ordered = [(kind, next_id + i) for i, kind in enumerate(wanted)]
    requests = b"".join(dial(kind, identity) for kind, identity in ordered)
    received = answer_pool.submit(receive_diameter, responder, len(ordered), False)
    requester.sendall(requests)
    actual = received.result()
    replies = b"".join(dial(kind, identity, True) for kind, identity in actual)
    received = answer_pool.submit(receive_diameter, requester, len(ordered), True)
    responder.sendall(replies)
    if actual != ordered or received.result() != ordered:
        raise ValueError("Diameter request/answer correlation failed")
    return Counter(kind for kind, _ in actual), next_id + len(ordered)


def observed_report(bins, active_peers):
    counts = [sum(bins[i] for i in range(8)), sum(bins[i] for i in range(8, 12))]
    ratios = [bins[i] / max(1, counts[i >= 8]) for i in range(12)]
    s6a = sorted(ratios[:8], reverse=True)
    return {
        "obs_total_messages": sum(counts), "obs_s6a_count": counts[0], "obs_gx_count": counts[1],
        "obs_rx_count": 0, "obs_s13_count": 0, "obs_other_count": 0,
        "obs_entropy": -sum(p * math.log2(p) for count in counts if count
                            for p in (count / sum(counts),)),
        "obs_inbound_outbound_ratio": 1.0, "obs_international_fraction": 0.0,
        "obs_n_unique_peers": active_peers,
        "obs_s6a_dominance_ratio": s6a[0] / max(.001, sum(s6a[1:])),
        "obs_s6a_top2_ratio": sum(s6a[:2]) / max(.001, sum(s6a[2:])),
        "obs_s6a_gini": (2 * sum(i * v for i, v in enumerate(s6a[::-1], 1)) - 9) / 8
        if counts[0] else 0.0,
        **dict(zip(PROFILE_COLUMNS, ratios, strict=True)),
    }


def poll(unit, agent, report, identity, lost, key, reference_ns):
    sent = perf_counter_ns()
    request = json.dumps({"id": identity, "relative_ns": sent - reference_ns}).encode()
    unit.sendto(request, agent.getsockname())
    payload, addr = agent.recvfrom(4096)
    request_rx = perf_counter_ns()
    if payload != request:
        raise ValueError("Unexpected testbed CU request")
    raw = json.dumps({"id": identity, "report": report}, separators=(",", ":")).encode()
    authentication = hmac.digest(key, raw, "sha256")
    reply_tx = perf_counter_ns()
    if lost:
        return None, {"cu_delivered": 0, "cu_req_delay_ms": (request_rx - sent) / 1e6,
                      "cu_resp_delay_ms": np.nan, "cu_processing_delay_ms": 0.0,
                      "cu_rtt_delay_ms": 0.0, "integrity_check": 1}
    agent.sendto(authentication + raw, addr)
    reply, _ = unit.recvfrom(4096)
    received = perf_counter_ns()
    parsed = json.loads(reply[32:])
    if parsed["id"] != identity:
        raise ValueError("Unexpected testbed CU response")
    telemetry = {
        "cu_delivered": 1, "cu_req_delay_ms": (request_rx - sent) / 1e6,
        "cu_resp_delay_ms": (received - reply_tx) / 1e6,
        "cu_processing_delay_ms": (reply_tx - request_rx) / 1e6,
        "cu_rtt_delay_ms": (received - sent) / 1e6,
        "integrity_check": int(hmac.compare_digest(reply[:32], hmac.digest(key, reply[32:], "sha256"))),
    }
    return parsed["report"], telemetry


class TemporalFeatures:
    def __init__(self):
        self.node_temporal_history = defaultdict(lambda: deque(maxlen=30))
        self.z_history = defaultdict(lambda: deque(maxlen=19))

    def add(self, row):
        node = row["node_id"]
        previous = self.z_history[node]
        for key in Z_FEATURES:
            values = [prior[key] for prior in previous]
            deviation = np.std(values) if len(values) >= 2 else 0
            row[f"z_{key}"] = (row[key] - np.mean(values)) / deviation if deviation > 1e-6 else 0
        previous.append({key: row[key] for key in Z_FEATURES})
        self.node_temporal_history[node].append({key: row[key] for key in TEMPORAL_INPUTS})
        row.update(DiameterSimulator._compute_temporal(self, node))


def reports_to_csv(events, destination, mode, phase_start, phase_end, fault_node, encoding):
    temporal = TemporalFeatures()
    last, cached = {}, {}
    loss_history = defaultdict(lambda: deque(maxlen=40))
    delay_history = defaultdict(lambda: deque(maxlen=10))
    misses = Counter()
    rows, labels = [], []
    for event in events:
        node, interval = event["node_id"], event["interval"]
        delivered = event["report"] is not None
        active = mode != "healthy" and node == fault_node and phase_start <= interval < phase_end
        telemetry = event["telemetry"]
        if delivered:
            report = dict(event["report"])
            if interval < phase_start:
                cached[node] = report.copy()
            if active:
                current_fields = PROFILE_COLUMNS if mode == "frozen_joint" else PROFILE_COLUMNS[:8]
                current_fields += ("obs_s6a_dominance_ratio", "obs_s6a_top2_ratio", "obs_s6a_gini")
                report.update({key: cached[node][key] for key in current_fields})
            last[node] = report
            misses[node] = 0
            delay_history[node].append(telemetry["cu_processing_delay_ms"]
                                       + telemetry["cu_resp_delay_ms"])
        else:
            misses[node] += 1
            report = last.get(node, dict.fromkeys(observed_report(Counter(), 0), 0))
        losses = loss_history[node]
        record = {
            "node_id": node, "interval": interval, "zone_id": 0,
            "node_type_enc": encoding[node], **report, **telemetry,
            "cu_combined_loss_prob": sum(losses) / len(losses) if losses else 0.0,
            "consecutive_cu_losses": misses[node], "cu_staleness": misses[node],
            "cu_response_time_jitter": np.std(delay_history[node]) if len(delay_history[node]) >= 3 else 0.0,
            "cu_consistency_error": abs(report["obs_total_messages"] - sum(
                report[f"obs_{interface}_count"] for interface in ("s6a", "gx", "rx", "s13", "other"))),
        }
        losses.append(not delivered)
        temporal.add(record)
        rows.append(record)
        label = "exclude" if interval < WARMUP else "compromise" if active else "benign"
        labels.append({"node_id": node, "interval": interval, "label": label})
    add_zone_deviation_features(pd.DataFrame(rows)).to_csv(destination / f"{mode}-features.csv.gz",
                                                         index=False, compression="gzip")
    pd.DataFrame(labels).to_csv(destination / f"{mode}-labels.csv", index=False)


def run_session(index, root, model, step, base_port):
    rng = random.Random(71000 + index)
    session_id = f"session-{index:02}"
    destination = root / session_id
    destination.mkdir()
    offset = base_port + 10 * index
    traffic_ports = [offset + i for i in range(3)]
    cu_ports = [offset + 3 + i for i in range(3)]
    phase_start = rng.randint(34, 42)
    phase_end = phase_start + rng.randint(30, 38)
    fault_node = ROLES[(index - 1) % len(ROLES)]
    key = hashlib.sha256(f"testbed-control:{session_id}".encode()).digest()
    loss_rate = [.01, .03, .06][index % 3]
    events = []
    answer_sets = 0
    peers = []
    units = []
    agents = []
    try:
        for tcp_port, cu_port in zip(traffic_ports, cu_ports, strict=True):
            with listener(tcp_port) as server:
                client = socket.create_connection((HOST, tcp_port), timeout=15)
                peer, _ = server.accept()
                peer.settimeout(15)
            peers.append((client, peer))
            unit = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            agent = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            unit.settimeout(2)
            agent.settimeout(2)
            unit.bind((HOST, 0))
            agent.bind((HOST, cu_port))
            units.append(unit)
            agents.append(agent)
        reference_ns = perf_counter_ns()
        capture_start_ns = time_ns()
        with ThreadPoolExecutor(max_workers=1) as answers:
            next_id = 1
            for interval in range(WINDOWS):
                for i, role in enumerate(ROLES):
                    phase = 1 + .3 * math.sin(.18 * interval + i + index)
                    weights = ROLE_MIX[role]
                    if index % 2 == 0 and role == fault_node and 21 <= interval <= 63:
                        # A scheduled batch has a stable command mix even when CU is healthy.
                        workload = [kind for kind, weight in enumerate(weights) for _ in range(weight)]
                    else:
                        workload = rng.choices(range(len(COMMANDS)), weights=weights,
                                               k=max(12, round(rng.randint(25, 62) * phase)))
                    client, peer = peers[i]
                    incoming, next_id = exchange(client, peer, workload, next_id, answers,
                                                 from_peer=role == "HSS")
                    answer_sets += sum(incoming.values())
                    messages = Counter({kind: 2 * count for kind, count in incoming.items()})
                    report = observed_report(messages, 1)
                    dropped = interval > 0 and rng.random() < loss_rate
                    payload, telemetry = poll(units[i], agents[i], report, [role, interval],
                                             dropped, key, reference_ns)
                    events.append({"node_id": role, "interval": interval,
                                   "logical_elapsed_s": interval * WINDOW_SECONDS,
                                   "utc_ns": time_ns(), "report": payload, "telemetry": telemetry})
                if step > 0:
                    deadline = reference_ns + round((interval + 1) * step * 1e9)
                    sleep(max(0, (deadline - perf_counter_ns()) / 1e9))
    finally:
        for client, peer in peers:
            client.close()
            peer.close()
        for connection in units + agents:
            connection.close()
    with gzip.open(destination / "cu-observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for event in events:
            raw = dict(event)
            raw["telemetry"] = {k: v if np.isfinite(v) else None
                                for k, v in event["telemetry"].items()}
            stream.write(json.dumps(raw, separators=(",", ":"), allow_nan=False) + "\n")
    for mode in MODES:
        reports_to_csv(events, destination, mode, phase_start, phase_end, fault_node,
                       model["node_type_encoding"])
    metadata = {"id": session_id, "traffic_seed": 71000 + index, "traffic_ports": traffic_ports,
                "cu_ports": cu_ports, "windows": WINDOWS, "capture_start_utc_ns": capture_start_ns,
                "capture_end_utc_ns": time_ns(), "logical_window_seconds": WINDOW_SECONDS,
                "target_wall_seconds_per_window": step, "test_fixture_node": fault_node,
                "test_fixture_start_interval": phase_start, "test_fixture_end_exclusive": phase_end,
                "nominal_loss_rate": loss_rate, "wire_request_answer_pairs": answer_sets,
                "received_reports": sum(event["report"] is not None for event in events)}
    write_json(destination / "session.json", metadata)
    print(f"{session_id}: {answer_sets} Diameter exchanges", flush=True)
    return metadata


def verify_capture(path, folder, session):
    """Reconcile captured CU requests/replies against the observation log."""
    requests, replies = {}, {}
    tcp_segments = 0
    tcp_ports = set(session["traffic_ports"])
    cu_ports = set(session["cu_ports"])
    with RawPcapReader(str(path)) as capture:
        if capture.linktype != 1:
            raise ValueError("Expected Ethernet-format Linux loopback PCAP")
        for packet, metadata in capture:
            if len(packet) < 38 or packet[12:14] != b"\x08\x00":
                continue
            start = 14 + (packet[14] & 15) * 4
            src, dst = struct.unpack("!HH", packet[start:start + 4])
            proto = packet[23]
            if proto == 6 and (src in tcp_ports or dst in tcp_ports):
                tcp_segments += 1
            if proto != 17 or (src not in cu_ports and dst not in cu_ports):
                continue
            length = int.from_bytes(packet[start + 4:start + 6], "big")
            raw = packet[start + 8:start + length]
            if dst in cu_ports:
                record = json.loads(raw)
                target = requests
            else:
                record = json.loads(raw[32:])
                target = replies
            identity = tuple(record["id"])
            if identity in target:
                raise RuntimeError(f"{session['id']}: duplicate CU packet {identity}")
            target[identity] = (record, metadata.sec)
    with gzip.open(folder / "cu-observations.jsonl.gz", "rt", encoding="utf-8") as log:
        events = {(e["node_id"], e["interval"]): e for line in log for e in (json.loads(line),)}
    delivered = {key for key, event in events.items() if event["report"] is not None}
    if requests.keys() != events.keys() or replies.keys() != delivered or tcp_segments == 0:
        raise RuntimeError(f"{session['id']}: PCAP and CU observations disagree")
    for identity, event in events.items():
        wall_sec = event["utc_ns"] / 1e9
        if abs(requests[identity][1] - wall_sec) > 2:
            raise RuntimeError(f"{session['id']}: mismatched capture clock")
        if identity in replies and replies[identity][0]["report"] != event["report"]:
            raise RuntimeError(f"{session['id']}: mismatched delivered CU payload")
    return {"polls": len(requests), "responses": len(replies), "diameter_tcp_segments": tcp_segments}


def capture(args):
    templates()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    model = json.loads(args.model_info.read_text(encoding="utf-8"))
    tcpdump = shutil.which("tcpdump")
    if not tcpdump:
        raise ValueError("tcpdump must be available")
    all_capture = root / "all.pcap"
    cutoff = args.port + args.sessions * 10 + 5
    packet_filter = f"(tcp or udp) and portrange {args.port}-{cutoff}"
    with (root / "tcpdump.log").open("wb") as log:
        recorder = subprocess.Popen([tcpdump, "-i", "lo", "-n", "-U", "-B", "8192",
                                     "-w", str(all_capture), packet_filter], stderr=log)
        try:
            for _ in range(100):
                if all_capture.exists() and recorder.poll() is None:
                    break
                if recorder.poll() is not None:
                    raise RuntimeError("tcpdump failed to start")
                sleep(.05)
            else:
                raise RuntimeError("tcpdump did not open its PCAP")
            with ThreadPoolExecutor(max_workers=min(5, args.sessions)) as batch:
                sessions = list(batch.map(lambda i: run_session(i, root, model, args.wall_step,
                                                               args.port),
                                          range(1, args.sessions + 1)))
        finally:
            if recorder.poll() is None:
                recorder.terminate()
                recorder.wait(timeout=15)
    stats = (root / "tcpdump.log").read_text()
    if "0 packets dropped by kernel" not in stats.splitlines():
        raise RuntimeError(f"Capture loss or missing capture status: {stats}")
    for session in sessions:
        folder = root / session["id"]
        port_filter = " or ".join(f"port {p}" for p in session["traffic_ports"] + session["cu_ports"])
        path = folder / "diameter-cu.pcap"
        subprocess.run([tcpdump, "-r", str(all_capture), "-w", str(path), port_filter],
                       check=True, capture_output=True)
        if path.stat().st_size <= 24:
            raise RuntimeError(f"{session['id']}: empty PCAP")
        session["capture_verification"] = verify_capture(path, folder, session)
        write_json(folder / "session.json", session)
    all_capture.unlink()
    for mode in MODES:
        entries = []
        for session in sessions:
            tag = session["id"]
            entries.append({
                "id": tag, "capture": f"{tag}/diameter-cu.pcap",
                "features": f"{tag}/{mode}-features.csv.gz",
                "labels": f"{tag}/{mode}-labels.csv",
                "source": "Local Scapy Diameter/TCP workload and loopback CU/UDP logs; "
                          f"accelerated logical windows; offline observation mode: {mode}.",
            })
        write_json(root / f"{mode}-manifest.json", {
            "data_origin": "independent_testbed", "window_seconds": WINDOW_SECONDS, "sessions": entries,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-info", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--wall-step", type=float, default=.25,
                        help="Minimum wall-clock seconds per logical 15-second window")
    parser.add_argument("--port", type=int, default=24680)
    args = parser.parse_args()
    if not 0 <= args.wall_step <= WINDOW_SECONDS or not 1024 <= args.port <= 64000:
        parser.error("Invalid wall-step or testbed port")
    capture(args)


if __name__ == "__main__":
    main()
