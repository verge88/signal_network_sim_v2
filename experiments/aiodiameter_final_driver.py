#!/usr/bin/env python3
"""Frozen independent Diameter wire driver using alikhalilli/aiodiameter.

The driver is copied into a pinned checkout of the third-party aiodiameter
repository and therefore uses that implementation's Message codec for both
request encoding and response decoding/encoding.  Transport is real TCP.

This is intentionally a small wire-level confirmation harness, not a full EPC
or production HSS.  Scenario parameters are frozen in the workflow before the
first scoring run and must not be changed after results are observed.
"""
from __future__ import annotations

import argparse
import socket
import time

from message import Message

APP_S6A = 16777251
CMD_CER = 257
CMD_DPR = 282
CMD_AIR = 318
CMD_ULR = 316
REQUEST_FLAG = 0x80


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Message:
    header = recv_exact(sock, 20)
    length = int.from_bytes(header[1:4], byteorder="big")
    if length < 20 or length > 1_048_576:
        raise ValueError(f"invalid Diameter length {length}")
    body = recv_exact(sock, length - 20) if length > 20 else b""
    return Message.decodeFromBytes(header + body)


def answer_for(req: Message) -> Message:
    return Message(
        cmdflags=req.header.cmdflags & ~REQUEST_FLAG,
        cmdcode=req.header.cmdcode,
        appId=req.header.appId,
        hopByHopId=req.header.hopByHopId,
        endToEndId=req.header.endToEndId,
    )


def run_server(host: str, port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(16)
    print(f"aiodiameter final wire server listening on {host}:{port}", flush=True)

    while True:
        conn, _ = srv.accept()
        conn.settimeout(20.0)
        try:
            while True:
                try:
                    req = recv_message(conn)
                except (ConnectionError, socket.timeout):
                    break
                conn.sendall(answer_for(req).encode())
        finally:
            conn.close()


def scenario_ops(name: str, count: int):
    n = max(1, int(count))
    if name == "benign_attach":
        return [CMD_AIR, CMD_ULR] * n, True
    if name == "benign_reauth":
        return [CMD_AIR] * n, True
    if name == "benign_update":
        return [CMD_ULR] * n, True
    if name == "air_flood":
        return [CMD_AIR] * n, False
    if name == "ulr_flood":
        return [CMD_ULR] * n, False
    if name == "mixed_flood":
        return [CMD_AIR if i % 2 == 0 else CMD_ULR for i in range(n)], False
    if name == "tracking_burst":
        return [CMD_AIR if i % 3 in (0, 1) else CMD_ULR for i in range(n)], False
    raise ValueError(f"unknown scenario {name!r}")


def request_for(cmd: int, app_id: int = APP_S6A) -> Message:
    return Message(cmdflags=REQUEST_FLAG, cmdcode=cmd, appId=app_id)


def run_client(host: str, port: int, scenario: str, count: int, delay_ms: int) -> None:
    ops, paced = scenario_ops(scenario, count)
    sock = socket.create_connection((host, port), timeout=10.0)
    sock.settimeout(20.0)
    delay_s = max(0, delay_ms) / 1000.0

    try:
        # Frozen RFC 6733 session overhead: one CER/CEA before S6a traffic and
        # one DPR/DPA after it.  This uses the independent codec and is not
        # derived from any Kelly/go-diameter capture.
        sock.sendall(request_for(CMD_CER, app_id=0).encode())
        cea = recv_message(sock)
        if cea.header.cmdcode != CMD_CER or cea.header.appId != 0:
            raise RuntimeError("unexpected CEA")

        if paced:
            for cmd in ops:
                sock.sendall(request_for(cmd).encode())
                ans = recv_message(sock)
                if ans.header.cmdcode != cmd or ans.header.appId != APP_S6A:
                    raise RuntimeError("unexpected Diameter answer")
                if delay_s:
                    time.sleep(delay_s)
        else:
            for cmd in ops:
                sock.sendall(request_for(cmd).encode())
            for cmd in ops:
                ans = recv_message(sock)
                if ans.header.cmdcode != cmd or ans.header.appId != APP_S6A:
                    raise RuntimeError("unexpected Diameter answer")

        sock.sendall(request_for(CMD_DPR, app_id=0).encode())
        dpa = recv_message(sock)
        if dpa.header.cmdcode != CMD_DPR or dpa.header.appId != 0:
            raise RuntimeError("unexpected DPA")
        time.sleep(0.15)
    finally:
        sock.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["server", "client"], required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3868)
    ap.add_argument("--scenario", default="benign_attach")
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--delay-ms", type=int, default=500)
    args = ap.parse_args()

    if args.mode == "server":
        run_server(args.host, args.port)
    else:
        run_client(
            args.host, args.port, args.scenario, args.count, args.delay_ms)


if __name__ == "__main__":
    main()
