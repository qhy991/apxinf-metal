#!/usr/bin/env python3
"""Forward stdin/stdout JSONL to a resident apxinf-decide-server socket.

The demo spawns ``APXINF_DECIDE_CMD`` once per episode and waits for one
``ready`` line, then speaks JSONL. With a resident server started separately
(``apxinf-decide-server --listen unix:/tmp/clef.sock``) this shim makes each
episode pay a socket connect instead of a 93-130 s M4 ANE load: it prints the
server's ready line and forwards both directions unchanged.

    APXINF_DECIDE_CMD="$PY scripts/decide_client.py --addr unix:/tmp/clef.sock"
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time


def connect(address: str, timeout: float) -> socket.socket:
    """Connect to unix:<path> or [tcp:]<host>:<port>, retrying until ready."""
    if address.startswith("unix:"):
        path = address[5:]
        deadline = time.monotonic() + timeout
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(path)
                return sock
            except (FileNotFoundError, ConnectionRefusedError) as error:
                sock.close()
                if time.monotonic() >= deadline:
                    raise SystemExit(f"decide_client: {path} not ready: {error}")
                time.sleep(0.1)
    host, _, port = address.removeprefix("tcp:").rpartition(":")
    if not host or not port:
        raise SystemExit(f"decide_client: {address!r} is neither unix:<path> nor <host>:<port>")
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection((host, int(port)), timeout=5)
        except OSError as error:
            if time.monotonic() >= deadline:
                raise SystemExit(f"decide_client: {address} not ready: {error}")
            time.sleep(0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", required=True, help="unix:<path> or <host>:<port>")
    parser.add_argument("--connect-timeout", type=float, default=600.0)
    args = parser.parse_args()
    sock = connect(args.addr, args.connect_timeout)

    def pump() -> None:
        try:
            for line in sys.stdin.buffer:
                sock.sendall(line)
        except OSError:
            pass
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    threading.Thread(target=pump, name="stdin-pump", daemon=True).start()
    try:
        while True:
            data = sock.recv(65536)
            if not data:
                break
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    except OSError:
        pass
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
