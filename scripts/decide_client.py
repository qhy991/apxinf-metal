#!/usr/bin/env python3
"""Socket client shim for a resident ``apxinf-decide-server --listen``.

Behaves like the pipe-mode server itself: the demo's ``APXINF_DECIDE_CMD``
spawns this shim, reads one ``ready`` line from its stdout and speaks JSONL on
stdin. The shim connects to the resident server, forwards the ``ready`` line
the server sends on connect and then relays stdin to the socket and the socket
to stdout, one thread each way, so a per-episode spawn pays no model load or
shape warmup:

    APXINF_DECIDE_CMD="python3 scripts/decide_client.py --addr unix:/tmp/clef.sock"

Stdlib only, no dependencies. The shim exits when the server closes the
connection or stdin closes, exactly like the pipe-mode server.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading


def forward_stdin(sock: socket.socket) -> None:
    try:
        for line in sys.stdin.buffer:
            if not line.strip():
                continue
            sock.sendall(line)
    except OSError:
        pass
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", required=True, help="unix:<path> or tcp:<host:port>")
    args = parser.parse_args()
    if args.addr.startswith("unix:"):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args.addr[len("unix:") :])
    elif args.addr.startswith("tcp:"):
        host, _, port = args.addr[len("tcp:") :].rpartition(":")
        if not host or not port:
            parser.error("--addr tcp needs host:port")
        sock = socket.create_connection((host, int(port)))
    else:
        parser.error("--addr must be unix:<path> or tcp:<host:port>")
    with sock:
        sender = threading.Thread(target=forward_stdin, args=(sock,), daemon=True)
        sender.start()
        incoming = sock.makefile("rb")
        while True:
            line = incoming.readline()
            if not line:
                return 0
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    sys.exit(main())
