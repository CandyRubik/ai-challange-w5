"""Serve real local RAG with external backend socket connections denied."""

import argparse
import atexit
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[1]


def restrict_network(report_path: Path) -> dict:
    report = {"mode": "loopback_only", "allowed_connections": 0, "blocked_attempts": []}
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_lookup = socket.getaddrinfo
    def check(address):
        if isinstance(address, (str, bytes)):  # Unix domain socket.
            return
        host = address[0]
        try:
            allowed = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = False
        if not allowed:
            report["blocked_attempts"].append(str(host))
            save()
            raise OSError(f"Offline mode denies external connection: {host}")
    def connect(connection, address):
        check(address)
        report["allowed_connections"] += 1
        save()
        return original_connect(connection, address)
    def connect_ex(connection, address):
        check(address)
        report["allowed_connections"] += 1
        save()
        return original_connect_ex(connection, address)
    def lookup(host, port, *args, **kwargs):
        if host is not None:
            check((host, port))
        return original_lookup(host, port, *args, **kwargs)
    def save():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = lookup
    save()
    atexit.register(save)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--db", type=Path, default=ROOT / "data/day28-offline.sqlite3")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/day28-artifacts/offline-network.json")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", LLM_DEFAULT_PROVIDER="ollama",
                      RAG_NETWORK_MODE="loopback_only",
                      DEEPSEEK_API_KEY="", CHAT_DB_PATH=str(args.db))
    sys.path.insert(0, str(ROOT))
    report = restrict_network(args.report)
    # Exercise the guard once; record it separately from application requests.
    try:
        socket.getaddrinfo("example.com", 443)
    except OSError:
        report["guard_self_test_passed"] = True
        report["blocked_attempts"].clear()
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, loop="asyncio")


if __name__ == "__main__":
    main()
