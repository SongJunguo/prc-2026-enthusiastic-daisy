#!/usr/bin/env python3
"""Acquire one pinned ADSB.lol July archive with bounded, reversible routing."""
from __future__ import annotations

import base64

from datetime import datetime, timezone

import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import requests
import yaml

RUNTIME = Path.home() / "clashctl/resources/runtime.yaml"
CONTROLLER = "http://127.0.0.1:9090"
PROXY_PORT = 7890


class AuthAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, credential: str | None):
        super().__init__()
        self.credential = credential

    def proxy_headers(self, proxy):
        headers = super().proxy_headers(proxy)
        if self.credential:
            headers["Proxy-Authorization"] = "Basic " + base64.b64encode(self.credential.encode()).decode()
        return headers


def asset_session(port: int, credential: str | None) -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": f"http://127.0.0.1:{port}", "https": f"http://127.0.0.1:{port}"}
    session.mount("https://", AuthAdapter(credential))
    return session


class DirectListener:
    """An in-memory listener for this one process; persistent rules stay intact."""

    def __init__(self, cfg: dict, folder: Path):
        self.path = RUNTIME
        self.original = self.path.read_text()
        self.runtime = yaml.safe_load(self.original)
        self.cfg = cfg
        self.folder = folder
        self.active = False
        self.controller = requests.Session()
        self.controller.trust_env = False
        self.auth = {"Authorization": "Bearer " + self.runtime["secret"]}
        self.credential = next(iter(self.runtime.get("authentication", [])), None)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = folder / f"runtime-before-{stamp}.yaml"
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(self.original)
        self.backup = str(backup)
        self.before_rules = self.rules()

    def request(self, method: str, suffix: str, **kwargs):
        response = self.controller.request(method, CONTROLLER + suffix, headers=self.auth, timeout=20, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"Mihomo configuration API returned HTTP {response.status_code}")
        return response

    def rules(self) -> list:
        return [{key: rule.get(key) for key in ("type", "payload", "proxy")}
                for rule in self.request("GET", "/rules").json()["rules"]]

    def apply(self):
        port = int(self.cfg["direct_listener_port"])
        if any(x.get("name") == self.cfg["direct_listener_name"] or x.get("port") == port
               for x in self.runtime.get("listeners", [])):
            raise RuntimeError("The isolated listener name or port is already in use")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError("The isolated listener TCP port is already occupied")
        configured = dict(self.runtime)
        configured["listeners"] = [*self.runtime.get("listeners", []), {
            "name": self.cfg["direct_listener_name"], "type": "mixed", "listen": "127.0.0.1",
            "port": port, "proxy": "DIRECT", "users": [],
        }]
        self.request("PUT", "/configs", json={"payload": yaml.safe_dump(configured, sort_keys=False)})
        self.active = True
        if self.rules() != self.before_rules:
            raise RuntimeError("An isolated listener unexpectedly changed the global routing rules")
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass

    def restore(self) -> dict:
        if not self.active:
            return {"listener_removed": True, "persistent_bytes_unchanged": self.path.read_text() == self.original}
        unchanged = self.path.read_text() == self.original
        payload = self.original if unchanged else self.path.read_text()
        self.request("PUT", "/configs", json={"payload": payload})
        self.active = False
        if unchanged and self.rules() != self.before_rules:
            raise RuntimeError("Global rules differ after removing the isolated listener")
        if any(x.get("name") == self.cfg["direct_listener_name"] for x in yaml.safe_load(payload).get("listeners", [])):
            raise RuntimeError("The persistent configuration unexpectedly contains the temporary listener")
        return {"listener_removed": True, "persistent_bytes_unchanged": unchanged,
                "global_rules_unchanged": self.rules() == self.before_rules}


def route_lines(since: str, hosts: list[str], outbound: str) -> list[str]:
    start = datetime.fromisoformat(since).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = subprocess.run(["journalctl", "-u", "mihomo.service", "--since", start,
                           "--no-pager", "-o", "cat"], capture_output=True, text=True, check=True).stdout
    return [line for line in text.splitlines()
            if any(f"{host}:443" in line for host in hosts) and f"using {outbound}" in line]
