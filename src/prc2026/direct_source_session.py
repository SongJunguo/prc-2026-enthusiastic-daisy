"""Ordinary HTTPS clients for public sources, independent of local proxy settings.

TLS certificate and hostname verification keep the library defaults.  Environment
proxy variables, netrc credentials and obsolete listener settings are not used.
"""
from __future__ import annotations

import requests


def requests_session(adapter=None):
    session = requests.Session()
    session.trust_env = False
    if adapter is not None:
        session.mount("https://", adapter)
    return session


def fsspec_options():
    return {"client_kwargs": {"trust_env": False}}
