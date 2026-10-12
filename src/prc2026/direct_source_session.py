"""Use only a task-local listener whose outbound is fixed to DIRECT."""
import base64
import os
from pathlib import Path

import requests
import yaml


def direct_port():
    value = int(os.environ.get('PRC_FINAL_DIRECT_PORT', '0'))
    if value and not 1024 <= value <= 65535:
        raise ValueError('Invalid task-local DIRECT port')
    return value


def credential():
    cfg = yaml.safe_load((Path.home() / 'clashctl/resources/runtime.yaml').read_text())
    return next(iter(cfg.get('authentication', [])), None)


class DirectAuthAdapter(requests.adapters.HTTPAdapter):
    def proxy_headers(self, proxy):
        headers = super().proxy_headers(proxy)
        value = credential()
        if value:
            headers['Proxy-Authorization'] = 'Basic ' + base64.b64encode(value.encode()).decode()
        return headers


def requests_session(adapter=None):
    session = requests.Session(); session.trust_env = False
    if direct_port():
        endpoint = f'http://127.0.0.1:{direct_port()}'
        session.proxies = {'http': endpoint, 'https': endpoint}
        session.mount('https://', adapter or DirectAuthAdapter())
    elif adapter is not None:
        session.mount('https://', adapter)
    return session


def fsspec_options():
    options = {'client_kwargs': {'trust_env': False}}
    if direct_port():
        import aiohttp
        options['proxy'] = f'http://127.0.0.1:{direct_port()}'
        value = credential()
        if value:
            username, _, password = value.partition(':')
            options['proxy_auth'] = aiohttp.BasicAuth(username, password)
    return options
