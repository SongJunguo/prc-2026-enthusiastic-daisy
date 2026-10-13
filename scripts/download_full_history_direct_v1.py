#!/usr/bin/env python3
"""Download the admitted full-history payloads over ordinary verified HTTPS."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import yaml
from prc2026.full_history_acquisition import settings, save, now, log, effective_days
from prc2026 import full_history_direct_transport as transport
from prc2026.run_guards import clean_head, execution_lock


def main():
    cfg = settings()
    tc = yaml.safe_load(transport.CONFIG.read_text())
    sha = clean_head()
    folder = ROOT / cfg['run_root'] / 'direct_transport'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / cfg['run_root'] / 'manifest.json').read_text())
    manifest = {**manifest, 'days': effective_days(cfg, manifest)}
    manifest['days'] = sorted(manifest['days'], key=lambda d: ('repair_reason' not in d, d['date']))
    started = now()
    with execution_lock(folder):
        save(folder / 'contracts' / ('contract_' + started.replace(':', '-') + '.json'), {
            'source_git_sha': sha, 'transport': tc, 'started_utc': started,
            'https_certificate_verification': True, 'environment_proxies_used': False,
            'dep_labels_read': False, 'effective_assets': manifest['days'],
        })
        result = transport.run(tc, manifest, folder, sha)
        log(event='full_history_direct_finished', **result)


if __name__ == '__main__':
    main()
