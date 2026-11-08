"""Freeze the confirmatory protocol before any confirmation-cohort run.

Writes docs/LOCK_v11.json with hashes of: the protocol text, the package source,
the analysis/figure scripts, and every confirmation config. run_confirmation.py
refuses to start if any of these changed after locking.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grnn_bio.io import sha256_file, source_fingerprint, object_hash  # noqa: E402

LOCK = ROOT / 'docs' / 'LOCK_v11.json'


def current():
    analysis = sorted((ROOT / 'scripts' / 'analysis').glob('*.py'))
    configs = sorted((ROOT / 'configs' / 'v11' / 'confirmation').glob('*.json'))
    if not configs:
        raise SystemExit('generate confirmation configs first (python scripts/v11_configs.py confirmation)')
    return {
        'protocol_sha256': sha256_file(ROOT / 'docs' / 'PREREGISTRATION_v11.md'),
        'source_hash': source_fingerprint(ROOT),
        'analysis_sha256': object_hash({p.name: sha256_file(p) for p in analysis}),
        'confirmation_configs_sha256': object_hash({p.name: sha256_file(p) for p in configs}),
        'confirmation_configs': [p.name for p in configs],
    }


def verify():
    if not LOCK.exists():
        raise SystemExit('protocol is not locked')
    locked = json.loads(LOCK.read_text())
    now = current()
    bad = [k for k in now if now[k] != locked.get(k)]
    if bad:
        raise SystemExit(f'lock mismatch, changed since locking: {bad}')
    return locked


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        print(json.dumps(verify(), indent=2))
        sys.exit(0)
    if LOCK.exists():
        raise SystemExit(f'{LOCK} exists; a lock is never overwritten')
    record = {'locked_at_utc': datetime.now(timezone.utc).isoformat(), **current(),
              'note': 'No confirmation-cohort run existed when this lock was written.'}
    if (ROOT / 'results' / 'v11' / 'confirmation').exists():
        raise SystemExit('confirmation results already exist; cannot lock retroactively')
    LOCK.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
