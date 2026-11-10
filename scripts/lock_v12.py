"""Write docs/LOCK_v12.json before any v12 confirmation run (never overwrites)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def h(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


files = ['docs/PREREGISTRATION_v12.md', 'configs/v12/confirmation_settings.json', 'scripts/v12_train.py',
         'scripts/v12_queue.py', 'scripts/v12_analyze.py'] + sorted(
    p.relative_to(ROOT).as_posix() for p in (ROOT / 'grnn_v1').glob('*.py'))
lock = ROOT / 'docs' / 'LOCK_v12.json'
if lock.exists():
    raise SystemExit('lock exists; never overwritten')
if any((ROOT / 'results/v12/conf').glob('*/*/*/done.json')):
    raise SystemExit('completed confirmation runs exist; cannot lock retroactively')
rec = {'locked_at_utc': datetime.now(timezone.utc).isoformat(), 'sha256': {f: h(f) for f in files},
       'note': ('Written before any completed v12 confirmation run. A first launch at 01:31 local '
                'started without this lock because the lock command failed; it was stopped within one '
                'minute, before any run finished, and its partial outputs were deleted.')}
lock.write_text(json.dumps(rec, indent=2) + '\n')
print(rec['locked_at_utc'], len(files), 'files')
