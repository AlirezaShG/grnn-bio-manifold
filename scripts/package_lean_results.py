"""Lean per-run archive for upload over a slow link (confirmation suites only).

Keeps every per-run summary file (manifests, completed.json, population and
gradient analyses, interventions, psychometrics, sampled cells, checkpoint
geometry). Drops per-step history (history.json, learning_curve.csv), per-trial
tables and checkpoints; validation curves (every 50 updates) are kept in one
CSV per suite. Output is split into parts of at most --part-mb megabytes.
"""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / 'results' / 'v11' / 'confirmation'
DROP = {'history.json', 'learning_curve.csv', 'trial_predictions.csv', 'rate_intervention_trials.csv'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--part-mb', type=float, default=10)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz', compresslevel=9) as tar:
        for suite in sorted(CONF.glob('c*_*')):
            if not (suite / 'runs').exists():
                continue
            curves = []
            for f in sorted((suite / 'runs').rglob('*')):
                if f.is_dir() or f.suffix == '.npz' or f.name in DROP:
                    if f.name == 'learning_curve.csv':
                        d = pd.read_csv(f).dropna(subset=['validation_bce'])
                        d.insert(0, 'seed', f.parent.name)
                        d.insert(0, 'condition', f.parent.parent.name)
                        curves.append(d)
                    continue
                tar.add(f, arcname=str(f.relative_to(ROOT)).replace('\\', '/'))
            if curves:
                data = pd.concat(curves).to_csv(index=False).encode()
                info = tarfile.TarInfo(str((suite / 'validation_curves.csv').relative_to(ROOT)).replace('\\', '/'))
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    size = int(a.part_mb * 1e6)
    parts = []
    for i in range(0, len(blob), size):
        p = a.out / f'confirmation_lean.tar.gz.part{i // size:02d}'
        p.write_bytes(blob[i:i + size])
        parts.append({'file': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {'archive': 'confirmation_lean.tar.gz', 'bytes': len(blob),
                'sha256': hashlib.sha256(blob).hexdigest(), 'parts': parts,
                'reassemble': 'cat confirmation_lean.tar.gz.part* > confirmation_lean.tar.gz  (or: copy /b on Windows)',
                'dropped_per_run': sorted(DROP) + ['*.npz'],
                'note': 'Validation curves (every 50 updates) are in <suite>/validation_curves.csv.'}
    (a.out / 'MANIFEST_confirmation_lean.json').write_text(json.dumps(manifest, indent=2))
    print(f'{len(blob) / 1e6:.1f} MB in {len(parts)} parts')


if __name__ == '__main__':
    main()
