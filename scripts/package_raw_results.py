"""Pack per-run raw outputs (everything except .npz checkpoints) into release archives.

Usage: python scripts/package_raw_results.py --out C:/pianshi/release_assets
One .tar.gz per confirmation suite plus one each for rehearsal, tuning and the
budget pilot. A manifest with SHA-256 per archive is written next to them.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'v11'


def keep(info):
    return None if info.name.endswith(('.npz', '.tmp')) else info


def pack(src: Path, dest: Path):
    with tarfile.open(dest, 'w:gz', compresslevel=6) as tar:
        tar.add(src, arcname=str(src.relative_to(ROOT)).replace('\\', '/'), filter=keep)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {'archive': dest.name, 'source': str(src.relative_to(ROOT)).replace('\\', '/'),
            'bytes': dest.stat().st_size, 'sha256': h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s / 'runs', f'confirmation_{s.name}_runs.tar.gz') for s in sorted((RES / 'confirmation').glob('c*_*'))
            if (s / 'runs').exists()]
    jobs += [(RES / 'rehearsal', 'rehearsal_all.tar.gz'), (RES / 'tune', 'tuning_stage1.tar.gz'),
             (RES / 'tune2', 'tuning_stage2.tar.gz'), (RES / 'dev', 'dev_budget_pilot.tar.gz')]
    manifest = []
    for src, name in jobs:
        if src.exists():
            entry = pack(src, a.out / name)
            manifest.append(entry)
            print(f"{name}: {entry['bytes'] / 1e6:.1f} MB")
    (a.out / 'MANIFEST_raw_results.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
