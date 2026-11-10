"""Parallel byte-range downloader with resume (direct connection, no proxy).

Usage: python download_ranges.py OUT_DIR URL [URL ...]
Each file is fetched in 32 MB ranges by 16 threads; finished ranges are kept
as .partNNNN files so an interrupted run resumes. Size is checked at the end.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

CHUNK = 32 * 1024 * 1024
THREADS = 16


def fetch(session, url, start, end, part):
    if part.exists() and part.stat().st_size == end - start + 1:
        return
    for attempt in range(20):
        try:
            r = session.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=120)
            r.raise_for_status()
            data = r.content
            if len(data) != end - start + 1:
                raise IOError('short read')
            tmp = part.with_suffix('.tmp')
            tmp.write_bytes(data)
            os.replace(tmp, part)
            return
        except Exception as e:  # noqa: BLE001 - retry anything transient
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f'failed range {start}-{end} of {url}')


def download(url, out_dir):
    s = requests.Session()
    s.trust_env = False
    size = int(s.head(url, timeout=60).headers['Content-Length'])
    name = url.rsplit('/', 1)[-1]
    final = out_dir / name
    if final.exists() and final.stat().st_size == size:
        print(f'{name}: already complete', flush=True)
        return
    parts_dir = out_dir / (name + '.parts')
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [(i, min(i + CHUNK, size) - 1) for i in range(0, size, CHUNK)]
    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as pool:
        futures = [pool.submit(fetch, s, url, a, b, parts_dir / f'{k:05d}.part') for k, (a, b) in enumerate(ranges)]
        for k, f in enumerate(futures):
            f.result()
            if k % 10 == 0:
                print(f'{name}: {k + 1}/{len(ranges)} parts, {(k + 1) * CHUNK / 1e6 / (time.time() - t0):.1f} MB/s', flush=True)
    tmp = final.with_suffix('.assembling')
    with open(tmp, 'wb') as out:
        for k in range(len(ranges)):
            out.write((parts_dir / f'{k:05d}.part').read_bytes())
    if tmp.stat().st_size != size:
        raise RuntimeError(f'{name}: size mismatch')
    os.replace(tmp, final)
    for p in parts_dir.glob('*.part'):
        p.unlink()
    parts_dir.rmdir()
    print(f'{name}: done {size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for u in sys.argv[2:]:
        download(u, out)
