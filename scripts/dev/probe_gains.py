"""Development only: untrained-network statistics for a grid of gains (dev cohort)."""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grnn_bio.data import load_dataset, partition_cells  # noqa: E402
from grnn_v1.core import Circuit, CircuitConfig, LGN, TYPES  # noqa: E402
from grnn_v1.learn import DataObjective  # noqa: E402

dev = torch.device('cuda')
df = partition_cells(load_dataset(ROOT / 'model/gfr_dataset.json', ROOT / 'data/metadata.csv'),
                     'development', holdout_fraction=.5, seed=1701, unit='donor')
lgn = LGN(dev)
gray = lgn(torch.zeros(1, 5, 48, 64, device=dev))
for lg, rg, ig in itertools.product([0.1, 0.2, 0.35], [1.0, 2.0], [1.0, 2.0]):
    cfg = CircuitConfig(lgn_gain=lg, rec_gain=rg, inh_gain=ig, baseline_fraction=0.02)
    t0 = time.time()
    c = Circuit(df, cfg, 1, dev); c.build_lgn_weights(lgn); c.set_ff_baseline(gray)
    obj = DataObjective(c, lgn, dev, n_gray=4, n_rep=2)
    with torch.no_grad():
        r, y = c.run(obj.batch(np.random.default_rng(0))['lgn'], noise_seed=0)
        spont, tun = obj.stats(r)
        loss = float(obj.loss(r, y, None)['total'])
    out = []
    for t in TYPES:
        i = np.flatnonzero(c.types == t)
        out.append(f"{t}: sp {float(spont[i].median()):.1f} pref {float(tun['pref'][i].median()):.1f} osi {float(tun['osi'][i].median()):.2f}")
    print(f'lgn {lg} rec {rg} inh {ig} | max {float(r.max()):.0f} Hz | loss {loss:.2f} | ' + ' | '.join(out) + f' | {time.time() - t0:.1f}s', flush=True)
