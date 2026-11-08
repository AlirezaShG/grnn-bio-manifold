"""Time one training step per learner for a given network size (development only)."""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grnn_bio.data import load_dataset, partition_cells, sample_cells
from grnn_bio.model import calibrate_current
from grnn_bio.circuit import Circuit
from grnn_bio.task import TaskConfig, generate_batch
from grnn_bio.learners import exact_bptt_gradients, local_eligibility_gradients, readout_only_gradients

counts = tuple(int(x) for x in sys.argv[1].split(',')) if len(sys.argv) > 1 else (80, 10, 5, 5)
df = partition_cells(load_dataset(ROOT/'model/gfr_dataset.json', ROOT/'data/metadata.csv'), 'development')
params, rows = sample_cells(df, counts, 7)
t0 = time.perf_counter(); cal = calibrate_current(params); print(f'calibration {time.perf_counter()-t0:.2f}s')
c = Circuit(params, rows.cell_type.to_numpy(), seed=7, calibration=cal)
print('N', c.N, 'edges', int(c.mask.sum()))
batch = generate_batch(TaskConfig(), 32, np.random.default_rng(0))
print('T', batch.n_steps)
for name, fn in [('bptt', exact_bptt_gradients), ('local', local_eligibility_gradients), ('readout', readout_only_gradients)]:
    fn(c, batch)
    t0 = time.perf_counter()
    for _ in range(3): fn(c, batch)
    print(f'{name}: {(time.perf_counter()-t0)/3*1000:.1f} ms/step')
