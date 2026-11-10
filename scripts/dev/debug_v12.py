import sys, numpy as np, torch
sys.path.insert(0, '.')
sys.argv = ['x']
sys.path.insert(0, 'tests'); from test_v12_gpu import small, DEV
from grnn_v1.learn import DataObjective, compute_gradients, tuning_stats
c, lgn = small()
obj = DataObjective(c, lgn, DEV, n_gray=2, n_rep=1)
b = obj.batch(np.random.default_rng(2))
print('lgn', b['lgn'].shape, float(b['lgn'].mean()), float(b['lgn'].max()))
r, y = c.run(b['lgn'], noise_seed=1)
print('rates nan?', bool(torch.isnan(r).any()), 'mean', float(r.mean()), 'max', float(r.max()))
spont, tun = obj.stats(r)
for k, v in tun.items(): print(k, float(v.mean()), bool(torch.isnan(v).any()))
L = obj.loss(r, y, b); print({k: float(v) for k, v in L.items()})
L['total'].backward(); print('grad nan', bool(torch.isnan(c.mag.grad).any()))
