import json, glob, os
import pandas as pd
pd.set_option('display.width', 200)
for d in sorted(glob.glob('results/v11/tune/t_*')):
    p = os.path.join(d, 'tuning_summary.json')
    if not os.path.exists(p): continue
    t = pd.DataFrame(json.load(open(p)))
    print(f'\n== {os.path.basename(d)}')
    a = t.pivot_table(index=['learner', 'lr_rec'], columns='lr_out', values='mean_validation_bce').round(3)
    b = t.pivot_table(index=['learner', 'lr_rec'], columns='lr_out', values='mean_validation_accuracy').mul(100).round(1)
    print(pd.concat({'BCE': a, 'ACC%': b}, axis=1).to_string())
