"""Predeclared paired factorial effects, not a one-factor-at-a-time surrogate."""
from itertools import combinations,product
import numpy as np
from .statistics import paired_comparison,holm_adjust


def factorial_contrasts(rows,metric='accuracy_incongruent',seed=1701):
    """2x2x2 contrast family on one evaluation split, paired by network seed.

    Positive coding: intact identity, type-prior topology, local learning.
    Main effects are high-minus-low; two-way terms are differences of
    differences averaged over the third factor; the three-way term is a
    difference of differences of differences. Holm covers all seven terms.
    """
    levels={'identity':('within_type_shuffle','authentic'),
            'topology':('degree_rewired','type_prior'),'learner':('bptt','local')}
    factors=list(levels)
    rows=list(rows)
    if len({r['split'] for r in rows if 'split' in r})>1:
        raise ValueError('Filter factorial rows to one prespecified evaluation split')
    selected=[r for r in rows if all(r[k] in levels[k] for k in factors) and r.get('training_noise_std',0)==0]
    groups={}
    for row in selected:
        code=tuple(levels[k].index(row[k]) for k in factors)
        raw_seed=row['seed']
        if isinstance(raw_seed,(bool,np.bool_)) or not isinstance(raw_seed,(int,np.integer)) or raw_seed<0:
            raise ValueError('Factorial network seeds must be nonnegative integers')
        s=int(raw_seed); group=groups.setdefault(s,{})
        if code in group: raise ValueError('Duplicate factorial condition per paired seed')
        group[code]=float(row[metric])
    full=set(product((0,1),repeat=3))
    if not groups or any(set(g)!=full for g in groups.values()):
        return {'status':'not_estimable_incomplete_factorial','metric':metric,
                'required_cells':8,'available_cells_per_seed':{str(s):len(g) for s,g in groups.items()},'contrasts':[]}
    contrasts=[]
    seed_ids=sorted(groups)
    for order in range(1,4):
        for subset in combinations(range(3),order):
            values=[]
            for s in seed_ids:
                values.append(2**order*np.mean([v*np.prod([2*code[k]-1 for k in subset]) for code,v in groups[s].items()]))
            result=paired_comparison(values,np.zeros(len(values)),seed=seed+len(contrasts))
            contrasts.append({'term':' x '.join(factors[i] for i in subset),'order':order,
                              'metric':metric,'network_seeds':seed_ids,'per_network_contrast':values,**result})
    for c,p in zip(contrasts,holm_adjust([c['p_value'] for c in contrasts])):
        c['p_holm']=p;c['family_size']=7
    return {'status':'complete','metric':metric,'positive_coding':{k:v[1] for k,v in levels.items()},
            'unit_of_inference':'network_run','contrasts':contrasts}
