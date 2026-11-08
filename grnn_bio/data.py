"""Audited cell selection, immutable fitted parameters and explicit controls."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .model import GFRParameters

CELL_TYPES = ('Pyr', 'PV', 'SST', 'VIP')
EXCITATORY_CRE_LINES = {
    'Ntsr1-Cre_GN220', 'Rorb-IRES2-Cre', 'Nr5a1-Cre', 'Scnn1a-Tg3-Cre',
    'Cux2-CreERT2', 'Rbp4-Cre_KL100', 'Ctgf-T2A-dgCre', 'Sim1-Cre_KJ18',
    'Tlx3-Cre_PL56', 'Scnn1a-Tg2-Cre', 'Glt25d2-Cre_NF107',
}


def load_dataset(json_path, metadata_path, min_validation_evr=.5,
                 species='Mus musculus', dt_ms=20, activation_dt_ms=20):
    """Select on original fitting validation EVR only; test EVR is descriptive.

    Cell class is a conservative operational label based on cre line and spiny
    morphology. It is not a claim about basket/Martinotti anatomy or L1 location.
    """
    rows = json.loads(Path(json_path).read_text(encoding='utf-8'))
    df = pd.DataFrame(rows)
    required = {'cell_id','bin_size','actv_bin_size','val_evr','params','cre-line'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing dataset columns: {required - set(df.columns)}')
    df = df[(df.bin_size == dt_ms) & (df.actv_bin_size == activation_dt_ms)].copy()
    if df.empty:
        raise ValueError('No models match requested bin sizes; dt cannot be silently changed')
    if df.cell_id.duplicated().any():
        raise ValueError('Duplicate cell IDs in selected dataset configuration')
    meta = pd.read_csv(metadata_path)
    if meta.specimen__id.duplicated().any():
        raise ValueError('Metadata has duplicate specimen IDs')
    df = df.merge(meta, left_on='cell_id', right_on='specimen__id', how='left', validate='one_to_one')
    if df.specimen__id.isna().any():
        raise ValueError('Some fitted cells have no metadata record')
    df = df[(df.val_evr >= min_validation_evr) & (df.donor__species == species)].copy()
    labels = np.full(len(df), '', dtype=object)
    labels[(df['tag__dendrite_type'] == 'spiny') & df['cre-line'].isin(EXCITATORY_CRE_LINES)] = 'Pyr'
    for label, cre in [('PV','Pvalb-IRES-Cre'),('SST','Sst-IRES-Cre'),('VIP','Vip-IRES-Cre')]:
        labels[df['cre-line'] == cre] = label
    df['cell_type'] = labels
    return df[df.cell_type != ''].sort_values('cell_id').reset_index(drop=True)


def _donor_holdout(df, holdout_fraction, seed):
    """Assign whole donors to the holdout so each type gets ~holdout_fraction.

    Donors are visited in a seeded random order; each goes to the cohort whose
    remaining per-type quota (relative to its target) is larger for the cell
    types that donor contributes. Deterministic for a given seed and table.
    """
    if 'donor__id' not in df:
        raise ValueError('Donor-level partition needs a donor__id column')
    rng = np.random.default_rng(seed)
    totals = df.cell_type.value_counts().reindex(CELL_TYPES, fill_value=0).to_numpy(float)
    targets = {'hold': holdout_fraction * totals, 'dev': (1 - holdout_fraction) * totals}
    counts = {'hold': np.zeros(4), 'dev': np.zeros(4)}
    donors = np.sort(df['donor__id'].unique())
    hold = set()
    for donor in donors[rng.permutation(len(donors))]:
        contrib = (df.loc[df['donor__id'] == donor, 'cell_type']
                   .value_counts().reindex(CELL_TYPES, fill_value=0).to_numpy(float))
        score = {k: float(np.sum(contrib * (targets[k] - counts[k]) / np.maximum(targets[k], 1)))
                 for k in ('hold', 'dev')}
        side = ('hold' if score['hold'] > score['dev'] else 'dev' if score['dev'] > score['hold']
                else ('hold' if rng.random() < holdout_fraction else 'dev'))
        counts[side] += contrib
        if side == 'hold':
            hold.add(donor)
    return df['donor__id'].isin(hold).to_numpy()


def partition_cells(df, partition='all', holdout_fraction=.25, seed=1701, unit='specimen'):
    """Deterministic disjoint cohorts, stratified by operational type.

    unit='specimen' (v7): cells are split, donors may appear in both cohorts.
    unit='donor' (v11): whole donors are split, so no animal contributes cells
    to both cohorts. Overlap is reported by the experiment manifest either way.
    """
    if partition not in {'all','development','confirmation'}:
        raise ValueError('partition must be all, development, or confirmation')
    if not 0 < holdout_fraction < 1:
        raise ValueError('holdout_fraction must lie strictly between zero and one')
    if unit not in {'specimen','donor'}:
        raise ValueError("unit must be 'specimen' or 'donor'")
    if partition == 'all':
        return df.copy()
    if unit == 'donor':
        holdout = _donor_holdout(df, holdout_fraction, seed)
        keep = holdout if partition == 'confirmation' else ~holdout
        return df.loc[keep].sort_values('cell_id').reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = []
    for ct in CELL_TYPES:
        idx = df.index[df.cell_type == ct].to_numpy()
        idx = rng.permutation(idx)
        n_hold = max(1, int(np.floor(len(idx) * holdout_fraction)))
        keep.extend(idx[:n_hold] if partition == 'confirmation' else idx[n_hold:])
    return df.loc[keep].sort_values('cell_id').reset_index(drop=True)


def parameters_from_rows(rows):
    ps = rows['params'].tolist()
    dt = {int(p['bin_size']) for p in ps}
    if len(dt) != 1:
        raise ValueError('Cannot mix differently discretized cell models')
    return GFRParameters(
        a=np.array([p['a'][0] for p in ps]),
        beta=np.array([p['b'][0] for p in ps]),
        ds=np.array([p['ds'] for p in ps]),
        max_current=np.array([p['g']['max_current'][0] for p in ps]),
        gamma_hz=1000. * np.array([p['g']['max_firing_rate'][0] for p in ps]),
        poly_coeff=np.array([p['g']['poly_coeff'][0] for p in ps]),
        threshold=np.array([p['g']['b'][0] for p in ps]),
        cell_ids=rows['cell_id'].to_numpy(), dt_ms=dt.pop(),
    )


def sample_cells(df, counts=(80,10,5,5), seed=42):
    if len(counts) != 4 or any(int(n) != n or n < 1 for n in counts):
        raise ValueError('Provide four positive integer cell counts (Pyr,PV,SST,VIP)')
    rng = np.random.default_rng(seed)
    sampled = []
    for ct, n in zip(CELL_TYPES, counts):
        pool = df[df.cell_type == ct]
        if len(pool) < n:
            raise ValueError(f'{ct}: requested {n} unique cells, cohort has {len(pool)}. Reduce network size.')
        positions = rng.choice(len(pool), size=int(n), replace=False)
        sampled.append(pool.iloc[positions])
    rows = pd.concat(sampled, ignore_index=True)
    rows['slot_index'] = np.arange(len(rows))
    return parameters_from_rows(rows), rows


def manipulate_identity(params, cell_types, mode='authentic', seed=0):
    """Return a new frozen bank and source mappings; never modify input arrays.

    Homogeneous controls average within type and are explicitly synthetic,
    not asserted to match nonlinear gain. Fast-decay control changes ds only
    and rescales a,beta to preserve a/ds and beta/ds (steady-state equations),
    not all transient/closed-loop dynamics.
    """
    valid = {'authentic','within_type_shuffle','cross_type_permutation',
             'type_homogeneous','fast_decay'}
    if mode not in valid:
        raise ValueError(f'Unknown identity control {mode}')
    arrays = {name: np.array(getattr(params,name),copy=True) for name in
              ['a','beta','ds','max_current','gamma_hz','poly_coeff','threshold','cell_ids']}
    ct = np.asarray(cell_types)
    if ct.shape != (params.n,):
        raise ValueError('cell_types must align with neuron slots')
    rng = np.random.default_rng(seed)
    source = {k: params.cell_ids.copy() for k in ['a','beta','activation','tuple']}
    if mode == 'within_type_shuffle':
        for label in CELL_TYPES:
            idx = np.flatnonzero(ct == label)
            for group, names in [('a',['a']),('beta',['beta']),
                                 ('activation',['max_current','gamma_hz','poly_coeff','threshold'])]:
                perm = rng.permutation(idx)
                for name in names:
                    arrays[name][idx] = getattr(params,name)[perm]
                source[group][idx] = params.cell_ids[perm]
    elif mode == 'cross_type_permutation':
        perm = rng.permutation(params.n)
        for name in arrays:
            arrays[name] = arrays[name][perm]
        source = {k: params.cell_ids[perm].copy() for k in source}
    elif mode == 'type_homogeneous':
        for label in CELL_TYPES:
            idx = np.flatnonzero(ct == label)
            for name in arrays:
                if name != 'cell_ids':
                    arrays[name][idx] = np.mean(getattr(params,name)[idx],axis=0)
        source = {k: np.full(params.n,-1,dtype=np.int64) for k in source}
    elif mode == 'fast_decay':
        new_ds = np.maximum(params.ds,1. - np.exp(-params.dt_ms/20.))
        ratio = new_ds/params.ds
        arrays['a'] *= ratio
        arrays['beta'] *= ratio
        arrays['ds'] = new_ds
    out = GFRParameters(**arrays,dt_ms=params.dt_ms)
    provenance = {'mode':mode,'synthetic':mode != 'authentic',
                  'source_cell_ids':{k:v.tolist() for k,v in source.items()},
                  'reference_fingerprint':params.fingerprint(),
                  'condition_fingerprint':out.fingerprint()}
    return out, provenance
