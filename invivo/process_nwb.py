"""Natural-scene responses of V1 (VISp) units from one Visual Coding Neuropixels session.

Output npz: unit_ids, waveform_duration_ms (RS > 0.4 ms putative excitatory,
FS < 0.4 ms putative PV), frames [n_pres] (0..117; -1 blank), counts [n_pres, n_units]
in 0-250 ms after onset.
Unit QC as in the Allen defaults: isi_violations < 0.5, amplitude_cutoff < 0.1,
presence_ratio > 0.9.
Usage: python process_nwb.py <session_id> [<session_id> ...]
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NWB = Path('E:/pianshi_invivo/nwb')
OUT = HERE / 'processed'


def good_visp_units(session_id: int) -> pd.DataFrame:
    u = pd.read_csv(HERE / 'units.csv', low_memory=False)
    c = pd.read_csv(HERE / 'channels.csv', low_memory=False)
    p = pd.read_csv(HERE / 'probes.csv')
    u = u.merge(c[['id', 'ecephys_probe_id', 'ecephys_structure_acronym']].rename(columns={'id': 'ecephys_channel_id'}),
                on='ecephys_channel_id')
    u = u.merge(p[['id', 'ecephys_session_id']].rename(columns={'id': 'ecephys_probe_id'}), on='ecephys_probe_id')
    m = ((u.ecephys_session_id == session_id) & (u.ecephys_structure_acronym == 'VISp') &
         (u.isi_violations < 0.5) & (u.amplitude_cutoff < 0.1) & (u.presence_ratio > 0.9))
    return u[m]


def process(session_id: int):
    path = NWB / f'session_{session_id}.nwb'
    units = good_visp_units(session_id)
    with h5py.File(path, 'r') as f:
        ids = f['units/id'][:]
        st = f['units/spike_times']
        idx = f['units/spike_times_index'][:]
        grp = f['intervals/natural_scenes_presentations']
        start = grp['start_time'][:]
        frame = grp['frame'][:]
        if frame.dtype.kind in 'SO':
            frame = np.array([float(x) for x in frame])
        frame = frame.astype(int)
        keep = np.flatnonzero(np.isin(ids, units.id.to_numpy()))
        counts = np.zeros((len(start), len(keep)), np.int16)
        order = np.argsort(start)
        for k, ui in enumerate(keep):
            lo = 0 if ui == 0 else idx[ui - 1]
            spikes = st[lo:idx[ui]]
            a = np.searchsorted(spikes, start)
            b = np.searchsorted(spikes, start + 0.25)
            counts[:, k] = b - a
        uid = ids[keep]
    meta = units.set_index('id').loc[uid]
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f'ns_{session_id}.npz', unit_ids=uid, frames=frame, counts=counts,
                        waveform_duration_ms=meta['duration'].to_numpy(float), start=start)
    print(session_id, 'units', len(uid), 'presentations', len(start), 'images', len(set(frame) - {-1}), flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        process(int(s))
