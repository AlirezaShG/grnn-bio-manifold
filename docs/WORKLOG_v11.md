# v11 work log

v11 starts from the v7 code (`reference/grnn_bio_manifold_project_v7`). The v8–v10 documents are
summaries only: no code or results for them exist in this folder. Their ideas are
re-implemented here and tested from scratch. None of their numbers are reused.

## Audit of v7 (2026-11-08)

Verified correct:
- 110/110 tests pass.
- The pilot tables in the v7 report match `results/pilot/*/metrics.csv` exactly.
- Cohort: 597 cells (339/107/86/65), 410 donors; specimen split 450/147 with 65 shared donors.
- GFR forward equation and units.
- BPTT adjoint (checked by hand against the forward equations).
- Local eligibility is random-feedback e-prop adapted to K-filter GFR cells.
- Task balance, masks, statistics (paired sign-flip, Holm, factorial coding), fit/eval separation.

Problems found in v7's design (not bugs in its code):
1. Identity controls keep the calibration (offset, input scale) of the intact bank, so
   shuffle, homogeneous and cross-type networks run at about 2x the rate with 3–9% of
   bins near the ceiling. That is an operating-point confound.
2. The specimen split shares 65 donors between cohorts.
3. Pilot runs had not converged (best step = last step for almost all runs). Learners were
   compared at an arbitrary, truncated budget.
4. Only one task family.
5. The rate hierarchy (PV > SST > VIP ≈ Pyr) is set by the calibration target, which is 5% of
   each fitted ceiling (PV 15.5, SST 8.2, VIP 7.8, Pyr 6.1 Hz in isolation).
6. The forward local kernel is 50x slower than BPTT (643 vs 13 ms per step at 100 neurons).

## v11 decisions

- Exact reverse-mode (adjoint) computation of the local e-prop gradient: BPTT with the paths
  through other neurons cut and the readout replaced by fixed feedback. It is identical to
  the forward eligibility trace (tested), only the simulation order differs.
- Post-manipulation calibration for every identity control (`Condition.calibration='matched'`).
  `'reference'` keeps the v7 behaviour, so the confound itself can be shown.
- Donor-disjoint 50/50 cohorts, stratified by cell type. The confirmation cohort stays untouched
  until the protocol is locked.
- Task 2: delayed match-to-sample with a simultaneous distractor (v9 idea), plus late context.
- Optimizer sensitivity: Adam (global clipping), plain SGD, and per-synapse RMS-normalized SGD
  (v9's "homeostatic_sgd", which is RMSProp: no momentum, no global clipping). It is not called
  synaptic scaling.
- Gradient alignment with BPTT is measured at checkpoints, with random and symmetric feedback, to
  separate truncation error from feedback misalignment.
- Not adopted from v10: the ALM composite "alignment" score (undefined, not like-for-like) and
  synaptic-physiology numbers that cannot be traced to a source table.

## Status

- 2026-11-08: Implemented the v11 changes. 126 tests pass (v7's 110 plus 16 new).
- D0 budget pilot (dev cohort, 2 seeds, 3000 updates). Validation accuracy plateaus at
  1000–1500 updates, but BCE keeps falling, so 3000 updates are used. ID accuracy is near ceiling
  for every learner, so unseen delays separate them.
- D1 tuning: 6 searches × 117 runs, about 3 min per run under full load.
- Not adopted from v10: Campagnola/Billeh connectivity numbers (only available as figures, nothing
  traceable) and the ALM composite score.
- Windows note: never read run files while workers write them. `write_json` now retries
  `os.replace`.
- 2026-11-08 → 09:
  - Two-stage LR search. With Adam, the reservoir and no-β rule needed lr_out up to 3.
  - Rehearsal (dev cohort, 4 seeds) of families A–D.
  - Family D (operating point) was added before locking.
  - Protocol locked 23:53 (local time).
  - Confirmation: 1,680 runs, no failures, 145 min.

## Confirmation results (held-out donor cohort, 16 paired networks)

Primary endpoint: unseen-delay conflict accuracy.

- **A1 (β in the trace):** +7 to +17 pp in 6/6 settings; Holm-significant in 5/6 (not DMS/RMS,
  Holm p = 0.10). Without β, accuracy is close to or below the frozen reservoir.
- **A2 (local vs frozen reservoir):** positive in 6/6, significant in 3/6 (late/RMS, DMS ×2).
- **A3 (BPTT vs local):** +11 to +22 pp on late and DMS (significant); none on early.
- **Mechanism:** with symmetric feedback, the local estimate has cosine 0.76–0.94 with β vs
  0.54–0.66 without. The per-Pyr-cell gain correlates with adaptation loop gain
  (r = 0.16–0.23, CIs > 0). β still adds 8–13 pp with symmetric feedback. Symmetric vs random
  feedback adds only about 4 pp (n.s.). The remaining BPTT gap comes from cut network paths.
- **Representation:** both streams are decodable at ≥98% in the delay for every learner, the
  reservoir included. Learners differ only in the response phase. Fast decay drops every task to
  chance.
- **Family B:**
  - identity is practically negligible in 4/6 cases and inconclusive in 2/6;
  - wiring is negligible in 6/6;
  - heterogeneity helps BPTT strongly (+12.6, +20.1, +24.9 pp; Holm p ≤ 0.001), but not the local
    rule.
- **Family C:** the reference-calibrated shuffle runs at 24 Hz with 7% of samples near ceiling,
  yet its accuracy does not differ significantly at the tuned budget.
- **Family D:** BPTT is insensitive to the operating point. The local rule's slope depends on the
  task: −4.6 pp per doubling (early), +3.5 (late), +1.8 (DMS, p = 0.08).

Paper: `paper/main.pdf` (16 pages). Every number comes from
`results/v11/confirmation/analysis/`.

# v12 (2026-11-10, night): representational geometry vs mouse V1

The user judged that v11's results were driven by its abstract task. v12 keeps the question,
"local (e-prop) vs global (BPTT) learning: representational geometry and its match to mouse V1",
and rebuilds the setup. Inputs consulted: user spec PDF (Downloads/Telegram Desktop) and
Ito et al. 2026 (Allen biorealistic V1; github AllenInstitute/biorealistic-v1-model).

- Code: `grnn_v1/` (PyTorch; env `braincog_env`, Python 3.8, torch 2.4.1+cu118):
  - `core.py`: stimuli, LGN, GFR circuit (N = 1000; V1DD p_max, like-to-like E→E, log-normal
    PSP-scaled weights; baseline 2%);
  - `learn.py`: data and change objectives, e-prop via a detach trick, Exponentiated Adam;
  - `evaluate.py`;
  - `compare.py`: RSA, CKA, PR, sparseness, decoding, tuning similarity;
  - `stats.py`: a Python 3.8 copy of `grnn_bio/statistics`.
- Scripts: `scripts/v12_train.py`, `v12_queue.py`, `v12_analyze.py`. GPU/CPU: per-step cost is
  launch-bound, so jobs run on 6 GPU slots plus CPU single-thread workers, with every job of a
  seed on one device type.
- In-vivo data:
  - `C:/pianshi/invivo`: Ito curated OSI/DSI/rates table (3,156 units), V1DD and PSP tables,
    118 natural-scene TIFFs, natural movie 1 (a .npy despite its .h5 name);
  - session NWBs on `E:/pianshi_invivo/nwb` (6 V1-rich BO1.1 sessions), processed by
    `invivo/process_nwb.py` into `invivo/processed/ns_*.npz`.
- Dev findings so far:
  - natural-scene RSA (about 0.24 disattenuated) is nearly the same for every rule and for the
    untrained net, i.e. set by the front end;
  - local learning lowers image decodability to a mouse-like level, while BPTT does not;
  - at lr_rec 5e-3, BPTT optimises the data objective more slowly, so LRs are being tuned.
- The change task was too hard with arbitrary frames, so it now uses 8 fixed images, as in
  Visual Behavior.
