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
