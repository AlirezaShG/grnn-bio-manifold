# v12 protocol: representational geometry after local vs global learning, against mouse V1

Status: see the end of this file. The lock is written before any
confirmation-cohort run.

## Question

In a V1 microcircuit made of patch-clamp-fitted GFR neurons, we compare two ways of learning the
same objective: a cell-local, forward-in-time rule (e-prop) and the exact global gradient (BPTT).
- Do they produce different population geometry for natural images?
- Which one is closer to mouse V1 recorded with Neuropixels?

The hypotheses are two-sided. No direction is presumed.

## Why v11 is replaced

v11 used abstract inputs and short-delay tasks. Its representations were dominated by the task
and by the cells' slow filters, and none of its stimuli overlapped with in-vivo data. v12 uses:
- the same visual stimuli as the in-vivo recordings, delivered through a fixed LGN front end;
- measured V1 connectivity;
- two different objectives, so that a rule effect found under both is not a property of one task;
- evaluation on images that were never used in training.

## Fixed design

- **Cells.** GFR bank, donor-disjoint cohorts as in v11 (development: 298 cells, confirmation:
  299 cells). A circuit has 1,000 cells (800 Pyr, 100 PV, 60 SST, 40 VIP), sampled from its
  cohort with replacement. The calibrated baseline is 2% of each cell's fitted ceiling.
- **Connectivity.**
  - Peak connection probabilities for L2/3 Exc/PV/SST/VIP come from the Allen V1DD matrix
    (Ito et al. 2026 repository, `pmax_matrix_v1dd.csv`).
  - E→E connections follow the like-to-like orientation rule of Ito et al. (β = 0.5).
  - Magnitudes are log-normal (σ = 0.8), scaled by the median 90th-percentile PSP per type pair
    (Allen synaptic physiology, `psp_characterization.csv`).
  - Dale's law holds; recurrent gain is 1.0 and inhibitory gain 2.0.
- **Visual input.** A 96° × 128° field on a 48 × 64 grid. The LGN has ON/OFF centre-surround
  units with sustained and transient channels (3,072 units). LGN→V1 weights are fixed: Gabor
  subfields for Pyr and broader, weaker inputs for interneurons. Input gain is 0.2.
- **Plasticity.** Only the recurrent magnitudes (with fixed sign) and the readout learn.
  Exponentiated Adam (Ito et al. 2026) is used for every rule.
- **Objectives**, trained with the same budget and learning rates for every rule:
  - *data*: match the in-vivo distributions of spontaneous rate, preferred-direction rate, OSI
    and DSI for each cell type (Neuropixels, curated by Ito et al.), using gray-screen and
    drifting-grating trials. The e-prop learning signal is the cell's own direct error.
  - *change*: Go/No-Go change detection between two flashed natural-movie frames (Visual Behavior
    task structure), with a weak homeostatic rate term. The e-prop learning signal reaches cells
    through fixed random feedback.
- **Rules.** `bptt`, `local` (e-prop with β), `local_nobeta`, `untrained` (the same circuit before
  training), `readout` (change objective only: frozen recurrent weights) and `local_sym` (change
  objective only: e-prop with symmetric feedback; mechanistic control).
- **Evaluation.** Never trained on:
  - the 118 Brain Observatory natural scenes, 240 ms each, back to back, 10 repeats;
  - drifting gratings for 2 s;
  - gray screen.
- **Mouse data.** VISp units passing Allen QC from Visual Coding Neuropixels sessions, with spike
  counts 0–250 ms after each of 50 presentations per image. Tuning distributions come from the
  curated Ito et al. table.

## Endpoints

- **Primary:** `rsa_dis_all`. This is the Spearman correlation between model and mouse
  natural-scene RDMs (1 − Pearson on z-scored units), divided by the square root of the product
  of the two RDMs' split-half reliabilities.
- **Secondary:**
  - raw RSA;
  - linear CKA;
  - participation ratio (200 units);
  - lifetime-sparseness similarity (1 − KS);
  - image decoding (200 units);
  - mean tuning-distribution similarity (1 − KS over 4 statistics × 4 types);
  - RS/Pyr-only versions.
- **Geometry vs performance:** RSA at every saved checkpoint against that checkpoint's
  validation value. This shows whether a rule difference is explained by task performance.

## Tests

- The unit is the network seed (10 paired seeds per condition, confirmation cohort).
- Exact two-sided sign-flip tests, with Holm correction within each objective's primary family:
  - local vs bptt;
  - local vs local_nobeta;
  - each trained rule vs untrained;
  - local and bptt vs readout (change objective).
- Secondary endpoints are corrected separately.
- No run is dropped.

## Settings fixed by the development sweep (validation only, development cohort)

- data: lr_rec = 0.015 for every rule (best for each rule in {0.002, 0.005, 0.015}; BPTT was
  unstable at 0.015 late in training, so higher values were not tried), lr_out unused, 1,500 steps.
- change: 8 fixed images (the 8 most distinct frames of natural movie 1); lr_out = 0.3 (best for
  BPTT, local and readout in {0.1, 0.3, 1.0}); lr_rec = 0.005 (not tuned); 800 steps.
- Confirmation seeds 1001–1010. Seeds 1001–1004 run on GPU and 1005–1010 on CPU, with all rules
  of a seed on the same device.
- Mouse data: all processed sessions available when the analysis runs (6 sessions requested).

Status: LOCKED by `docs/LOCK_v12.json`.
