# v11 confirmatory protocol (locked before the confirmation cohort is used)

Status: LOCKED. Its SHA-256 and the source hash are recorded in
`docs/LOCK_v11.json`, written before any confirmation-cohort run started.

## 1. Question

Networks are built from data-fitted, differentiable single-neuron models of mouse visual-cortex
cells (Allen GFR bank). Their intrinsic parameters are frozen, and only the recurrent magnitudes
(sign fixed by Dale's law) and the readout learn. The questions:

1. **Learning rules.** How well does a cell-local, forward-in-time learning rule (e-prop with
   random feedback, adapted to K-filter GFR cells) learn working-memory tasks compared with the
   exact network gradient (BPTT) and with a frozen recurrent reservoir? Does the rule need the
   cell's own spike-history (adaptation) term β in its eligibility trace?
2. **Biological specificity.** Once each cell's operating point is matched, does the natural
   pairing of a cell's parameters (intact vs within-type shuffle), its heterogeneity (intact vs
   type-mean cells) or the type-based wiring prior (vs degree-preserving rewiring) change how well
   networks learn and generalize?

## 2. Fixed design

- **Cells.** Mouse cells with validation EVR ≥ 0.5, Pyr/PV/SST/VIP labels as in v7.
  Donor-disjoint 50/50 split (seed 1701, stratified by type):
  - development: 298 cells (169/53/44/32)
  - confirmation: 299 cells (170/54/42/33)
  - 0 shared donors.
  - Only the confirmation cohort is used below.
- **Network.** 100 neurons (80 Pyr, 10 PV, 5 SST, 5 VIP) sampled without replacement per network
  seed. Type-prior graph (v7 matrix), recurrent gain 0.6, inhibitory gain 1.5, readout from Pyr
  only, fixed random feedback vector, rate reference 100 Hz.
- **Calibration.** Label-free current sweep: each cell's isolated baseline is set to 5% of its own
  fitted ceiling, with input scale at 25% of its current range. Every identity control is
  recalibrated after manipulation (`matched`), except the declared `reference` control.
- **Tasks** (20 ms bins; training delays 100/200/300 ms; unseen test delays 400/600 ms):
  - T1-early: context-dependent decision, context during the stimulus.
  - T1-late: the same, but the context arrives after the delay.
  - T2-dms: delayed match-to-sample. Sample on channel 0, a simultaneous distractor on channel 1,
    a post-delay probe, and target = sign(sample) × probe.
- **Training.** 3000 updates × 32 trials. Validation (256 ID trials) every 50 updates. The
  selected checkpoint is the one with minimum validation BCE. Learning rates come from the D1
  validation-only search on the development cohort (Section 6).
- **Optimizers.** *Adam* (β1 = 0.9, β2 = 0.999, global-norm clip 1.0) is the primary optimizer.
  *RMS* (per-parameter RMS normalization, no momentum, no clipping) is the secondary optimizer.
- **Seeds.** 16 paired network seeds, 1001–1016. A seed fixes the sampled cells, initial weights,
  graph, feedback vector and every training and test trial, across all conditions.
- **Tests.** 512 trials per split: ID, unseen delay, higher sensory noise (σ = 0.5),
  delay distractor (σ = 0.75), combined.

## 3. Conditions

Family A (learning rules; intact cells, type prior), for each of 3 tasks × 2 optimizers:
`bptt`, `local` (β kept), `local_no_beta` (only the β term removed), `readout_only` (frozen
recurrent weights).

Family B (biology; learners `bptt` and `local`), for each of 3 tasks × 2 optimizers:
- the 2×2×2 factorial: identity {intact, within-type shuffle (matched)} × topology
  {type prior, degree-preserving rewired} × learner {bptt, local};
- type-homogeneous (matched) cells, with each learner.

Family C (descriptive controls, local learner, Adam, 3 tasks):
- within-type shuffle with *reference* calibration (the v7/v8 operating-point confound);
- fast-decay cells (time constants ≤ 20 ms, matched);
- cross-type permutation (matched);
- local and local_no_beta with *symmetric* feedback (readout weights instead of a random vector).

Family D (operating point, descriptive; added after the development rehearsal showed a large
difference between reference- and matched-calibrated shuffles, before locking): intact cells,
type prior, Adam, learners `bptt` and `local`. Each cell's isolated baseline is set to 2%, 10% or
20% of its fitted ceiling; 5% is the main-suite intact condition.

## 4. Endpoints

Primary endpoint: accuracy on conflict trials at unseen delays (`test_delay`,
`accuracy_incongruent`), per network. Secondary endpoints: unseen-delay accuracy, ID accuracy,
combined-OOD accuracy.

## 5. Hypotheses and tests

The unit of analysis is the network seed. All tests are paired two-sided exact sign-flip tests
(65,536 sign assignments), with percentile bootstrap 95% CIs over networks. Effects are in
percentage points (pp).

**Family A** (Holm over 18 tests: 3 contrasts × 3 tasks × 2 optimizers):
- A1: local − local_no_beta > 0
- A2: local − readout_only > 0
- A3: bptt − local > 0

**Family B** (per task and optimizer; Holm over the 7 factorial terms, plus 2 homogeneity
contrasts, so a family of 9):
- the 7 factorial effects (identity, topology, learner and their interactions);
- intact − homogeneous for bptt, and for local.

Equivalence: for each identity, topology or homogeneity contrast that is not significant, it is
called *practically negligible* if its 90% bootstrap CI lies within ±5 pp (TOST, α = 0.05).
Otherwise it is *inconclusive*.

**Family C**: descriptive. Effect, CI and unadjusted p are reported with no confirmatory claims.

**Family D**: descriptive. For each network, the least-squares slope of the primary endpoint
against log2(baseline fraction) over the four levels; an exact sign-flip test on the slopes, and
each level against 5%.

Run count: (12 A+B conditions × 6 task/optimizer suites + 5 C conditions × 3 + 6 D conditions × 3)
× 16 seeds = 1,680 runs.

## 6. Learning rates

The rates are fixed in `results/v11/tune/final_selection.json` (all candidates are in
`all_candidates.csv`). The search had two stages, both on the development cohort with
validation data only:
- stage 1 grid: lr_rec ∈ {3e-4, 1e-3, 3e-3, 1e-2} × lr_out ∈ {0.03, 0.1, 0.3, 1.0}. This grid
  replaced an earlier one that topped out at lr_out = 0.1, where every learner had landed on the
  upper edge; that search is kept in `results/v11/aborted/`;
- stage 2: at each learner's stage-1 lr_rec, lr_out ∈ {3, 10}, because with Adam the reservoir
  and the no-β rule were still improving at 1.0;
- 3 development seeds per candidate;
- criterion: mean best validation BCE over all candidates from both stages. The single upper-edge
  selection (BPTT, T1-early, RMS, lr_out = 10) sits at 100% validation accuracy, where BCE differs
  by 0.0002 from lr_out = 3;
- shuffle, homogeneous, rewired, fast-decay, cross-type and symmetric-feedback conditions use the
  rates of their learner;
- `local_no_beta` uses its own tuned rates.

## 7. Rules

- No run is dropped. A nonfinite or failed run is reported, and its paired contrasts are marked
  not estimable.
- Confirmation results do not feed back into the design. Anything added afterwards is labelled
  exploratory.
- The v8–v10 documents' numbers are not used anywhere.
