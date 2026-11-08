# Fixed-physiology GFR learning experiments — version 7

A reproducible research package for asking how **fitted single-cell diversity**, a **synthetic cell-type connectivity prior**, and **credit assignment** interact during context-dependent delayed decisions.

This version preserves the Allen GFR parameters in the primary condition, implements exact network BPTT and a genuine forward-time local approximation, and separates training, model selection, final testing and analysis fitting. It replaces the unsupported physiological and statistical claims in the supplied v6 setup. The old files are retained under `legacy_v6/` for audit only.

**Status:** executable and numerically checked research implementation. Supplied development runs are exploratory pilots. They are not an independent biological validation or a claim of journal acceptance. Read `docs/RESEARCH_PROTOCOL.md` and the release validation record before interpreting any performance difference.

## Source resource and fidelity

The upstream resource is [AllenInstitute/GRNN](https://github.com/AllenInstitute/GRNN), pinned to commit **835a535f6831b88339c6a5c17180fdbc504f7448**. The associated paper is Yeung, Ganjam, Zhang, Mihalas and Lu, **“A dataset of differentiable biologically derived single-neuron models,” Cell Reports Methods (2026)**, [DOI 10.1016/j.crmeth.2026.101638](https://doi.org/10.1016/j.crmeth.2026.101638).

The 1,124 supplied `(20 ms, 20 ms)` fitted records match the corresponding subset of the pinned upstream dataset after parsing. The supplied metadata values also match their upstream specimen records. This was checked against the full upstream resource; see `provenance/upstream/PROVENANCE_VERIFICATION.json`.

GFR neurons are differentiable **firing-rate models**, fitted to slice electrophysiology. They are not conductance-based compartmental neurons, spike-by-spike reconstructions, or a measured connectome. Their eight history filters must not be renamed somatic/apical compartments without a separate identification study. The recurrent graph in this package is a synthetic hypothesis inspired by cell-type motifs.

The primary recurrence, using pA and Hz, is:

```text
h[i,k,t] = (1-ds[i,k]) h[i,k,t-1] + a[i,k] I[i,t] + beta[i,k] r[i,t-1]
r[i,t] = gamma_hz[i] ReLU(tanh(sum_d coeff[i,d]^2 ((mean_k(h)-threshold[i])/max_current[i])^d))
```

Stored `a` and `beta` already belong to the discrete recurrence. There is **no additional dt multiplier**. Rates are converted from upstream spikes/ms to Hz once. All fitted positive and negative taps, activation coefficients, thresholds, ceilings and decay coefficients are frozen in the `authentic` condition. State and rate history start at zero; fixation is an explicit trial phase.

Recurrent currents and the readout use a shared **100 Hz physical reference scale**, independent of cell identity. This avoids cancelling fitted rate-ceiling diversity with per-cell normalization. A label-free isolated-current sweep chooses operating offsets on the original intact bank. Those offsets and scales remain fixed across identity controls. This is a disclosed external current calibration, not a modification or new validation of the fitted cells.

## Quick start

Python 3.10 or later is required. Work from this extracted project directory:

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio validate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \
  --config configs/smoke.json --output results/my_smoke
```

No download, PyTorch installation or GPU is required for the included experiments. `requirements-tested.txt` records the exact four numerical-library versions used for release verification. `python -m unittest discover -s tests -v` runs the tests directly. The original source snapshot uses PyTorch, but the new implementation uses independently verified NumPy equations and analytical derivatives; no autograd run is claimed.

The shared primary rates are in Hz, time in milliseconds, and injected currents in pA. One training step means **one minibatch update**, not an epoch over a finite dataset.

### Available runs

```bash
# Inspect a plan without training
python -m grnn_bio plan --config configs/factorial_pilot.json

# Four-method development pilot
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \
  --config configs/pilot.json --output results/my_core_pilot

# 2x2x2 factorial plus seven mechanistic/control conditions: 45 paired pilot runs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \
  --config configs/factorial_pilot.json --output results/my_factorial_pilot

# Resume an interrupted run; code, inputs and configuration must still match
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \
  --config configs/factorial_pilot.json --output results/my_factorial_pilot --resume

# Development-only equal-budget learning-rate search; no test assays are called
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio tune \
  --config configs/pilot.json --output results/my_tuning

# Selected rates on fresh development seeds 303/404/505: 12 follow-up runs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \
  --config configs/tuned_pilot.json --output results/my_tuned_pilot

# Secondary task and stricter optimizer sensitivity (separate studies)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \
  --config configs/late_context.json --output results/my_late_context
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m grnn_bio run \
  --config configs/sgd_sensitivity.json --output results/my_sgd
```

The optional editable entry point `python -m pip install -e .` provides the same `grnn-bio` command. Keep the checkout's `data/`, `model/`, `configs/` and `tests/` alongside its package. A standalone wheel without these resources is not the supported research distribution.

`configs/confirmation_template.json` contains a **future 96-run** plan: eight factorial cells × 12 paired seeds, 100 neurons, 1,500 updates per run, and a reserved specimen cohort. It is a template, not an executed or externally preregistered study. Fill in development-selected learning rates, freeze all hypotheses/settings and perform a precision/power calculation before treating it as confirmatory. A late-context factorial template is provided separately. Do not reuse inspected pilot tests for confirmation.

## Executed development results

The release includes **45 initial pilot runs, 36 validation-only tuning executions and 12 selected-rate follow-up runs**, plus a passing **110-test** validation record. Those 93 training executions are not 93 independent replicates: each condition and paired comparison has **three network runs**. The follow-up uses fresh development seeds 303/404/505, not the reserved confirmation cohort.

For the tuned follow-up, BPTT achieved **94.40 ± 3.93% ID accuracy** and **71.61 ± 11.52% unseen-delay accuracy**; the local eligibility rule achieved **82.81 ± 4.99%** and **69.66 ± 14.12%**, respectively (mean ± sample SD, n=3). All seven primary contrasts in the initial factorial pilot have **Holm-adjusted p = 1.0**. Shuffling also changed the whole-trial firing-rate and saturation regime, which limits a covariance-specific interpretation. See [Executed development results](docs/EXECUTED_RESULTS.md) for the four-method table, exact accounting, raw evidence paths and limitations. The 96-run confirmation template remains unexecuted.

## Methods and controls

| Method | Recurrent update | Teaching information | Important qualification |
|---|---|---|---|
| `bptt` | Full analytical network gradient | Decision error propagated through all recurrent paths | Nonlocal and backward in time |
| `local` | K-channel forward eligibility, including intrinsic beta Jacobian | Fixed random projection of current decision error | Approximate network gradient; not “exact biological e-prop” |
| `local_no_beta` | Same local rule with only beta's derivative removed | Same feedback and trainable support | All K channels and forward dynamics remain unchanged; not advertised as RFLO |
| `readout_only` | Frozen recurrent reservoir | Readout delta rule | Establishes how much the task can be solved without recurrent learning |
| `node_perturbation` | Gaussian likelihood-ratio score × scalar trial-loss advantage | Scalar modulator plus local perturbation history | Finite-noise objective; clean-baseline extra rollout is counted |

Every method uses the same per-trial average of decision-bin binary cross-entropy. Its decision is the sign of mean decision logits. Input projections and intrinsic physiology stay fixed. Recurrent magnitudes obey the same mask and Dale constraints. The primary optimizer is shared projected Adam; **Adam and global gradient clipping are not biological plasticity mechanisms**. The SGD sensitivity configuration removes those conveniences. The scalar-modulator model is still a computational learning hypothesis, not evidence that a circuit can sense its exact artificial Gaussian perturbation.

The scalar-modulator claim applies to the **recurrent update** in node perturbation. Its readout still uses a supervised delta rule, as do the other methods. It is not an entirely reward-only end-to-end agent.

The identity controls are explicit:

- `within_type_shuffle`: independently permute input-history vectors, firing-history vectors, and intact activation tuples within each class. Marginal distributions are preserved while within-cell covariance is disrupted.
- `cross_type_permutation`: a bijection of entire cell tuples across fixed anatomical slots. No duplication or parameter-vector splicing is hidden in the operation.
- `type_homogeneous`: within-class mean coefficients, marked synthetic. Nonlinear response gain is **not** asserted to be matched.
- `fast_decay`: accelerate slow `ds` while rescaling `a` and `beta` to preserve `a/ds` and `beta/ds`. Static equations are matched; full transient and stability behavior need not be.

The primary topology control performs directed double-edge swaps, preserving **both** incoming and outgoing degree. Each source's outgoing magnitude multiset, total outgoing strength and Dale sign are preserved. Finite rewiring is not claimed to sample uniformly; accepted swaps and edge overlap are recorded. The numerical type-probability matrix is a modeling assumption, not an empirical reconstruction of the sampled cells. No hand-wired task-specific 40/40 sensory communities or fixed indices 80:100 remain.

## Task and evaluation

Inputs are sensory A, sensory B, context A, context B and a go cue. Context × relevant sign × irrelevant sign is balanced exactly when trial counts are multiples of eight. Delay duration is sampled independently of the target; explicit masks exclude padding. Training uses 100, 200 and 300 ms silent delays by default. OOD delays are 400 and 600 ms. The late-context task reveals the rule after the delay and must be trained as a separate task, not silently mixed into a primary assay.

The selected checkpoint minimizes BCE on one fixed validation bank. Training always consumes the stated update budget. Independent named RNG streams isolate training, validation, ID test, each OOD test, analysis fitting, analysis evaluation, interventions and perturbations. Conditions paired by network seed use the same cells, initial shared parameters, task examples and appropriately paired noise. Test labels never select a checkpoint or a learning rate.

Final tests separate ID, unseen delay, increased sensory noise, delay distractor and their combined shift. Both context accuracies and conflict/nonconflict accuracies are reported. Psychometrics include both contexts and zero evidence; zero-coherence trials have no invented correct class. Inference-time context removal and recurrence removal quantify shortcuts on exactly the same trials.

Cell admission uses the original **validation** EVR, not test EVR. A source limitation remains: upstream preprocessing uses Noise 1 during both fitting and validation. The new code cannot turn that inherited score into untouched physiological validation. The development/confirmation split is disjoint by specimen ID, **not necessarily donor ID**; overlap is reported. Revalidating physiology on raw held-out sweeps and donor-disjoint cohorts is separate work.

## Analysis after learning

The following analyses are executed for every paired run when enabled:

- Phase-specific population geometry, participation ratio, held-out choice/context decoding and signed d-prime. PCA, standardization, regression coefficients and discrimination axes are fit on an independent analysis-fit bank.
- Held-out condition trajectories in fitted PC coordinates, with time-specific trial counts; phase-normalized temporal decoding during delay; saved checkpoint geometry on a common analysis bank.
- Exact/local/no-beta gradient alignment and temporal full-gradient RMS on independent diagnostic trials. Zero sensitivity remains undefined/inconclusive rather than a perfect numerical result.
- Per-cell descriptive role tables, original electrophysiological metadata and network-level association summaries. Synthetic tuples are identified; individual cells or time bins are not counted as independent network replicates.
- Phase/dose-specific idealized rate-clamp interventions for PV, SST and VIP, plus sham and matched-size controls. The same exogenous noise is reused for each pair. This is virtual rate silencing, not opsin photophysics.
- Physical-current pulse sensitivity with unmodified negative/divergent recovery values and full GFR hidden-state displacement. Finite recovery alone is not an attractor test.
- Optional full NK-state fixed-point solving, residuals and Jacobian eigenvalues. Stability is reported only for sufficiently stationary, differentiable candidates; solver failure is retained.

The primary complete factorial crosses intact/shuffled identity × type-prior/degree-rewired topology × BPTT/local learning. Seven predeclared factorial effects share a Holm family on unseen-delay conflict accuracy. Other condition contrasts are exploratory. Intervals resample complete network runs; paired sign-flip tests retain pairing and never silently discard missing runs. Inference is conditional on this fitted-cell resource, not automatically about biological populations.

## Saved evidence and reproduction

Each run has a full manifest, source/config/input hashes, sampled cell IDs and donor metadata, unchanged intrinsic fingerprint, initialization and graph diagnostics, learning curve, safe NPZ best/last checkpoints including optimizer state, per-trial predictions, model analysis and intervention tables. Completed runs are reused only with an exact identity match. Restoring a checkpoint validates immutable construction and commits candidate state atomically. Python pickle is not used for new checkpoints.

`results/validation/` holds the actual release validation record. Supplied `results/pilot/` folders hold only executed development runs. Aggregated figures display individual network runs and their mean, with no transformation into journal-ready significance claims. Full raw per-trial records accompany summaries. Rerun plotting with:

```bash
python -m grnn_bio plot --output results/pilot/factorial
```

The detailed rationale is in `docs/UPSTREAM_AUDIT.md`, `docs/LEARNING_RULES.md`, `docs/ANALYSIS_METHODS.md`, `docs/STATE_DYNAMICS.md`, `docs/RESEARCH_PROTOCOL.md` and the companion Persian change report. `docs/MIGRATION.md` maps original modules to their replacements.

## What would support a journal submission?

A credible manuscript needs a prespecified scientific question, robust effects across paired seeds and both tasks, equal tuning budgets, operating-current/topology/noise sensitivity, clearly separated biological assumptions, and independent validation of any biological generalization. High task accuracy alone does not provide those. Strong negative results—including a competitive frozen reservoir or failure of local learning—must remain visible. No positive result or novelty claim is guaranteed by this code release.

## Attribution and licensing

The original Allen software notice is preserved at `provenance/upstream/LICENSE.txt`. It is BSD-like with a noncommercial clause; it is not MIT or unrestricted BSD. The upstream dataset metadata separately declares CC-BY-4.0 and contains stale placeholder metadata; the distinction is recorded. This package does not silently relicense third-party code/data or the user-supplied project. See `THIRD_PARTY_NOTICES.md` before redistributing.
