# Research protocol: fitted cellular identity, topology and learning

## Status, question and allowed claims

This is a prospective protocol to lock after development. It is not a claim
that confirmation experiments, an adequate power analysis, independent
patch-clamp revalidation, or journal acceptance have already been obtained.
Completed runs must be identified by their manifests and actual results.

The main question is: **Does preserving the association of fitted parameters
within a cell improve delayed context-dependent computation, and does that
effect depend on a cell-type wiring prior and the credit-assignment rule?**
The hypothesis can fail. Intact cells may match or underperform the shuffled
control, topology may have no useful effect, and a local learning rule may fail.
Those outcomes must remain in the report.

The model preserves the distributed Allen GFR equation and frozen fitted
coefficients in the authentic condition. It models firing rates, not
intracellular voltages, detailed morphology, or actual spikes. No measured
connectome connects the sampled cells. Upstream fit quality is inherited;
validation-EVR selection is not a new biological validation, and the upstream
Noise1 training/validation overlap must be disclosed.

## Primary factorial experiment

Use all eight combinations of the following factors, with the same network
seed identifiers and common random task streams across conditions:

| Factor | Low coding | High coding |
|---|---|---|
| Cellular identity | Within-type independent parameter-group shuffle | Intact fitted parameter tuples |
| Topology | Directed degree-preserving rewiring | Cell-type connection-probability prior |
| Credit assignment | Exact BPTT | Local eligibility with beta feedback |

The identity shuffle separately permutes input-filter coefficients,
history-feedback coefficients and the activation group within operational
cell types. It retains their marginals while disrupting their fitted
association; it creates synthetic cells and does not preserve each original
cell's response. Finite samples may retain accidental associations, so retain
the actual mappings and characterize the manipulation.

The graph control preserves every node's in-degree and out-degree, Dale signs,
and each source's outgoing magnitude multiset. The type probabilities are
illustrative modeling assumptions, not measurements for these specimens.
Double-edge switches are not guaranteed uniform samples of the constrained
graph space; record accepted swaps, edge overlap and mixing sensitivity before
strong topology claims.

Input projections, zero initial states, plastic support and calibration stay
matched. Negative fitted filter coefficients remain intact. The tonic-current
calibration uses isolated, label-free probes of the original cell bank and is
then fixed across identity controls. This isolates the manipulation at a
shared external operating point, but nonlinear gain and firing-rate regimes
can still change. Report those changes rather than interpreting every effect
as a pure change in computational identity. Rates use one fixed 100 Hz scale.

## Task and experimental budgets

The default task uses 20 ms bins: 100 ms fixation, 200 ms noisy sensory input,
an independently sampled 100/200/300 ms silent delay, and 60 ms response.
Context indicates which sensory stream determines the correct sign.
Context, relevant sign and irrelevant sign are balanced jointly. Conflict
trials require ignoring an oppositely signed irrelevant stream.

The primary test changes only delay to 400/600 ms. Separate tests increase
sensory noise, introduce a mid-delay distractor, or combine shifts. Do not use
the combined-shift result to attribute an effect specifically to memory.
Context-removal and recurrent-removal tests assess dependence on the intended
mechanisms. The late-context task is a separate sensitivity experiment that
requires retaining both sensory streams; it is not an interchangeable test
of a network trained with early context.

The current confirmation template specifies 100 neurons, with 80/10/5/5
pyramidal/PV/SST/VIP slots; 1,500 updates of 32 trials; validation every
50 updates on 1,024 trials; and 2,048 trials per final test split. It contains
12 paired network seeds, 1001–1012, and the eight primary factorial conditions:
96 planned runs. **Twelve is a planning default, not demonstrated statistical
power.** Use development variability and a prespecified practically meaningful
effect to revisit the required sample size before locking confirmation.

The full condition catalog contains 15 conditions: the factorial eight plus
no-beta learning, frozen recurrence, type-homogeneous cells, fast-decay cells,
cross-type permutation, node perturbation and noise-matched BPTT. The bundled
development factorial pilot uses that full catalog. The confirmation template
reserves the primary eight; confirmatory claims about auxiliary conditions
require adding their runs and multiplicity plan before examining their tests.

## Development, tuning and the lock

Cell cohorts are disjoint by specimen ID, stratified by operational cell type.
Sampling within a network is without replacement. Donors may occur in both
cohorts, and the manifest reports donor overlap. Call this specimen
generalization within the fitted resource, not donor generalization.

The small development pilots use 32 neurons. Before a 100-neuron confirmation
study, repeat development and tuning at the final architecture, duration and
training budget on the development cohort. A learning rate tuned only at the
pilot size is not automatically validated at the larger size.

The tuning command uses development validation only. For each eligible
noiseless, intact, type-prior learner it evaluates three fixed pairs of
recurrent/readout learning rates:

| Candidate | Recurrent rate | Readout rate |
|---|---:|---:|
| 0 | 0.0003 | 0.01 |
| 1 | 0.001 | 0.03 |
| 2 | 0.003 | 0.10 |

Every learner receives the same seeds, update count, batch size and number of
distinct candidates, including the reservoir's three distinct readout rates.
Select by mean per-seed best validation BCE; ties choose the first declared
candidate. Ambiguous representatives or incomplete/nonfinite candidate
results fail explicitly. The command does not run final tests, population
analyses or interventions. This is a limited declared search, not proof that
each algorithm is optimally tuned.

For example, after selecting an appropriate development configuration:

    python -m grnn_bio.cli tune --config configs/pilot.json --output results/tuning

Retain the selection record and selected configuration. Transfer the locked
learning-rate settings to the intended confirmation configuration, together
with the final architecture and protocol. The distributed confirmation
template has no completed tuning selections and must not be represented as a
previously optimized configuration. Noisy BPTT and node perturbation are
outside this noiseless tuner and require their own equal-budget,
noise-matched development study before comparative performance claims.

Lock source/data hashes, cohort split, seed list, task, primary endpoint,
candidate search, selected rates, optimizer, training budget, stopping rule,
exclusions and analysis family before confirmation. All updates are performed;
the selected checkpoint minimizes validation BCE, including the untrained
checkpoint if it remains best. Final test outcomes cannot determine any of
these choices. Do not drop failed or poorly performing seeds; diagnose and
report failures under a fixed rule.

## Primary inference

The primary outcome is **incongruent-trial accuracy on the delay-only OOD test**.
Accuracy is stored as a fraction: 0.01 is one percentage point.
For each paired network seed compute three high-minus-low main effects,
three differences of differences averaged over the remaining factor, and
one three-way difference of differences of differences. The positive
directions are intact identity, type-prior wiring and local learning.

Treat these seven contrasts as one family and apply Holm correction at
familywise alpha 0.05. Report all effects, network-bootstrap intervals and
adjusted p values, including negative effects. The paired sign-flip test
assumes exchangeability/symmetry of network-level contrasts under its null;
it is not assumption-free. Missing factorial cells make the family
non-estimable instead of silently removing a seed. Duplicate cells, mixed
evaluation splits and invalid seed identifiers are rejected.

The unit of inference is the independently initialized/sampled **network run
conditional on this finite fitted-cell resource**, not a neuron, trial, bin,
or checkpoint. Reusing fitted specimens across different networks does not
create additional biological subjects. Per-cell correlations remain
descriptive or are summarized across networks; pooled-cell p values are not
evidence of biological population effects.

The runner also exports condition-versus-reference comparisons with their own
Holm adjustment. These are secondary/exploratory and cannot replace the
predeclared seven-contrast primary family. A small pilot's intervals and
p values are exploratory, regardless of their numerical size.

## Mechanism controls and interpretation

The main algorithmic comparison uses the common projected Adam optimizer.
Adam is an engineering optimizer, not an established biological plasticity
mechanism. Run the separate SGD sensitivity configuration, which disables
global gradient clipping and the upper magnitude cap while preserving
nonnegative Dale-constrained magnitudes. Keep its conclusions separate;
success only with Adam limits claims of biological plausibility.

The no-beta control removes only beta's eligibility derivative. Frozen
recurrence tests whether learning the recurrent graph is necessary. The
homogeneous, fast-decay and cross-type controls are explicitly synthetic;
they do not all isolate the same property or match every nonlinear response.

Node perturbation uses a noisy-trajectory objective and a scalar-loss-modulated
local score. Its default clean baseline requires a second rollout. Compare it
with BPTT at the same injected-noise distribution, retain supervised-readout
qualifications, and report estimator variance. The default matched number of
training trials and updates is **sample matching**, not compute matching.
Report forward rollouts, simulated trial bins and measured training time.
Equal-wall-time or equal-simulated-cost conclusions require a separately
declared budget; different memory/gradient operations have different costs.

## Analysis, reporting and decisions

Fit PCA, axes and decoders on an independent analysis-fit bank; assess them on
an independent analysis-evaluation bank. Align variable-delay trials by
declared phase, and separate representation from behavior. Geometry and
checkpoint evolution do not prove attractor dynamics or learning causation.
Fixed-point/Jacobian diagnostics are optional and conditional on convergence.

Rate suppression and state pulses are interventions within the simulator.
Use shared inputs/noise, multiple doses and phases, and matched-count controls.
Do not select trials by baseline correctness or describe these as replicated
opsin experiments. Preserve per-trial outputs and network-level summaries.

The final report must show calibration errors, silence/saturation, current
excursions beyond fitted scales, training curves, selected steps, all network
outcomes, diagnostic failures and uncertainty. At-chance behavior, a reservoir
matching learned recurrence, a reversed identity effect, or failure of strict
SGD is scientifically informative. If these occur, narrow the conclusions and
revisit hypotheses in a new development cycle; do not alter confirmation
conditions until an attractive result appears.

Archive exact configurations, source/data provenance, sampled-cell mappings,
initial/final checkpoints and optimizer states, test predictions, analysis
fits, numerical checks and runtime information. Code readiness and a complete
protocol support reproducibility; publishability depends on completed,
credible evidence and a defensible contribution.
