# Task design, analysis, interventions and statistical estimands

## Scope and interpretation

The redesigned task is an explicitly synthetic context-dependent delayed
decision task inspired by Mante et al. (2013), not a reproduction of that
experiment's stimuli, trial schedule, neural recordings or results [1]. A
successful model demonstrates computation within the declared simulation. It
does not by itself establish cortical working-memory physiology or the
biological implementation of a learning rule.

The independent replication unit for comparing learning algorithms is a
network run: a cell sample, connectivity realization, weight initialization,
training stream and prescribed random seeds. Neurons, trials and time bins
within a network do not increase the number of independent network runs. This
distinction is central to the nested-data problem discussed by Aarts et al.
[3]. All intervals and condition comparisons in `grnn_bio.statistics` use
network runs. Inference remains conditional on the finite Allen fitted-cell
resource, the sampling scheme and the chosen parameter ranges.

## Why the supplied analysis needed replacement

The v6 project made the following methodological shortcuts:

| Original behavior | Threat to interpretation | Replacement |
|---|---|---|
| Choice axis and d-prime calculated on the same trials | High-dimensional axis selection inflates separation | Axis fit on analysis-fit trials; signed d-prime evaluated on separate trials |
| Delay indexed as `9:17`, pyramidal population implicitly 80 cells | Silent mistakes after changing temporal or population parameters | Trial-level phase masks and explicit neuron indices |
| Pooled Pearson p values across 480 cells from six networks | Pseudoreplication and repeated fitted-cell identities | Within-network associations, unique-cell aggregation and network-level intervals |
| Welch tests on conditions sharing seeds | Discards the paired design | Per-run differences, paired sign flips, run-bootstrap intervals |
| Many p values without a declared family | Uncontrolled multiplicity | Raw and Holm-adjusted p values for a declared contrast family |
| Psychometric/noise batches regenerated inside the model loop | Algorithms see different random examples | Reusable immutable-by-convention batches, caller-owned split RNGs |
| Zero-coherence trials assigned target +1 | Artificially rewards an arbitrary response | Target zero marked ambiguous and excluded from accuracy/loss |
| Only the first seed's optogenetics/geometry presented | Unreplicated mechanistic claims | APIs operate per run; the study runner must retain all seed-level outputs |
| Recovery clipped to [0,100] and called attractor contraction | Hides divergence and confuses decay with recurrent stability | Unclipped full-state/rate displacement and explicitly finite-horizon language |
| “Orthogonal” pulse sampled as a generic Gaussian direction | Assay differs from its description | Clearly labeled random physical-current pulse |
| Complete cell-class silence for the whole trial at fixed cell indices | Timing, dose and population specificity confounded | Named classes, phase/dose masks, paired same-size controls |

Circular use of fitting and evaluation observations is a general source of
optimistic selective statistics [2]. The new code prevents the most direct
same-object reuse and separates fitting from evaluation interfaces; the caller
must still generate genuinely independent trials, rather than passing copies
of the same data.

## Task contract

`grnn_bio.task.TaskConfig` owns task parameters independently of network and
optimizer configuration. A call to `generate_batch(config, n_trials, rng)`
returns a `TrialBatch` with:

- `X[B,T,5]`: sensory A, sensory B, context A, context B, response/go.
- `y[B]`: the sign of the context-relevant latent coherence, normally -1 or +1.
- `decision_mask`, `delay_mask`, `stimulus_mask`, `context_mask`,
  `fixation_mask`, `distractor_mask`, `valid_mask`: Boolean `[B,T]` arrays.
- `meta`: coherence, context, congruence, delay, per-trial decision/end timing,
  noise settings and a `valid_choice` indicator.

Every training batch jointly balances context, relevant sign and irrelevant
sign. With a multiple of eight trials the eight cells are exactly equal;
otherwise their counts differ by at most one, with a randomized partial block.
Coherence magnitudes and trial delays are sampled independently of response.
All padding inputs are zero and excluded from measurement by masks. A
response cue communicates *when* to respond, never *which* response is correct.

Defaults are 20 ms bins, 5 fixation bins, 10 sensory bins, delays sampled from
5/10/15 bins, and a 3-bin response window. These are transparent engineering
defaults, not experimentally established optimum values. A publication study
must declare task difficulty and delay support before inspecting held-out
outcomes. Multiple task durations or noise levels are sensitivity analyses,
not a license to retain only favorable variants.

### Early and late context

With `context_timing="early"`, context is available during fixation and sensory
presentation. Both sensory streams and context disappear during the delay.
The network must retain the selected decision through a variable silent delay.

With `context_timing="late"`, the sensory streams arrive without context. A
separate `t_context` epoch appears *after* the silent delay and before the
response cue. The network therefore has to preserve information about both
streams until selection becomes possible. This variant is a stronger test of
flexible storage but remains solvable by intrinsic slow filters in principle;
the no-recurrence and readout-only controls are essential.

### Generalization axes

Call the same generator with explicit overrides to separate shifts:

| Evaluation | Override | What changes |
|---|---|---|
| ID test | None | New trials from the declared training distribution |
| Delay OOD | `delay_bins=(...)` outside training support | Retention duration only |
| Noise OOD | `noise_std=...` | Sensory observation noise only |
| Distractor OOD | `distractor_std=...` | Short independent sensory perturbation during delay |
| Joint OOD | Explicit combination of overrides | Prespecified compounded shift |
| Psychometrics | `relevant_coherence=c`, optionally `context_rule=0/1` | Fixed signed relevant evidence, random irrelevant evidence |

The generator does not decide which set is training, validation or test. The
runner creates deterministic named RNG namespaces, saves the exact split
configuration and uses identical evaluation draws for paired algorithms. Test
splits must never select hyperparameters or checkpoints. An OOD condition must
be named accurately: a delay equal to a training delay is not delay OOD even
if another attribute changes.

At zero relevant coherence there is no uniquely correct latent sign. Such
trials have `y=0` and `valid_choice=False`; they contribute to
`choice_probability_plus`, never target accuracy or target loss. Training on
these targets is prohibited by the learner. Psychometric curves should include
both contexts and record trial denominators, rather than treating all curves
as if they used an identical sample size.

### Behavior measurements

`decision_values(logits,batch)` averages each trial's own response window,
including variable timing. `behavior_metrics` reports fractions in [0,1], not
percentages, and includes scored-trial counts, context-specific accuracy,
congruent/incongruent accuracy, balanced accuracy, decision MSE, positive-choice
probability and tie fraction. Exact-zero decision values use a documented +1
tie break. Undefined subgroups return NaN, later serialized as unavailable,
rather than a misleading zero result. “Incongruent error” is not renamed
“distractor leakage”: memory loss, a response bias and distractor intrusion
can all produce such errors.

## Population analysis without selection leakage

### Phase-specific features

`phase_average(rates,batch,phase="delay")` computes one vector per trial using
that trial's phase mask. It never includes padding, guesses phase boundaries
or assumes a population size. The analysis accepts either rates in Hz or a
clearly declared normalized rate. Never mix the two across a comparison.

The participation ratio is computed from the eigenvalues of the covariance
of the phase-averaged trial vectors:

\[
\mathrm{PR}=\frac{(\sum_j\lambda_j)^2}{\sum_j\lambda_j^2}.
\]

This measures linear dimension of *between-trial phase-average variation*.
It is not the dimension of all within-trial trajectories, the number of
independent neurons, or the dimension of a nonlinear attractor. Constant
activity returns an undefined value, not a positive dimension.

### Fitting and evaluation

`fit_population_analysis` fits on a dedicated analysis-fit split:

1. A mean-centered PCA basis of phase-averaged activity.
2. A raw class-mean difference choice axis.
3. A raw context-mean difference context axis.
4. Standardized ridge probes for choice and context.

Each ridge probe fits its centering and feature scales on fitting trials only.
Its objective is `mean_squared_error + alpha * squared_weight_norm`; the
regularization coefficient therefore does not change meaning with the number
of fitting trials. The default `alpha=1` is fixed, not optimized on test
labels. If selecting alpha, use an inner validation split and record that
selection.

`PopulationFit.evaluate` or `analyze_population` then measures on independent
evaluation trials:

- Held-out choice decoding and context decoding accuracy.
- **Signed** held-out d-prime along the fitting-derived choice axis. A negative
  d-prime remains negative. If an axis or projected variance is degenerate the
  result is explicitly undefined.
- Held-out PCA reconstruction fraction in the fitting-derived basis.
- Descriptive evaluation participation ratio.
- An optional readout projection onto that fixed PCA basis.

The context-choice angle is explicitly named an angle between fitted *axes*,
not a multi-dimensional subspace result. It is a fitting-set descriptor, not
independent mechanistic evidence. PCA axes are called PC1, PC2, etc.; they
receive no automatic “choice” or “context” interpretation.

`PopulationFit.condition_trajectories` projects held-out paths into the fitted
PC coordinates. It returns the number of trials contributing at every time
bin because variable-length trials leave the valid set at different times.
These illustrative average paths must not replace per-seed outcome analysis.

### Temporal generalization

`temporal_generalization` divides each trial's silent delay into a declared
number of nonempty relative-time bins. A choice probe fit at one delay bin is
tested at every evaluation bin. The resulting train-time × test-time matrix
can distinguish stable decodability from time-specific coding. Scaling and
weights for each row are fit exclusively on analysis-fit trials. This does
not establish a particular storage mechanism without interventions and
appropriate controls.

## Paired interventions

### Idealized rate silencing

`evaluate_rate_interventions` calls the same circuit and the same input batch
for intact, targeted and size-matched conditions. A supplied background-current
noise tensor is reused exactly. Multipliers act only on selected cells and
declared phases; dose 0 is a numerical sham and dose 1 suppresses their output
rate completely. The function returns both summary rows and individual trial
decision values/predictions, enabling later reanalysis.

For inhibitory cell types the same-size control first samples other inhibitory
cells. If too few exist it uses all other cells and labels the fallback.
Controls preserve intervention count, not baseline activity, synaptic degree,
projection geometry or molecular identity. A strong type-specific claim needs
multiple control draws and additional activity/connectivity-matched analyses.
The appropriate causal statement is an intervention effect **inside this
specified model**. No light propagation, opsin kinetics or biological
photocurrent is simulated.

Run the phase/dose suite on every declared network seed. Summarize the change
in accuracy and prediction flips per network, then compare those seed-level
effects. Trial-level pairs improve measurement but do not convert one network
into hundreds of independent network replications. Use held-out task trials
and freeze training while intervening.

### Current-pulse recovery

`evaluate_pulse_recovery` adds a zero-mean random current pulse (pA) halfway
through each trial's silent delay. It measures displacement both in firing
rates and in the complete internal GFR state. The endpoint is the final
silent-delay bin, before context/go inputs return. The intact and pulsed runs
share the identical background noise.

The circuit API accepts background noise in dimensionless drive units, which
are multiplied by each cell's fixed current scale. The assay divides the
sampled physical pA pulse by that scale before passing it to the circuit; a
20 pA pulse therefore remains 20 pA rather than being scaled twice. This unit
conversion is tested explicitly.

`perturbation_recovery` reports the raw endpoint-to-initial displacement ratio
and `1 - ratio`, with no clipping. A ratio above 1 denotes expansion, and a
zero initial displacement makes the ratio undefined. All per-trial distances
are retained. Hidden-state Euclidean distance depends on the state
coordinates/scales, so it must not be equated with a measured biophysical
distance.

A decaying transient can produce apparent recovery with no recurrent
attractor. Accordingly, this suite explicitly reports
`attractor_claim_supported=False`. Fixed-point residuals, local Jacobians,
perturbation directions, basin tests and matched no-recurrence controls are
needed before interpreting a network as an attractor system. Fixed-point and
linearization analyses have a distinct role in mechanistic interpretation
[4,5].

## Statistical reporting

### Paired algorithm and ablation effects

For each prespecified split and outcome, match network runs by their declared
paired run IDs. `paired_condition_table` rejects missing/duplicate pairs rather
than silently treating incomplete outputs as a complete design. It computes:

- The condition-minus-reference difference for each network run.
- Mean difference and a percentile bootstrap interval, resampling whole pairs.
- A paired standardized difference when it is defined.
- A two-sided sign-flip test, enumerating all assignments through 16 pairs and
  using Monte Carlo with a +1 correction for larger samples.
- Holm correction across the declared condition-versus-reference family.

The sign-flip null requires exchangeability/symmetry of paired condition
differences; pairing alone does not establish that assumption. Report this
assumption, the direction of subtraction and the estimand alongside results.
Finite-sample p values cannot be more precise than the randomization space:
with six nonzero paired differences the smallest exact two-sided p is
`2/2**6 = 0.03125`. Six seeds are therefore inadequate for many subtle or
multiply corrected claims. The number of seeds should be selected using a
declared precision/power plan informed by a separate pilot.

With a single run, the module returns point estimates only: p values and
intervals are unavailable. Fewer than ten runs are explicitly marked as a
small-sample pilot; ten is a warning threshold, not a promise of adequate
power. Bootstrap intervals can also be poorly calibrated with very small or
degenerate samples. Raw per-seed outcomes must always accompany summary
tables. Rounding is for display only.

Holm correction includes declared but unavailable contrasts in the family
size. It does not remove inconvenient missing experiments to reduce the
correction. The caller must define the family in advance; separate calls for
many outcomes must not be presented as one corrected comprehensive family.

### Single-cell phenotype associations

`network_level_cell_association` first aggregates duplicated Allen cell IDs
within a network, computes a descriptive correlation per network, and combines
those correlations in Fisher-z space. The interval resamples networks, not
neurons. No pooled-cell p value is produced. Filter to one condition before
calling, or use a run ID incorporating all condition factors.

This analysis cannot establish that a cell parameter causes its learned
functional role. Confounds include cell class, layer, fitted gain and network
input/output allocation. A stronger follow-up uses prespecified covariates,
crossed network/cell effects, disjoint cell-cohort validation and controlled
parameter interventions. The present summary is explicitly conditional on the
distributed fitted-cell resource.

## Validity checks supplied with the code

Run `python -m unittest discover -s tests -p test_task_analysis.py -v` from the
project root. Checks exercise scientific failure modes rather than only
repeating code formulas: balanced designs, variable readout timing, late
context, zero-coherence ambiguity, no fitting leakage, held-out sign reversal,
null decoding, unclipped expansion, finite exact-p resolution, incomplete
pairs, nested cells and shared-noise phase-specific interventions. They validate
implementation contracts, not the substantive research hypothesis.

## Primary references

1. Mante V, Sussillo D, Shenoy KV, Newsome WT. Context-dependent computation by
   recurrent dynamics in prefrontal cortex. *Nature* 503, 78–84 (2013).
   https://doi.org/10.1038/nature12742
2. Kriegeskorte N, Simmons WK, Bellgowan PSF, Baker CI. Circular analysis in
   systems neuroscience: the dangers of double dipping. *Nature Neuroscience*
   12, 535–540 (2009). https://doi.org/10.1038/nn.2303
3. Aarts E, Verhage M, Veenvliet JV, Dolan CV, van der Sluis S. A solution to
   dependency: using multilevel analysis to accommodate nested data.
   *Nature Neuroscience* 17, 491–496 (2014). https://doi.org/10.1038/nn.3648
4. Sussillo D, Barak O. Opening the black box: low-dimensional dynamics in
   high-dimensional recurrent neural networks. *Neural Computation* 25,
   626–649 (2013). https://pubmed.ncbi.nlm.nih.gov/23272922/
5. Maheswaranathan N, Williams AH, Golub MD, Ganguli S, Sussillo D. Universality
   and individuality in neural dynamics across large populations of recurrent
   networks. *NeurIPS* 32 (2019). https://arxiv.org/abs/1907.08549
