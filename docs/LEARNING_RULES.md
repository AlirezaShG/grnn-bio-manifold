# Learning rules, derivations and verification

## Scope and terminology

This implementation compares learning rules on a common recurrent network of
fixed, fitted Generalized Firing Rate (GFR) neurons. Intrinsic parameters,
activation parameters, cell identity, input projections, anatomical support and
outgoing synaptic signs do not change during a training run. The trainable
parameters are recurrent magnitudes on existing edges, readout weights on their
declared support, and readout bias.

The code is a rate-model study. An exact GFR forward computation is not a claim
that a simulated network reproduces intracellular voltages, dendritic
compartments, actual spikes, a measured connectome, or in-vivo plasticity. The
local learning rule uses an approximate teaching signal. It is named
*local_beta*, not “exact e-prop” or “biologically proven e-prop.” Its no-beta
control is named *local_no_beta*, not RFLO, because it is a specific ablation of
this multi-filter rule and is not a reproduction of Murray's original RFLO
network and learning equations.

The original Allen implementation and fitted dataset are pinned to commit
[835a535f6831b88339c6a5c17180fdbc504f7448](https://github.com/AllenInstitute/GRNN/tree/835a535f6831b88339c6a5c17180fdbc504f7448).
The saved input and feedback coefficients are used in the discrete recurrence
as stored; an additional multiplication by dt would change the fitted model.

## Common dynamics, units and objective

Let \(i\) identify the postsynaptic neuron, \(j\) the presynaptic neuron, and
\(k\in\{1,\ldots,K\}\) a fitted internal filter. Every neuron has its own frozen
coefficient vectors \(a_i,\beta_i,d_i\). Rate is represented in **Hz**, current in
**pA**, and one step is the fitted bin width, normally 20 ms. The following is
the implemented discrete model:

$$
\begin{aligned}
q_{j,t} &= r_{j,t}/R, \qquad R=100\,\mathrm{Hz},\\
W_{ij} &= M_{ij}s_jw_{ij}, \qquad w_{ij}\geq0,\\
I_{i,t} &= I^0_i+S_i\left[(U x_t)_i+\sum_jW_{ij}q_{j,t-1}+\xi_{i,t}\right],\\
h_{i,t}^{(k)} &= (1-d_{ik})h_{i,t-1}^{(k)}
                +a_{ik}I_{i,t}+\beta_{ik}r_{i,t-1},\\
r_{i,t} &= g_i\left(K^{-1}\sum_k h_{i,t}^{(k)}\right),\\
z_t &= \sum_i v_iq_{i,t}+b.
\end{aligned}
$$

Here M[i,j] means the directed edge \(j\to i\), \(s_j=+1\) for a pyramidal
slot and \(-1\) for an inhibitory slot, and \(\xi\) is dimensionless current-drive
noise. \(I^0_i,S_i\) are a declared operating-current calibration, held fixed
across identity manipulations. Neither this calibration nor \(R\) is learned
from task labels. A fixed 100 Hz normalization is used in every condition; it
does not cancel each cell's distinct fitted maximum rate. Normalized rates can
exceed one.

Initial filter states and initial rates are exactly zero. The explicit fixation
period is part of the task, rather than an unreported steady-state pre-warm.
The current-based network has no additional PV/SST shunting gates: such gates
would introduce unvalidated physiology and additional gradient paths. A GFR
filter is not identified with a measured dendritic compartment.

Labels are \(y_b\in\{-1,+1\}\), and \(m_{bt}\) selects decision bins. Define
\(c_b=\sum_t m_{bt}>0\). Every method uses the same loss:

$$
\mathcal L=\frac{1}{B}\sum_b\frac{1}{c_b}\sum_t m_{bt}
            \log\left(1+e^{-y_bz_{bt}}\right).
$$

The normalization gives every trial equal weight even if its decision duration
differs. It is a mean of **per-time-bin** cross-entropies, not cross-entropy of
the mean logit. The predicted class uses the sign of the decision-weighted mean
logit. A zero logit is assigned to +1 deterministically. Zero-coherence stimuli
must not be supplied as zero labels to a training function. They can be used in
psychometric evaluation with a defined binary target or analyzed as response
probabilities.

The stable loss derivative used by every rule is

$$
\delta_{bt}= -\frac{m_{bt}}{Bc_b}y_b\,
                  \operatorname{sigmoid}(-y_bz_{bt}).
$$

Targets enter only the loss/teaching signal. They never enter neuron dynamics.

## Exact BPTT benchmark

The *exact_bptt_gradients* function differentiates the whole recurrent network.
It keeps rates, activation slopes and logits for the forward trajectory and
propagates adjoints backward through all filter channels, the neuron-intrinsic
beta feedback and every recurrent edge. If \(u_{i,t}^{(k)}=\partial\mathcal L/
\partial h_{i,t}^{(k)}\), its current adjoint is

$$
\frac{\partial\mathcal L}{\partial I_{i,t}}
   =\sum_k u_{i,t}^{(k)}a_{ik}.
$$

The recurrent-magnitude gradient therefore accumulates

$$
\frac{\partial\mathcal L}{\partial w_{ij}}
 =M_{ij}s_jS_i\sum_{b,t}q_{bj,t-1}
                  \sum_k u_{bi,t}^{(k)}a_{ik}.
$$

The derivative passed to the preceding firing rate includes both the
cross-neuron path through \(W\) and the same-neuron path through \(\beta_i\).
The derivative passed to the preceding filter state includes \(1-d_{ik}\).
These paths are retained even when the postsynaptic neuron is inhibitory or
has no direct readout weight. Consequently BPTT can train inhibitory targets
through their indirect influence on subsequent network activity.

“Exact” refers to the analytical derivative of the implemented discrete model,
at differentiable points. The activation includes ReLU; its derivative at the
kink is defined as zero. Central finite differences should not be interpreted
as proof at a nondifferentiable point. No PyTorch installation or surrogate
spike gradient is required.

## Forward-time local eligibility

For every present edge define a local K-vector eligibility

$$
E_{ij,t}^{(k)}\equiv
\left.\frac{\partial h_{i,t}^{(k)}}{\partial w_{ij}}\right|_
{\text{presynaptic recurrent histories clamped}}.
$$

The target cell's own intrinsic beta history remains live under this
derivative. Write \(g'_{i,t}=g'_i(\bar h_{i,t})\) and
\(\bar E_{ij,t}=K^{-1}\sum_k E_{ij,t}^{(k)}\). Starting from zero eligibility:

$$
E_{ij,t}^{(k)}=(1-d_{ik})E_{ij,t-1}^{(k)}
 +a_{ik}S_i s_jq_{j,t-1}
 +\beta_{ik}g'_{i,t-1}\bar E_{ij,t-1}.
$$

The output-rate eligibility is

$$
e_{ij,t}=\frac{g'_{i,t}}{R}\bar E_{ij,t}.
$$

The *local_beta* method accumulates the approximate recurrent update direction

$$
\widehat G_{ij}=\sum_{b,t}\delta_{bt} B_i e_{bij,t}.
$$

\(B_i\) is a frozen random feedback coefficient. It is nonzero in distribution
for **all postsynaptic types**, including inhibitory neurons. This removes the
old difference in plastic support between local learning and BPTT. The optional
symmetric feedback mode sets \(B_i=v_i\) on the readout support; that is a
weight-transport control, not the default biological approximation.

The eligibility recursion is exact for the explicitly clamped local problem.
It does **not** imply an exact full-network gradient: the current learning
signal omits future influence through other neurons. This distinction follows
the eligibility/learning-signal separation motivating e-prop [1], but this
implementation is a GFR-specific rate-model approximation rather than a
reproduction of that spiking-network study.

The implementation advances only forward in time. It retains an array of shape
[batch, present_edges, K], not a full history or a backward adjoint. The memory
cost is \(O(BEK)\), plus neuron state, independently of trial duration; the
Python implementation is aimed at microcircuits, not million-synapse networks.
Parameters remain fixed inside a trial batch, and one accumulated update is
applied afterward. “Online” here describes causal forward-time eligibility and
teaching-signal computation, not an assertion of continuous within-bin weight
changes. The supervised error is available during the response window, and
not during the preceding silent delay.

### Isolating the beta derivative

The *local_no_beta* method removes only the final beta term from the eligibility
equation. The forward model still contains exactly the same beta feedback and
all K filters. The task, weights, loss, feedback, input support and optimizer
are otherwise identical. This answers a narrow question: what does that local
Jacobian term contribute? It does not simultaneously abolish slow intrinsic
memory. The separate fast-decay physiology control addresses a different
question and must be reported separately.

When every beta coefficient is zero, the two eligibility rules are identical;
this is tested bit for bit. The original RFLO paper [2] is useful context for
local approximations and random feedback. Calling this ablation “RFLO” would
overstate the implemented correspondence.

## Scalar-loss-modulated node perturbation

The *node_perturbation_gradients* function provides a distinct, exploratory
three-factor mechanism without any activation derivative in its recurrent
update. Each trial receives independent perturbations of normalized input
drive, \(\xi_{bi,t}\sim\mathcal N(0,\sigma^2)\). Given the observed past, the
conditional drive is Gaussian with mean \((Ux_t)_i+\sum_jW_{ij}q_{j,t-1}\).
Its score with respect to a recurrent magnitude is local:

$$
A_{bij}=\sum_t\frac{\xi_{bi,t}}{\sigma^2}s_jq_{bj,t-1}.
$$

After the trial, its scalar decision loss \(\ell_b\) modulates that accumulated
trace:

$$
\widehat G_{ij}=\frac1B\sum_b (\ell_b-c_b^{\mathrm{base}})A_{bij}.
$$

The default baseline is the loss from a **separate, unperturbed rollout of the
same input**. It depends on the task and fixed parameters, but not the current
noise draw, so its expected product with the score is zero. The zero-baseline
option removes the second rollout and usually increases estimator variance.
The baseline is not updated using the current noisy loss before computing the
advantage. The implementation does not divide by an estimated within-batch
advantage standard deviation, which would change the finite-sample estimator.

There is no extra \(S_i\) factor in this score: both signal and perturbation are
defined before the current scale \(S_i\). If perturbations were instead
specified in pA, the score formula would need to change. The code and metadata
state this convention explicitly.

This likelihood-ratio estimator targets the expectation of loss under noisy
trajectories. At finite \(\sigma\) that is not the deterministic BPTT objective.
Use the optional fixed-noise argument of BPTT/local learning for a matched
noise control. Deterministic evaluation, noise-free BPTT and noisy-training
results must be identified separately. The estimator can have high variance;
a correct derivation does not imply useful learning speed. Reward-modulated
local plasticity in recurrent task models is motivated by work such as Miconi
[3]; this Gaussian score implementation is not claimed to duplicate that
paper's particular biological learning rule.

All scores and losses are accumulated forward without storing a trajectory.
Noise after a trial's final response is excluded from its score, so padding
cannot add avoidable estimator variance.
The clean baseline doubles the number of forward rollouts, and that cost is
reported. Readout learning uses the exact conditional local delta rule on the
noisy trajectory, so this is a loss-modulated recurrent rule with a supervised
readout. A strictly reward-only end-to-end model would require an explicit
policy/action model and policy-gradient readout, which is not claimed here.
Negative loss can be read as reward, but that naming change alone does not
establish a biological reward system.

## Reservoir control, optimization and fair comparisons

The *readout_only_gradients* function freezes every recurrent magnitude and
learns only the readout. It is a necessary control because good performance can
arise from fixed intrinsic memory and random recurrent features. The optimizer
respects the frozen-recurrence marker even if weight decay is nonzero. A
readout-only baseline cannot support a claim that recurrence itself learned
the task.

The *ProjectedOptimizer* class offers SGD and Adam. Recurrent magnitudes are
projected onto the nonnegative interval, optionally bounded above; absent
edges stay exactly zero. Readout updates are restricted to their declared
support. A global gradient-norm clip and weight decay are optional, explicit
settings. Projection fraction and clip factors are returned. Intrinsic GFR
parameters are immutable and fingerprinted; optimizer code never changes them.

Adam includes global engineering machinery and is not offered as biological
evidence. A comparison using the same Adam settings can isolate update
directions operationally, while the SGD sensitivity experiment is needed to
assess dependence on the optimizer. Identical numeric learning rates do not
establish equal tuning: the methods have different gradient scales and
variance. A publication must allocate the same development-validation tuning
budget, freeze the chosen settings before confirmation, and report both
sample and computation costs. Learning rates cannot be selected on final
OOD or confirmation performance.

Optimizer checkpoints contain step counts and all Adam first/second moments.
Resuming an optimizer with only network weights is not an exact continuation.
RNG states, task stream position and construction provenance are the runner's
additional responsibilities. Tests verify identical next-step updates after
restoring optimizer state.

## Validation and what it establishes

The standard-library unittest suite checks independently useful invariants:

1. The learner's forward kernel agrees with the public circuit's rates and
   logits, including exogenous current noise.
2. BPTT agrees with central finite differences on every edge of a small
   multi-filter circuit, with both excitatory and inhibitory sources and
   targets, and on every supported readout parameter and bias.
3. The same check runs on sampled authentic Allen cells with K=8. This checks
   the parameter/units integration, rather than only a synthetic example.
4. Local eligibility matches an independent scalar, clamped-presynaptic
   finite-difference problem. That test intentionally does not claim identity
   with the complete recurrent gradient.
5. The no-beta control changes the derivative term alone; with beta zero, its
   gradients are identical to the full local rule.
6. All deterministic rules agree on objective, accuracy and direct readout
   derivative. Their gradient estimators leave the circuit unchanged.
7. A small exact-gradient step decreases the same independently evaluated
   objective.
8. The node-perturbation score estimator and conditional noisy-path BPTT agree
   in Monte Carlo expectation on a small circuit, within measured standard
   error. This catches sign and noise-variance scaling errors; it is not a
   performance benchmark.
9. Projection, frozen intrinsic coefficients, feedback immutability, optimizer
   restoration, invalid label/mask rejection and perturbation reproducibility
   are checked.
10. A finite-difference audit with no nonzero recurrent sensitivity returns
    “inconclusive.” It cannot manufacture a passing zero relative error by
    skipping the check.

The tests are deterministic apart from explicitly seeded Monte Carlo. They do
not train a publication-scale experiment. Numerical derivative agreement
validates implemented equations; it does not validate a biological mechanism
or the scientific hypothesis.

## Why the old learning setup was replaced

| Legacy issue | Consequence | v7 resolution |
|---|---|---|
| A routine described as online first saved full trajectories, then iterated backward | Memory and biological locality claims did not describe the implemented procedure | Explicit forward-time, per-edge eligibility; BPTT is named separately |
| Local recurrent gradients ignored direct derivatives of plastic PV/SST shunt weights | The advertised local eligibility was not the full fixed-presynaptic derivative for those edges | Common current-based dynamics with a stated local derivative and independent checks |
| “No-beta RFLO” training discarded all K>0 traces, while its audit retained them | Mechanism attribution and diagnostic comparisons used different algorithms | No-beta removes only beta's Jacobian contribution; all K channels retained |
| Frozen random feedback was zero on all inhibitory neurons | Local learning and BPTT had different effective plastic supports | Random feedback reaches every postsynaptic type; support remains identical |
| Finite differences checked only the first eligible pyramidal-to-pyramidal edge | Inhibitory paths and shunts could be wrong while the reported test passed | Multiple signed source/target classes, strongest and random edges, all output parameters |
| If no eligible edge was found, finite-difference errors were set to zero | A missing check looked like perfect agreement | Explicit “inconclusive” status for a flat recurrent test |
| Filter states started at a nonzero steady-state approximation despite a zero-reset claim | Delay-memory comparisons depended on hidden initial conditions | Exact zero state and explicit task fixation |
| Physiology and baseline manipulations differed across ostensibly matched conditions | Algorithm/physiology conclusions were confounded | Frozen intact coefficients for authentic cells; named one-factor controls and fixed calibration |
| No frozen-recurrence learning baseline or scalar reward-like estimator | Task learning could be confused with exploitation of pre-existing features | Readout-only reservoir and exploratory node-perturbation baselines |
| No optimizer state was saved | An apparent resumed experiment could follow a different update sequence | Serializable moments/step counts and a continuation-equivalence test |

## Primary references

1. Bellec G. et al. (2020). *A solution to the learning dilemma for recurrent
   networks of spiking neurons*. Nature Communications 11, 3625.
   https://doi.org/10.1038/s41467-020-17236-y
2. Murray J. M. (2019). *Local online learning in recurrent networks with random
   feedback*. eLife 8:e43299. https://doi.org/10.7554/eLife.43299
3. Miconi T. (2017). *Biologically plausible learning in recurrent neural
   networks reproduces neural dynamics observed during cognitive tasks*.
   eLife 6:e20899. https://doi.org/10.7554/eLife.20899
4. Allen Institute. *GRNN*, pinned source and distributed fitted GFR models.
   https://github.com/AllenInstitute/GRNN/tree/835a535f6831b88339c6a5c17180fdbc504f7448
