# Full-state, fixed-input dynamics diagnostics

`grnn_bio.dynamics` analyzes the deterministic circuit under a specified
constant input. A network of N GFR neurons with K history filters has an
N-times-K-dimensional hidden state. Firing-rate trajectories alone are not
a complete Markov state for this model.

`constant_input_step(circuit, h, input_vector)` evaluates the autonomous map
with prior rates consistent with `h`, namely `r = g(mean(h))`. This is the map
used after the first ordinary step. The simulator's deliberate initial rates
of zero are preserved even if a polynomial intercept makes `g(0)` positive;
the diagnostic does not redefine that initialization. No rate clamps or noise
are part of this autonomous diagnostic.

`fixed_point_residual(...)` returns `F(h; x) - h`. A candidate only passes the
stationarity check if its infinity-norm residual is below `atol + rtol *
max(1, norm_inf(h))`. The defaults are numerical tolerances, not biological
measurement precision.

`full_state_jacobian(...)` includes both the intrinsic firing-history path
and all inter-neuron recurrent paths. In the flattening order `(cell, filter)`,
the derivative entry for target cell/filter `(i,k)` and source `(j,l)` is:

```text
delta_ij delta_kl (1-ds_ik)
  + a_ik current_scale_i W_ij gprime_j / (K reference_rate_j)
  + delta_ij beta_ik gprime_i / K
```

Here `gprime` is measured in Hz/pA, `W` already includes the fixed structural
mask and Dale signs, and `reference_rate` is the circuit's fixed physical
rate scale. There is no additional timestep factor. The complete Jacobian
is verified against central finite differences of the autonomous map in the
tests, including nonzero off-diagonal cell blocks.

`fixed_point_diagnostics(...)` reports the residual, spectral radius,
distance to a ReLU activation kink and dimension. The field
`linearly_stable_fixed_point` is left `None` for a transient state or a
nondifferentiable candidate. A small residual and a full-state spectral
radius below one support a local fixed-input linear stability interpretation.
A radius close to one needs numerical tolerance sensitivity. None of these
quantities establish global attraction, a large basin of attraction, or a
task-relevant memory attractor. Contraction between two finite trajectories
is not substituted for a fixed-point test.

The dense Jacobian defaults to a maximum dimension of 1,024 to avoid accidental
large allocations. Computation and eigendecomposition scale with full state
dimension, so use this diagnostic selectively. Raising the guard is explicit.

## Searching for a candidate

`find_fixed_point(circuit, initial_state, input_vector=None)` calls
`scipy.optimize.root` with the `hybr` method on the full-state residual. It
supplies the analytic residual Jacobian **J minus the identity**, rather than
using a rate-only or cell-local approximation. The default external input is
the five-channel zero vector; it is explicitly recorded in the result. The
default numerical budget is 2,000 residual evaluations, with solver `xtol`
equal to 1e-9, and the dense state-size guard also applies before solving.

The output records solver status and message, actual residual/Jacobian callback
counts, the independently recomputed stationarity residual, the full-state
spectral diagnostic, and the candidate state as JSON-compatible nested lists.
`success` requires both solver success and a residual passing the separate
stationarity criterion. A misleading solver termination flag therefore cannot
turn a transient point into a successful fixed-point result. Numerical failure
is reported explicitly. A nonfinite terminal iterate is rejected; any retained
finite candidate is identified as diagnostic fallback, not a successful solve.

The experiment's optional `run_fixed_points` flag invokes this search from one
recorded task state and saves `fixed_point.json`. One starting point can at most
find one candidate. It does not test uniqueness, basins, multiple memories, or
the persistence of task information after the external input is removed. The
presence of this analysis utility does not itself claim that successful
fixed-point results were obtained in any supplied training run.
