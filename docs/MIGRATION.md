# Module-by-module migration from supplied v6 to v7

The full original tree is retained unchanged under `legacy_v6/`; original file digests are in `provenance/original_v6_files.json`. Its historical results are not new v7 results and are not used to populate any v7 metric table.

| Original module | Replacement | Why the change was necessary |
|---|---|---|
| `allen_grnn/model.py` | `grnn_bio/model.py` | Immutable complete fitted tuples; arbitrary-degree/constant activation; explicit pA/Hz/ms contract; independent equation tests against all bundled cells. |
| `allen_grnn/network.py` | `grnn_bio/circuit.py` | The old wrapper was unreachable (`HAS_TORCH=False`) and did not match the upstream constructor. One operational NumPy circuit now owns the actual dynamics. |
| `allen_grnn/utils.py` | `grnn_bio/data.py` | Remove test-EVR admission; conservative operational cell labels; no replacement sampling; explicit specimen cohorts and source mappings. |
| `src/config.py` | `grnn_bio/config.py`, `configs/*.json` | Validated serializable settings; meaningful step counts; full factorial plans; no ignored noise settings; no hidden fixed network size. |
| `src/circuit.py` | `grnn_bio/circuit.py`, `grnn_bio/model.py` | Remove silent coefficient clipping/scaling and unfitted shunting; fixed current calibration; correct directed null graph; consistent physical rate scale; disclose synthetic graph. |
| `src/task.py` | `grnn_bio/task.py` | Balanced rule/conflict design; per-trial delays; go cue and masks; early/late tasks; separate OOD axes and ambiguous zero coherence. |
| `src/learners.py` | `grnn_bio/learners.py` | True forward-time K-channel local traces; exact full BPTT; derivative-only beta ablation; common plastic support; reservoir and scalar-modulated comparison; stronger numerical checks. |
| `src/manifold.py` | `grnn_bio/analysis.py`, `statistics.py`, `dynamics.py` | Independent fitted/evaluated axes; signed held-out d-prime; run-level inference; complete NK stability; no attractor claim from clipped transient contraction. |
| `src/optogenetics.py` | `grnn_bio/interventions.py` | Named cell classes, phase and dose, paired current noise, matched-size controls, per-trial evidence, physical pulse units. |
| `main.py` | `grnn_bio/experiment.py`, `cli.py`, `io.py`, `design.py`, `plotting.py` | Modular workflows; fair paired streams; validation-only tuning; frozen tests; recoverable checkpoints; provenance; executed-only figures; complete factorial contrasts. |
| Historical CSV/PNG output | `legacy_v6/results/` | Preserved for audit; not evidence for the new recurrence, tasks or controls. |
| No automated scientific suite | `tests/test_*.py`, `grnn_bio/validation.py` | Mathematical, statistical, API and resume properties checked with actual results saved by the validation command. |

## Important behavioral changes

Running `python main.py` now requests an explicit command; it no longer launches a large experiment silently. Use `python main.py run --config configs/smoke.json --output results/my_smoke` or the package entry point. The new output schema intentionally does not pretend historical v5/v6 CSV names encode the same experiments.

A checkpoint stores only safe numeric/string arrays and JSON metadata, never executable pickle. Code/config/data mismatches reject resume. Choose a new output directory after changing source, numerical settings or data. The original attachment is retained in this delivery's legacy tree; do not copy its pretrained result claims into a v7 manuscript.

## Scientific scope of the redesign

Preserving single-cell fits constrains physiology but does not identify physiological synaptic weights, axonal delays, cortical area, neuromodulation or a connectome. Fixed-current identity manipulations intentionally include operating-point differences. Report rate/saturation/current distributions and add independently calibrated controls before attributing an effect specifically to within-cell covariance or history timescale. No modification of intrinsic coefficients is hidden behind an “authentic” label.
