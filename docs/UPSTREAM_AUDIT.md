# Upstream GRNN audit and scientific fidelity review

Audit date: 2026-11-07. Scope: read-only audit of AllenInstitute/GRNN and comparison with the supplied `grnn_bio_manifold_project_v6_upgraded.zip`. This is an engineering and scientific design audit, not an empirical claim that the revised network succeeds on any task.

## 1. Source identity and what the upstream project actually supplies

Pinned repository commit: **835a535f6831b88339c6a5c17180fdbc504f7448**, default branch `main`, committed 2026-08-17. Source: <https://github.com/AllenInstitute/GRNN/commit/835a535f6831b88339c6a5c17180fdbc504f7448>.

The source paper has now been published; it is no longer necessary to describe it only as a preprint. The indexed primary record gives Calvin Yeung, Kris Ganjam, Zihan Zhang, Stefan Mihalas and Zhixin Lu, **A dataset of differentiable biologically derived single-neuron models**, *Cell Reports Methods*, 2026-10-14, article 101638, DOI <https://doi.org/10.1016/j.crmeth.2026.101638>, PMID <https://pubmed.ncbi.nlm.nih.gov/42874586/>. Publisher record: <https://www.sciencedirect.com/science/article/pii/S2667237526003395>. The web search returned primary publisher/PubMed metadata and abstract, but direct full-text page opening was blocked. The implementation analysis below comes from the successfully retrieved repository source, not from an inaccessible paper PDF.

GFR is a differentiable **firing-rate** model fitted to single-neuron patch-clamp current-to-rate responses. It has a bank of linear input/history filters and a saturating polynomial nonlinearity. It is not a conductance-based spiking model, does not reproduce voltage traces or spike waveforms, has no morphological dendritic compartments, and does not supply a measured connectome between the fitted cells. Multiple timescales should be called filter or history timescales, not anatomically identified dendritic integration compartments.

The upstream `network.py` supplies generic dense linear input, recurrent and output layers. Biological initialization imports fitted neurons, but does not import biological synaptic connectivity. A new Pyr/PV/SST/VIP circuit is therefore a **synthetic circuit with cell-type motif priors**. It cannot be described as an Allen reconstructed connectome. Likewise, these fitted cells came from separate recordings and need not have belonged to one connected circuit. Source: <https://github.com/AllenInstitute/GRNN/blob/835a535f6831b88339c6a5c17180fdbc504f7448/network.py>.

## 2. Verified provenance of supplied assets

The upstream full `model/gfr_dataset.json` was fetched through its Git blob, not inferred from a matching filename:

- Upstream dataset Git blob SHA: `f453374386a48f2ac55d06a170e9b5a73e4b55a5`.
- Upstream data includes **7,229 model entries** across six `(bin_size, actv_bin_size)` configurations.
- The attachment includes **1,124 entries**, all configuration `(20,20)`.
- Every local entry, every nested fitted parameter and every quality metric matches the corresponding upstream `(20,20)` entry after parsing. Both missing-value representations (`NaN`/`null`) were normalized for comparison. Array ordering also matches.
- Upstream metadata Git blob SHA: `bccb1d926d2f6113fc464a9acd33b345c5dea80d`.
- Upstream metadata has 2,333 specimen rows; supplied metadata contains exactly the matching 1,124 specimen rows and 54 columns (including specimen ID). There are **zero unequal parsed values**, counting paired NaNs as equal.

Original attachment asset SHA-256 checksums:

| Asset | SHA-256 |
|---|---|
| `model/gfr_dataset.json` | `0539cbe9dc4202f33448c76a43ce88839300360c85fa4289c85be3da755a6a0e` |
| `data/metadata.csv` | `ce0a046996b1abc6338c194899405c592bbb01cdfde638cec150446c647997dc` |

A machine-readable record and source-file hashes are in `provenance/upstream/PROVENANCE_VERIFICATION.json`. The matching dataset establishes authenticity of the supplied **parameters**; it does not validate modifications subsequently applied by `src/circuit.py`, nor does it provide an independent replication of patch-clamp performance.

## 3. Exact discrete equation and units

The operative definition is `model.py`, especially `GFR.forward` and `BatchGFR.forward`, not a loose interpretation of the README equation. Source: <https://github.com/AllenInstitute/GRNN/blob/835a535f6831b88339c6a5c17180fdbc504f7448/model.py>.

For neuron i and filter k, the published code computes:

`h[i,k,t] = (1 - ds[i,k]) * h[i,k,t-1] + a[i,k] * I[i,t] + 1000 * b[i,k] * f[i,t-1]`

`z[i,t] = mean_k(h[i,k,t])`

`x[i,t] = (z[i,t] - threshold[i]) / max_current[i]`

`poly[i,t] = sum_d(poly_coeff[i,d]**2 * x[i,t]**d)`

`f[i,t] = max_firing_rate[i] * max(tanh(poly[i,t]), 0)`

The current is in pA and `f` is in spikes/ms. In a Hz-based reimplementation let `r=1000*f` and `Gamma=1000*max_firing_rate`; then the history term is exactly `b*r_previous`, and the same state equation is recovered. This is a change of output units, not a change in dynamics.

**Do not multiply the stored a or b by another timestep.** Although the README equation writes a delta-t factor, the released coefficients are fitted in the discrete implementation above. Adding `dt=20` to those terms breaks equivalence. `bin_size` is metadata and controls the fitted decay basis, data binning and conversion of rate to expected counts during fitting; it is not multiplied inside `forward`.

The input preprocessing computes `I=1e12*stimulus_in_amperes`. The firing-rate target is bin spike count divided by bin duration in milliseconds. The fitting loss compares `f*bin_size_ms` with observed `f*bin_size_ms`, i.e. expected counts with observed counts. Sources: <https://github.com/AllenInstitute/GRNN/blob/835a535f6831b88339c6a5c17180fdbc504f7448/data.py>, <https://github.com/AllenInstitute/GRNN/blob/835a535f6831b88339c6a5c17180fdbc504f7448/train.py>.

For `(20,20)`, timescales are `[instantaneous,20,50,100,200,500,1000,2000]` ms. The instantaneous coefficient is `ds=1`; others are `ds=1-exp(-20/tau_ms)`. Do not use `tau=0` numerically. Source: <https://github.com/AllenInstitute/GRNN/blob/835a535f6831b88339c6a5c17180fdbc504f7448/model_pipeline.py>.

The correct activation applies powers to the normalized variable `(z-threshold)/sigma`; it is not equivalent to computing a polynomial in unnormalized current and dividing the entire result by sigma. That distinction matters for all degrees except certain special cases. The ReLU makes the activation piecewise differentiable; gradient checks should avoid its exact kink.

## 4. Fidelity defects in supplied v6

### 4.1 The condition named “authentic” is not the fitted model

`src/circuit.py` changes the imported arrays before entering the identity manipulation switch:

- Negative slow input coefficients are clipped to zero.
- Instantaneous input coefficients are multiplied by 0.16.
- Pyr, SST and VIP slow coefficients are multiplied by 5.50.
- PV intermediate coefficients are multiplied by 1.20 and longer coefficients set to zero.
- All history coefficients are multiplied by 0.085.

These signed coefficients are learned filter terms. Their signs are not excitatory versus inhibitory synaptic identities and must not be constrained by Dale’s law. Negative filter components can produce adaptation or biphasic history responses; clipping them removes the physiology the project intends to preserve. Dale’s sign belongs on an outgoing synaptic column, separately from a single-cell filter sign.

**Required change:** preserve original a, b, ds, thresholds, activation coefficients and saturation levels verbatim in the primary intact-fitted condition. Any transformed condition must have an explicit name, formula and rationale and cannot inherit the original patch-clamp fit claim.

### 4.2 Additional outcome-relevant changes differ between conditions

The homogeneous condition amplifies mean history coefficients by **7.80**. Tonic baseline changes vary with slow-filter properties, with separate treatment of homogeneous and fast/swap conditions. The “fast gain matched” variant conserves `sum(a)`, which is not the constant-input filter gain `mean(a/ds)` and does not account for the history feedback/nonlinearity. Independent parameter shuffling keeps the original cell-slot calibration even after changing the assigned activation and history filters. Cross-type swap repeatedly copies PV tuples to 80 pyramidal slots rather than permuting all intact tuples bijectively.

These choices change operating point, adaptation magnitude, cell repertoire and input sensitivity simultaneously. An apparent superiority of the “authentic” condition cannot identify an effect of fitted cellular heterogeneity or cell-type alignment.

**Required change:** distinguish exact tuple permutation, independent within-type component permutation, representative/homogeneous control and explicit fast-filter control. Preserve full empirical parameter marginals for permutation controls and report all changes to operating-point distributions. Use one declared calibration rule for all conditions and do not tune it to test accuracy. A control is not automatically fair merely because it has the same number of neurons or shared code branches.

### 4.3 Synthetic motifs are hard-coded to 100 neurons and encode the task

Connectivity, weight initialization and input routing contain literal slices `0:40`, `40:80`, `80:90`, `90:95`, `95:100`. Thus the nominal config is not general. Two sensory streams, two VIP groups and two SST/Pyr gates are explicitly aligned to the two task contexts; this scaffold should be disclosed as architectural prior. Claims that routing emerges fully from learning are too strong. Priors can be useful, but require randomized routing controls or heldout task variants.

The “degree_matched_random” mask preserves each presynaptic out-degree, not postsynaptic in-degree. Column-sum matching with uniform new weights does not preserve the edge-weight distribution. Uniform random sign topology changes Dale compliance as well as topology. Use directed double-edge swaps if the hypothesis needs in-degree and out-degree preservation, and preserve or shuffle existing positive edge magnitudes in the relevant sign classes. Keep motif, degree and sign controls distinct.

### 4.4 Added shunts have no patch-clamp calibration

The supplied learners implement SST and PV divisive gates with manually chosen constants. These gates are not part of the fitted Allen GFR equation. GFR scalar state filters do not identify dendritic versus somatic conductance. If such gates are retained, place them in an optional explicitly phenomenological circuit extension and compare with the unchanged additive-current model. Sharing the same gating code across topologies does not remove the confounding introduced by an unsupported nonlinear mechanism.

### 4.5 Model selection and naming need care

The local loader requires `test_evr>=0.70`, selecting cells on an upstream heldout response metric. Use the released resource's validation/training admission criteria for the primary sample instead, and report heldout EVR diagnostically. Note a further caveat: upstream `data.py` places **Noise 1 in both the training list and validation list**. Released `val_evr` therefore must not be described as a strictly independent heldout score. Noise 2 supplies the named test set. Retaining original release criteria is reasonable for reproducibility, but independent physiological validation requires raw sweeps with a genuinely untouched split.

Cre lines and dendrite labels support broad class assignments. `Pvalb-IRES-Cre` does not establish that every sampled cell is a basket cell; SST does not establish every cell is a Martinotti cell; VIP does not establish layer 1. Original metadata contains cortical layer, so report it rather than replacing it with a label-based assumption. Negative/zero population pools need explicit errors and sampling with replacement must be disclosed.

## 5. What a mathematically defensible biological learning comparison needs

For the exact Hz-based neuron above, with fixed other-neuron inputs, define `gprime_i,t = dr_i,t/dz_i,t`. The cell-local K-by-K state Jacobian is

`J_local[i,t] = diag(1-ds[i]) + outer(b[i], ones(K)) * gprime[i,t-1] / K`.

If `I_i,t = I0_i + scale_i*(...+sum_j(M_ij*s_j*w_ij*rho_j,t-1))`, the direct synaptic drive derivative is `scale_i*M_ij*s_j*rho_j,t-1`. A forward eligibility tensor can therefore use

`E[i,j,:,t] = J_local[i,t] @ E[i,j,:,t-1] + a[i] * scale_i*M_ij*s_j*rho[j,t-1]`.

An output-error signal times the resulting rate eligibility is a local **approximation** to the full-network gradient because it omits paths through other neurons. Even symmetric instantaneous readout feedback does not restore those missing recurrent paths. The temporal eligibility itself is exact for the declared cell-local derivative; that is the appropriately limited exactness claim.

A reverse-time adjoint calculation that is algebraically equal to local forward eligibility does not constitute an online implementation. It can be used as a test oracle, but the actual learning procedure must run forward to justify online temporal credit assignment. A no-history ablation should change only the b-feedback term in the eligibility recursion. Deleting every slow eligibility channel changes two factors and is not a clean adaptation ablation.

The original e-prop work provides the eligibility/learning-signal factorization and analyzes approximations: Bellec et al., 2020, <https://doi.org/10.1038/s41467-020-17236-y>. Its source and numerical factorization tests are at <https://github.com/IGITUGraz/eligibility_propagation>. RFLO is an explicitly approximate random-feedback local online rule: Murray, 2019, <https://doi.org/10.7554/eLife.43299>, <https://elifesciences.org/articles/43299>.

A separate reward-based mechanism can test whether delayed scalar reward suffices. A principled baseline is stochastic current perturbation with a score-function eligibility accumulated over a trial; the reward prediction baseline must be updated after computing the current advantage and must not depend on the sampled perturbation. If perturbation variance, eligibility decay, reward clipping or antithetic rollouts are used, specify their effect on estimator bias and count all rollouts in the compute/sample budget. Do not call such a method an experimentally established synaptic rule. Related biological motivation: Miconi, 2017, <https://doi.org/10.7554/eLife.20899>; Gerstner et al., 2018, <https://doi.org/10.3389/fncir.2018.00053>. Reward-modulated plasticity and random-feedback supervised eligibility are different information regimes; report that distinction rather than collapsing both into one “biological” category.

Required verification should include independent finite differences for the **same loss, forward dynamics and optimizer-facing parameters** that training uses; forward-eligibility versus restricted-autodiff/finite-difference tests; immutable feedback checks; full BPTT versus numerical gradients; and invariants for mask, signs, fixed physiology and deterministic no-noise evaluation. A gradient test on a simplified proxy is insufficient evidence for the production implementation.

## 6. Motif priors: appropriate sources and bounds

Three sources support qualitative class-level priors:

1. Pfeffer et al., 2013, **Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons**, <https://doi.org/10.1038/nn.3446>, primary record <https://www.nature.com/articles/nn.3446>. Supports structured inhibitory interactions, including VIP-to-SST preference, rather than arbitrary homogeneous inhibitory mixing.
2. Pi et al., 2013, **Cortical interneurons that specialize in disinhibitory control**, <https://doi.org/10.1038/nature12676>, primary record <https://www.nature.com/articles/nature12676>. Supports VIP-mediated suppression of other interneurons as a disinhibitory motif.
3. Campagnola et al., 2022, **Local connectivity and synaptic dynamics in mouse and human neocortex**, <https://doi.org/10.1126/science.abj5861>, full text <https://pmc.ncbi.nlm.nih.gov/articles/PMC9970277/>. Supports dependence of connectivity and synaptic dynamics on cell type, layer and species.

These papers do **not** justify attributing the supplied numerical probability matrix or hand-tuned synaptic weights to measurements without extracting the appropriate population/layer/experimental estimates. Default probabilities should be called modeling choices inspired by motifs. Literature calibration, uncertainty sweeps and layer-specific sampling are next evidence steps; merely citing a motif paper is not quantitative connectome validation.

## 7. License, attribution and publication claims

The upstream `LICENSE.txt` is an Allen Institute license based on two-clause BSD with an additional restriction on commercial use without written permission. Preserve the complete notice for derived/distributed code; do not label the entire package MIT or plain BSD. Source: <https://github.com/AllenInstitute/GRNN/blob/835a535f6831b88339c6a5c17180fdbc504f7448/LICENSE.txt>.

The upstream `metadata.json` separately declares CC-BY-4.0 for the dataset, but contains stale anonymous author/URL/placeholder citation fields. Preserve both the software license and dataset attribution; the metadata's CC-BY declaration is not grounds to overwrite the explicit software license. Source: <https://github.com/AllenInstitute/GRNN/blob/835a535f6831b88339c6a5c17180fdbc504f7448/metadata.json>.

The v6 README includes numerical significance claims and strong cellular/circuit interpretations. Those results cannot be carried into a new system after correcting dynamics, task splits and controls. Retain old tables only as historical supplied outputs, never as freshly reproduced evidence. A journal-level submission requires the preregistered comparison to actually run, matched tuning effort, independent replicate units, uncertainty intervals, negative controls, honest negative results and a scientific contribution beyond reusing a released cell database. A well-tested code release can support that work but cannot guarantee either empirical superiority or publication acceptance.
