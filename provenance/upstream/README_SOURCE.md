# Pinned Allen GRNN source and data verification

Repository: <https://github.com/AllenInstitute/GRNN>

Commit: `835a535f6831b88339c6a5c17180fdbc504f7448` (2026-08-17).

The Python files and README in this directory are unmodified source snapshots,
retained for provenance and equation review. They are **not** imported by the
v7 experiment code and do not define its runtime dependencies. Preserve the
complete Allen Institute notice in `LICENSE.txt` when distributing them.

`PROVENANCE_VERIFICATION.json` records the supplied data checksums, upstream
Git blob identities, exact subset comparison and hashes of these source files.
On 2026-11-07 the full upstream dataset and metadata were retrieved through
GitHub's blob API. All 1,124 supplied `(20,20)` parameter entries were compared
with the corresponding upstream entries, including the nested fitted arrays,
identifiers and metrics. They matched after normalizing missing NaN/null values.
The 1,124 metadata rows matched every parsed value and column of the corresponding
upstream subset; paired missing values counted as equal. There were zero
mismatched values. Full upstream data contain 7,229 model entries across six
configurations and 2,333 metadata specimen rows.

The data files shipped in the main `model/` and `data/` directories are the
unchanged user-supplied subset whose identity was verified. Full upstream
data need not be duplicated here. These comparisons establish parameter
provenance, not an independent reproduction of physiological fits on raw sweeps.

The upstream `metadata.json` declares CC-BY-4.0 for the dataset but also retains
anonymous placeholder citation and URL fields. The explicit software license
adds a restriction on commercial use. Both records are preserved; no blanket
MIT/BSD relicensing is implied. The current published citation supersedes the
placeholder citation:

Yeung C, Ganjam K, Zhang Z, Mihalas S, Lu Z. *A dataset of differentiable
biologically derived single-neuron models*. Cell Reports Methods (2026),
article 101638. <https://doi.org/10.1016/j.crmeth.2026.101638>.

See `docs/UPSTREAM_AUDIT.md` for the exact equation, units, biological scope,
upstream validation-split caveat and the audit of the original v6 project.
