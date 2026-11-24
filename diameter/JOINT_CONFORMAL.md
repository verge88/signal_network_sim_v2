# Diameter joint conformal calibration (development)
The frozen local score combines shape and rate/burst evidence. Separate
component cutoffs do not establish a 5% false-positive budget for their union.
Fit the existing local score on reference and inner calibration PCAPs, freeze
it, then compute a conservative rank p-value on a third benign-only set:
p(s) = (1 + count(S_i >= s))/(n+1). Alarm when p <= alpha.
At alpha=.05 at least 19 exchangeable calibration units are needed for a
nontrivial decision; ties are conservative. No attack labels enter fitting.
This gives marginal, not per-scenario or time-uniform, benign FPR control
if the outer calibration and future benign scores are exchangeable given
the fitted score. Correlated captures, drift and stratified sampling can
invalidate that assumption. Capture hashes detect exact duplicates only.
The archived replay publishes a per-capture audit. It excludes undecodable
captures, counts byte-identical copies once within consistent dataset/role/
scenario/label metadata, and aborts on cross-dataset or metadata conflicts.
For the historical replay, each procedure contributes four reference and six
inner calibration captures, preserving the fitter's procedure-specific
profiles. The split prefers original reference captures and then orders by
filename, independently of anomaly scores and evaluation labels. Excess
captures form the outer set, which needs at least 19 usable captures at
alpha=.05. The manifests include both original and assigned roles.
The archived jDiameter replay is development, not a new independent
confirmation; the original frozen detector and its results remain unchanged.
A prospective confirmation must freeze this candidate and collect disjoint
benign calibration and test sessions, including mixed legitimate load and
low-intensity anomaly cases, with enough independent units for useful CIs.
Report both sensitivity and FPR; do not choose alpha from final results.
