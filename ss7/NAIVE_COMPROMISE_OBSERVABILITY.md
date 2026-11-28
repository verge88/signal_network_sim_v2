# NAIVE compromise: observability boundary and sensor redesign

This note records negative development results so that later SS7 detector work
does not silently retune the same information after inspecting RANK worlds.

## Evaluation rule

The hard case is a compromised node in `NAIVE` mode. It reports its own
traffic honestly, so there is no report-suppression inconsistency to exploit.
A candidate supplement is accepted only if it transfers across independent
simulator worlds and has non-zero worst-world episode recall at a strict false
window budget. TEST worlds are not used during method selection.

The stable detector core remains `hybrid_soph_core_8`: it excludes the
unstable NAIVE channel and, on the already-consumed development holdout
22401/22402, detects 36/48 episodes (75%) in each world while satisfying the
operational budget. Those worlds are development evidence only, not a final
unseen test.

## Negative RANK results

The following NAIVE supplements were tested on independent RANK worlds and
rejected rather than tuned on TEST:

- role-specific regularized linear compromise experts;
- destination / route concentration and grouped route sketches;
- aggregate privacy-preserving subscriber hash sketches (v6);
- rolling subscriber concentration (v6b);
- commissioning-normalized subscriber residuals (v7);
- node-self-normalized residuals (v7.1);
- MAP-count CUSUM and signed service-template CUSUM;
- peer-conditioned subscriber residuals (v8);
- role-stratified v8.1 and node-tail-normalized v8.2 calibration;
- v8 neighborhood oracle (compromised node or any direct signaling neighbor).

Representative strict results:

| Method | Mean / min episode recall | Worst false rows/day | Comment |
|---|---:|---:|---|
| v6 aggregate subscriber KIE | 4.2% / 0% | 29.9 | rejected |
| v6b rolling subscriber KIE | 4.2% / 0% | 142.9 | rejected |
| v7 residual KIE | 0% / 0% | 182.4 | rejected |
| v7.1 self-normalized residual | 0% / 0% | 76.8 | rejected |
| MAP-count CUSUM | 4.2% / 0% | 30.9 | rejected |
| service-template signed CUSUM | 0% / 0% | 125.9 | rejected |
| v8 peer-conditioned subscriber KIE | 0% / 0% | 86.4 | rejected |
| v8.1 role-stratified calibration | 0% / 0% | 182.4 | rejected |
| v8.2 node-tail normalization | 0% / 0% | 34.1 | rejected |
| v8 neighborhood oracle | 0% / 0% | 3.2 | oracle; rejected |

The v8 sensor also increased compact-KIE overhead to roughly 2.1--2.2% maximum
mean link-capacity fraction in the observed RANK worlds, with responses up to
about 5 KB, without producing transferable episode detection. Even an
optimistic neighborhood oracle that accepted an alert at the compromised node
or any directly adjacent signaling node detected 0/24 NAIVE episodes across
the three v8 RANK worlds at about 3.2 false rows/day. Thus the failure is not
explained by source-localization scoring alone.

## Oracle observability audit

An intentionally optimistic diagnostic used the attack labels only for
evaluation and allowed the best single observable feature and same-world normal
threshold to be selected after seeing each RANK world. Even this oracle reached
only 1/8 NAIVE episodes (12.5%) per world at about 3.2 false rows/day. The best
feature was not stable across worlds. A feature chosen consistently across
worlds had worst-world episode recall of 0%.

Therefore the principal limitation is observability, not classifier capacity or
threshold selection. Further ML/hyperparameter tuning on the same aggregated
KIE is not scientifically justified.

## v9 hypothesis

v8 still mixes several MAP services inside broad service groups. A low-rate
targeted campaign can therefore be diluted by unrelated legitimate traffic
sharing the same peer and group.

v9 changes the *sensor semantics*, not the classifier:

```
node -> adjacent peer -> exact MAP service -> keyed subscriber-hash bins
```

The raw subscriber identity is never exported. Only keyed counters are carried
in KIE. The coordinator estimates commissioning-normalized residuals per
peer/service/hash cell. This is a new observable and is evaluated on fresh
RANK seeds 28101--28103. CAL 28201--28202 and TEST 28301--28302 remain frozen
and must not be generated unless the RANK acceptance criterion is met.

If v9 also fails, the current simulation should explicitly report NAIVE source
localization as below the observable detection limit at the chosen KIE budget,
rather than forcing a high recall by retuning thresholds or reusing TEST.
