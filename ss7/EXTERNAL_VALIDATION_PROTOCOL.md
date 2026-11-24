# Confirmatory experiment and external data acceptance protocol

## Scope

The canonical SS7 generator labels **node compromise only**. Its generic
`ext_attack` is a negative control for compromise detection. It is not a
validated positive example of any particular external attack. Do not relabel
it post hoc. Neither the Telecom Italia activity CDR nor a JSS7-generated
trace is a real SS7 node compromise trace.

The independent confirmation workflow uses development seeds 8101–8103
and twelve held-out worlds 8201–8212. It fits all classifiers before test
generation. A later chronological test period is separated from the earlier
local benign threshold-calibration period. The primary planned contrast
is HGB full versus HGB OWN with an equal validation alert budget
(20 false-positive windows/network-day under all-benign exposure).
Each held-out world must have at least eight *complete nonoverlapping*
positive episodes. At least 96 complete test episodes are required for
the aggregate to exist. Actual counts and failed worlds must be disclosed.
Bootstrap intervals resample worlds, not windows. Paired sign-flip tests
use world-level recall differences, with Holm correction over the listed
contrasts. An improvement is not established until the measured difference,
uncertainty and actual false-alert rates support it.

## Observability and validity audit

- `fix.py` originally included the evaluated node in its zonal median;
  the confirmation branch implements a true leave-one-out peer reference.
- Link reconciliation and TCAP closure consume `lstate` loss estimates
  computed in simulation. In deployment these estimates must come from
  independent, synchronized operational telemetry, not true hidden state.
  The experiment evaluates both the full model and the model without all
  loss-dependent coordinates and derivatives.
- Transit residual at endpoint nodes combines message counts with
  originated/terminated transaction counts. It requires a unit-consistency
  audit using counters expressed in the same unit before operational use.
  A conservative report-only variant omits transit as well as loss-dependent
  coordinates. OD aggregates still require independently observed reports.
- CDR calibration previously trained a detector on observable activity proxies;
  it does not establish a calibrated joint distribution of SS7 invariants or
  a calibrated simulator in the full statistical sense.

## Evidence required from an operator or controlled network

Request synchronized, anonymized data, under a documented agreement:

1. Pseudonymous stable node, peer, link and observation-point identifiers;
   network role, topology, link capacity and routing/linkset changes.
2. Direction-specific transmitted/received signalling-message counts at
   BOTH independently monitored ends of each measured link; timestamps,
   counter resets, missingness, clock offsets and independently measured
   loss/retransmission counts.
3. Independently measured source/destination transaction counts, TCAP
   begin/end/abort/timeout counters, ISUP/MAP operation-class counts.
   Keep message units and transaction units distinct.
4. Actual control reports from each evaluated node and independent reference
   observations from its peers; record whether reports can be forged.
5. Benign incidents (maintenance, traffic changes, failures, rerouting)
   and independently confirmed security incidents, with start/end,
   type, affected nodes and confidence. Unknown windows are not automatically
   benign. No subscriber phone numbers, IMSI or message contents are needed.

Separate training, threshold-calibration and held-out test periods before
looking at scores. For a frozen detector use only confirmed benign
calibration windows, and independently labeled test windows. No overlap
of incident IDs or physical observation windows. Keep an audit of
timestamp alignment and transformations. Use secure exchange; do not
commit confidential telemetry to a public repository.

## Five-class evaluation: independent outcome labels

The proposed study has two distinct endpoints, not one ambiguous label:

- Primary: compromise of a trusted node, compared with benign behaviour
  and other external events.
- Secondary: detection of any of five security incident categories, reported
  separately for compromise, SMS interception, location tracking, signalling
  denial of service, and international revenue share fraud.

For each category use independently confirmed outcomes or controlled,
protocol-level scenarios with separate ground truth. Volume increase
alone does not prove interception, location disclosure or fraud.
Use a network/affected-node scope defined before scoring, match events
one-to-one, report the full denominator and missed incidents. Record window
FPR and incident-level false alarms on benign controls; evaluate overlaps
and external attack classes separately. If any category lacks labels,
report "not evaluated", never zero or 100 percent recall.
A synthetic protocol benchmark remains synthetic even if stored as PCAP.

## Independent sources

- Guo et al. (2023), DOI 10.3390/info14090509: operator traffic from
  POST Luxembourg is restricted; Figshare 10.6084/m9.figshare.23666397.v1
  contains simulated data. It is not an operational multi-node dataset.
- Macedo and Campista (2023), "Attacks to mobile networks using SS7
  vulnerabilities: a real traffic analysis": contact authors/operator for
  raw anonymized telemetry; published aggregate findings do not expose
  paired node reports.
- Telecom Italia Milan CDR: DOI 10.7910/DVN/EGZHFV, normal-activity proxy
  only.
- Wireshark SampleCaptures: short heterogeneous protocol examples useful
  for parsing verification, not for statistical attack recall or paired
  operator invariants.
- ITU-T Q.3066 (01/2026): relevant detection/mitigation framework,
  not a validation dataset.

No claim of real attack recall, real paired-invariant validation,
five-class detection or statistical significance may be made without
corresponding evidence.
