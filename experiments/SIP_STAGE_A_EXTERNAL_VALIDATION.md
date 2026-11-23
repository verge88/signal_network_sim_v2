# SIP external validation — Stage A (November 23, 2026)

## Scope and scientific status

This is a **partial external protocol audit**, not a full TICD detector validation.
The original TICD model in `sip/ticd.py` requires slave-side report fields
(`cu_ack_closure_mismatch`, `cu_txn_state_consistency_error`), zone deviations,
and cross-node telemetry that are not present in public SIP PCAP alone. The
model is **not fitted, scored, thresholded, or evaluated as if those fields
could be reconstructed**. A PCAP-observed ACK/200 ratio alone is not the TICD
disagreement between claimed and independently observed completed dialogs.

In particular: `TICDetector.score` normalizes using *quantiles of the scored
batch*. Treating each external capture as an independently normalized batch
and comparing numerical cutoffs across captures would be invalid.

## Independent benign traces — validated

* **Mario Di Mauro et al., real LTE-A/Linphone experiments**:
  `https://github.com/mariodim/ml_mobile_dataset`. Exact Google Drive
  archive ID `1-r2Xd1VK6r7O_1KaXVYPus1Rcj6TF9DF`.
  `LTE_Pcap.zip`, 159,057,695 bytes, SHA256
  `e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7`.
  16 captures (8 mobile + 8 fixed; `__MACOSX/._` resource forks excluded).
  These are real captured calls in a controlled LTE-A experiment, *not*
  operator-scale uncontrolled traffic.
* **Wireshark SampleCaptures**, mirrored by `NormB/sipnab` at commit
  `a2ae37cbc3152fab55c1a3cca754c7397148e755`:
  eight distinct SIP capture files from multiple test/scenario sources.
  Original acquisition/benign labels are not independently proven.

Executed GitHub Actions run:
`https://github.com/verge88/signal_network_sim_v2/actions/runs/45227777360`

Results: 24/24 successfully decoded captures; 387 decodable SIP messages;
36 unique 200 OK responses to INVITE matched to ACK; no unmatched 200 OK
in the selected captures. These count **network-observable protocol records**
only. No model-based FPR, AUC, recall, or F1 is inferred.

Important limitations: repeated codec/scenario captures from the same
experiment are correlated. Some captures may not observe both directions,
and even a 200/ACK mismatch in PCAP need not mean node compromise.

## Alvares attack corpus — access inventory attempted

Reference: C. Alvares et al., *Dataset of attacks on a live enterprise VoIP
network for machine learning based intrusion detection and prevention
systems*, Computer Networks 197 (2021) 108283,
doi:10.1016/j.comnet.2021.108283.

Public Drive catalog:
`https://drive.google.com/drive/folders/1rGBsHSS3CUQa2ntQkaQfcn-tQggJcm_o`

The GitHub runner received HTTP 200 and the document title
`VoIP Enterprise - Attack Dataset`, but `gdown.download_folder(skip_download=True)`
returned **zero listed files** and direct embedded-folder view did not
expose downloadable PCAP entries. HTTP success is not evidence that
the PCAPs are actually accessible. This corpus **has not been analyzed**
and no Alvares attack-recall result has been computed.

Reproducible inventory:
* `.github/workflows/sip-stage-a.yml`
* `.github/workflows/sip-stage-a-alvares.yml`
* `experiments/sip_stage_a_inventory.py`
* `experiments/sip_stage_a_alvares_discovery.py`

## External attack controls (not independent attack benchmark)

A separate action analyzes packet records from (1) Honeynet Project VoIP
forensic challenge (constructed scenario), (2) Wireshark Metasploit SIP
INVITE spoof example, and (3) a sipnab test fixture (explicitly excluded
from research validation for unknown provenance). These test the parser
and observable method counters, but **are not treated as samples for
attack recall**, since ground-truth attack-active intervals and verified
compromised node labels are absent.

Workflows:
* `.github/workflows/sip-stage-a-benign.yml`
* `.github/workflows/sip-stage-a-attacks.yml`

Artifacts: `benign_coverage.json`, `benign_capture_metrics.csv`,
`attack_controls.json`, `SHA256SUMS` / trace checksum inventories.
No raw external captures are uploaded as artifacts.

## What remains to complete stage A as originally proposed

1. Obtain accessible copies of the Alvares PCAPs including a manifest
   of clean vs attack scenarios and attacker IP/time boundaries.
2. Freeze the model architecture, calibrate only on independent benign
   training observations, and keep test captures held out by origin.
3. If evaluating full TICD rather than protocol-only indicators, obtain
   independent slave reporting and zone telemetry aligned to PCAP.
4. Report attack-scenario coverage separately from packet-level recall.
   Do not report numerical attack recall/FPR without valid labels.

All reports represent actual machine execution; missing labels and
unobservable features are null, not invented.
