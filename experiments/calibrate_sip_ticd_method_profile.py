#!/usr/bin/env python3
"""Fit narrow SIP *benign method-profile* calibration to real LTE-A PCAP.

Only relative INVITE/ACK/BYE method shares are identifiable from the
independent call traces. Transport loss, topology, attack intensity,
per-node volumes and all compromised-node reports remain uncalibrated.
This is a partial, domain-limited calibration, NOT full simulator validation.

Original real capture metadata:
  Di Mauro et al. LTE-A Linphone, Google Drive ID
  1-r2Xd1VK6r7O_1KaXVYPus1Rcj6TF9DF
  SHA256 e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7
"""
from __future__ import annotations
import argparse,hashlib,json,math,shutil,sys,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"sip"))
from experiments.sip_external_pcap_validation import analyze
from sip_sim_v4_claude import SIPNetworkTopology,SimulationConfig,NodeRole,SIP_METHODS

SOURCE_SHA256="e455d66eef30d9399ba3beefc36ad2a4a7e2fac0684aa1ae405a30a2371af9a7"
METHODS=("INVITE","ACK","BYE")

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def aggregate_node_profile(node_specs,weights):
    total=0.0;totals=Counter()
    for node in node_specs:
        d={key:max(1e-8,float(node.method_dist[key]))*weights.get(key,1.) for key in SIP_METHODS}
        z=sum(d.values())
        for key in SIP_METHODS:totals[key]+=node.base_rate*d[key]/z
        total+=node.base_rate
    return {key:totals[key]/total for key in SIP_METHODS}

def calibrate(p:Path,out:Path,confidence_strength=16.0,regularization=25.0):
    if sha256_file(p)!=SOURCE_SHA256:
        raise ValueError("External LTE-A archive checksum mismatch; abort calibration")
    out.mkdir(parents=True,exist_ok=True)
    result=[];counts=Counter()
    with zipfile.ZipFile(p) as archive:
        captures=[e for e in archive.infolist() if
            e.filename.startswith("LTE_Pcap/") and
            not e.filename.split("/")[-1].startswith("._") and
            e.filename.lower().endswith((".pcap",".pcapng"))]
        if len(captures)!=16:raise ValueError(f"Expected 16 main captures, got {len(captures)}")
        for z in captures:
            if z.file_size>120_000_000:raise ValueError("Unexpected decompressed trace size")
            q=Path("/tmp/sip_calibration.pcapng" if z.filename.endswith(".pcapng")
                   else "/tmp/sip_calibration.pcap")
            with archive.open(z) as f,q.open("wb") as o:shutil.copyfileobj(f,o)
            stats=analyze(q);q.unlink()
            if stats["unique_call_ids"]==0:raise ValueError("Zero calls in real capture")
            row={"capture":z.filename,"sha256":stats["sha256"],"sip_messages":stats["sip_messages"],
                 "method_counts":stats["method_counts"],"unique_call_ids":stats["unique_call_ids"],
                 "matched_200_ack_keys":stats["matched_200_ack_keys"],
                 "unique_invite_200_keys":stats["unique_invite_200_keys"]}
            result.append(row);counts.update(stats["method_counts"])
    anchors={m:int(counts[m]) for m in METHODS}
    if min(anchors.values())<3:
        raise ValueError(f"Insufficient anchor methods in real LTE traces: {anchors}")
    denom=sum(anchors.values())
    target=np.array([anchors[m]/denom for m in METHODS],dtype=float)

    topo=SIPNetworkTopology(SimulationConfig(seed=42))
    nodes=[n for n in topo.nodes.values() if n.role==NodeRole.SLAVE]
    base=aggregate_node_profile(nodes,{m:1. for m in METHODS})
    def projected(weight_vec):
        profile=aggregate_node_profile(nodes,{m:float(v) for m,v in zip(METHODS,weight_vec)})
        q=np.array([profile[m] for m in METHODS])
        return q/q.sum()
    target_before=projected(np.ones(len(METHODS)))
    def objective(x):
        pp=projected(np.exp(x))
        distance=np.sum((pp-target)**2/np.maximum(target,0.05))
        return confidence_strength*distance+regularization*np.sum(x*x)
    res=minimize(objective,np.zeros(len(METHODS)),bounds=[(-np.log(2),np.log(2))]*len(METHODS),
                 method="L-BFGS-B",options={"maxiter":400,"ftol":1e-12})
    if not res.success:raise RuntimeError(f"Calibration optimizer: {res.message}")
    weights={m:float(v) for m,v in zip(METHODS,np.exp(res.x))}
    target_after=projected(np.exp(res.x))
    report={
      "calibration_kind":"partial_benign_SIP_method_profile",
      "source":"Di Mauro et al. independently captured real LTE-A Linphone calls (controlled field experiment)",
      "source_url":"https://drive.google.com/file/d/1-r2Xd1VK6r7O_1KaXVYPus1Rcj6TF9DF/view",
      "source_archive_sha256":SOURCE_SHA256,
      "capture_count":len(captures),
      "independence_caveat":"16 PCAPs are codec/location repeats from one experiment, not 16 independent operators",
      "captured_sip_messages":sum(x["sip_messages"] for x in result),
      "request_method_counts":dict(counts),
      "anchor_methods":list(METHODS),
      "anchor_method_counts":anchors,
      "anchor_conditional_target":{m:float(v) for m,v in zip(METHODS,target)},
      "original_simulated_conditional":{m:float(v) for m,v in zip(METHODS,target_before)},
      "calibrated_simulated_conditional":{m:float(v) for m,v in zip(METHODS,target_after)},
      "method_weights":weights,
      "method_weights_for_other_methods":1.0,
      "calibration_effect_l1_before":float(np.sum(np.abs(target_before-target))),
      "calibration_effect_l1_after":float(np.sum(np.abs(target_after-target))),
      "optimiser":{"method":"bounded_regularized_relative_profile_fit",
                   "independent_dialog_effective_sample_size":confidence_strength,
                   "regularization_l2":regularization,
                   "weight_bounds":[0.5,2.0],
                   "optimizer_success":bool(res.success)},
      "not_calibrated":["per_node_traffic_intensity","topology","network_loss","RTT",
                        "intrusion_prior","compromise_intensity","CU_telemetry","zonal_invariants"],
      "out_of_sample_SIP_TICD_validation":False,
      "captures":result,
    }
    if report["calibration_effect_l1_after"]>report["calibration_effect_l1_before"]+1e-9:
        raise RuntimeError("Calibration worsened the chosen observable metric")
    (out/"sip_real_method_calibration.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("SIP_REAL_PROFILE_CALIBRATION",json.dumps({k:v for k,v in report.items() if k!="captures"},
          ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lte-zip",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,default=Path("results_ticd_calibrated"))
    a=p.parse_args()
    calibrate(a.lte_zip,a.out_dir)

if __name__=="__main__":main()
