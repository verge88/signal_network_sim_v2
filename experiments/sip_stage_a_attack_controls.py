#!/usr/bin/env python3
"""External attack-scenario SIP trace inventory (not a TICD classifier).

PCAP acquisition is distinct from synthetic data generation by this project,
but controlled challenge/fixture traces cannot substitute for unlabelled
real-world attack campaigns. There is no per-packet ground-truth here.
"""
import argparse, csv, json, subprocess, tempfile, shutil
from pathlib import Path
from collections import Counter,defaultdict
from experiments.sip_external_pcap_validation import analyze

def method_rates(pcap):
    with tempfile.TemporaryDirectory(prefix="stage-a-attack-",dir="/tmp") as td:
        p=Path(td)/pcap.name;shutil.copyfile(pcap,p)
        cmd=["tshark","-r",str(p),"-Y","sip","-T","fields","-E","separator=/t",
             "-E","occurrence=f","-e","frame.time_epoch","-e","sip.Method",
             "-e","sip.Status-Code","-e","sip.CSeq.method"]
        x=subprocess.run(cmd,text=True,capture_output=True,check=True)
    times=[]; methods=Counter(); statuses=Counter(); bins=defaultdict(Counter)
    for line in x.stdout.splitlines():
        a=line.split("\t");a+=[""]*(4-len(a))
        try:t=float(a[0])
        except ValueError:continue
        times.append(t)
        m=a[1].split(",")[0].strip().upper()
        if m:methods[m]+=1
        s=a[2].split(",")[0].strip()
        if s:statuses[s]+=1
        bins[int(t//10)][m or "RESPONSE"]+=1
    span=max(times)-min(times) if times else None
    return {"sip_messages_time_parsed":len(times),"capture_time_span_seconds":span,
            "request_methods":dict(methods),"status_codes":dict(statuses),
            "peak_10s_invite":max((v["INVITE"] for v in bins.values()),default=0),
            "peak_10s_register":max((v["REGISTER"] for v in bins.values()),default=0),
            "peak_10s_options":max((v["OPTIONS"] for v in bins.values()),default=0)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,default=Path("results_stage_a"))
    a=ap.parse_args();a.out_dir.mkdir(exist_ok=True,parents=True)
    captures=[
      ("Forensic_challenge_4.pcap","honeynet_forensic_challenge","controlled_forensic_exercise",
       "PCAP created by honeynet members to demonstrate attack, NOT a passive wild-attack capture"),
      ("metasploit-sip-invite-spoof.pcap","wireshark_sample_captures","controlled_attack_tool",
       "Metasploit 3.0 SIP INVITE spoof example, no independent event labels"),
      ("sip-scanner-and-register-flood.pcap","sipnab_test_fixture","test_fixture_excluded",
       "Test fixture of unknown production provenance, excluded from real external validation")]
    result=[]
    for fn,source,role,disclaimer in captures:
        p=a.input_dir/fn
        if not p.exists():raise SystemExit(f"Missing {fn}")
        try:
            r=analyze(p)
            r.update(method_rates(p))
            r.update({"source":source,"label_quality":role,"provenance_note":disclaimer,"status":"decoded"})
        except Exception as e:
            r={"file":fn,"source":source,"label_quality":role,"status":"error",
               "error":f"{type(e).__name__}: {str(e)[:350]}"}
        result.append(r)
        print("ATTACK_CAPTURE_RESULT",json.dumps({k:v for k,v in r.items()
                 if k not in ("method_counts","response_counts","request_methods","status_codes")},
                 ensure_ascii=False),flush=True)
    summary={
      "status":"external_attack_controls_not_full_detector_validation",
      "captures":result,
      "external_observed_sip_packets":sum(r.get("sip_messages",0) for r in result),
      "number_of_controlled_scenario_captures":2,
      "excluded_unverified_fixture_captures":1,
      "ticd_attack_recall":None,"ticd_fpr":None,"ticd_f1":None,"ticd_auc":None,
      "limit":"No verified attack-active timestamps or node telemetry. Scenario capture is not packet-level ground truth.",
      "alvares_status":"Not yet accessible in public gdown folder inventory as of this run"}
    (a.out_dir/"attack_controls.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    if any(x["status"]!="decoded" for x in result):
        raise SystemExit("Some SIP attack controls could not be decoded")
if __name__=="__main__":main()
