#!/usr/bin/env python3
"""Stage A audit of independent benign SIP wire traces.

Important: no claimed TICD false-positive rate without independently observed
node reports, zonal information, and a frozen TICD threshold. Captures with no
decodable SIP are kept in denominator and reported as coverage failures.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, zipfile
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.sip_external_pcap_validation import analyze

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def audit_one(path,source,label):
    row={"file":path.name,"source":source,"cohort_label":label,"status":"not_evaluated"}
    try:
        result=analyze(path)
        row.update(result)
        row["status"]="decoded"
    except ValueError as e:
        row["status"]="no_decoded_sip"
        row["error"]=str(e)[:200]
        row["bytes"]=path.stat().st_size
        row["sha256"]=sha256_file(path)
        row["sip_messages"]=0
    except Exception as e:
        row["status"]="parser_error"
        row["error"]=f"{type(e).__name__}: {str(e)[:300]}"
        row["bytes"]=path.stat().st_size
        row["sha256"]=sha256_file(path)
        row["sip_messages"]=None
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dimauro-zip",type=Path,required=True)
    ap.add_argument("--wireshark-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,default=Path("results_stage_a"))
    args=ap.parse_args()
    args.out_dir.mkdir(exist_ok=True,parents=True)
    results=[]
    zip_sha=sha256_file(args.dimauro_zip)
    with zipfile.ZipFile(args.dimauro_zip) as archive:
        files=[v for v in archive.infolist() if v.filename.lower().endswith((".pcap",".pcapng"))
               and v.filename.startswith("LTE_Pcap/") and not v.is_dir()
               and not v.filename.split("/")[-1].startswith("._")]
        if len(files)!=16:
            raise SystemExit(f"Expected 16 main LTE traces, got {len(files)}")
        for info in files:
            # Reject zip bombs and traversal.
            if info.file_size>120*1024*1024:
                raise SystemExit(f"Unexpected oversized member: {info.filename}")
            out=Path("/tmp/sip_dimauro_audit.pcapng" if info.filename.endswith("pcapng")
                     else "/tmp/sip_dimauro_audit.pcap")
            with archive.open(info) as src, out.open("wb") as dst:
                import shutil
                shutil.copyfileobj(src,dst)
            # Keep original descriptive path, not sensitive packet fields.
            row=audit_one(out,"dimauro_lte_a","benign_experiment")
            row["file"]=info.filename
            row["member_bytes"]=info.file_size
            results.append(row)
            out.unlink()
    ws=list(args.wireshark_dir.glob("*.pcap"))+list(args.wireshark_dir.glob("*.pcapng"))
    if len(ws)<5:
        raise SystemExit(f"Insufficient Wireshark reference files: {len(ws)}")
    for path in ws:
        results.append(audit_one(path,"wireshark_sample_captures","benign_reference_unverified"))
    statuses=Counter(r["status"] for r in results)
    cohorts=Counter(r["source"] for r in results)
    output={
      "method":"external SIP transaction inventory, no TICD weights or predictions",
      "dimauro_archive_sha256":zip_sha,
      "external_traffic_synthetic_by_this_project":False,
      "capture_count":len(results),
      "by_source":dict(cohorts),
      "statuses":dict(statuses),
      "sip_messages_decoded":sum(r.get("sip_messages") or 0 for r in results),
      "invites_200_observed":sum(r.get("unique_invite_200_keys",0) for r in results),
      "matched_acks_observed":sum(r.get("matched_200_ack_keys",0) for r in results),
      "no_external_node_reports":True,
      "no_external_zone_telemetry":True,
      "ticd_fpr":None,
      "ticd_auc_roc":None,
      "ticd_recall":None,
      "note":"The observations are actual PCAP protocol records, not full TICD input. Incomplete recording paths may create ACK gaps; no attack detector validity is asserted.",
      "captures":results}
    (args.out_dir/"benign_coverage.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    cols=["file","source","cohort_label","status","bytes","sha256","sip_messages",
         "unique_call_ids","unique_invite_keys","unique_invite_200_keys",
         "unique_ack_keys","matched_200_ack_keys","missing_ack_after_200_keys",
         "physical_ack_closure_ratio","error"]
    with (args.out_dir/"benign_capture_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for row in results:w.writerow({k:row.get(k) for k in cols})
    print("SIP_STAGE_A_BENIGN_SUMMARY",json.dumps({k:v for k,v in output.items() if k!="captures"},ensure_ascii=False),flush=True)
    for r in results:
        print("SIP_CAPTURE",json.dumps({k:v for k,v in r.items() if k in cols},ensure_ascii=False),flush=True)
    if statuses["parser_error"]:
        raise SystemExit("Some PCAP parses failed")
    if sum(r.get("sip_messages") or 0 for r in results)==0:
        raise SystemExit("No independently recorded SIP messages found")
if __name__=="__main__":main()
