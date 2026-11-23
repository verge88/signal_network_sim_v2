#!/usr/bin/env python3
"""Aggregate real-world SIP honeynet events from the MURHCAD Zenodo archive.

These are independently observed events, NOT a benign+attack benchmark nor
SIP transaction PCAP; no TICD sensitivity, false positive rate or AUC.
Sensitive IP addresses and individual payloads are NOT published.
"""
from collections import Counter,defaultdict
import csv, datetime, io, json, hashlib, re, sys, zipfile
from pathlib import Path
ARCHIVE=Path(sys.argv[1] if len(sys.argv)>1 else "CloudHoneyNetDataset.zip")
OUT=Path("results_stage_a");OUT.mkdir(exist_ok=True)
PATH="PreprocessedData/HoneyNetEvents_Clean.csv"
protocol_counts=Counter(); attack_counts=Counter(); sip_attacks=Counter()
sip_ports=Counter(); sip_by_hour=Counter(); sip_by_day=Counter()
sip_per_source=Counter(); sip_per_5m=Counter(); rows=0; sip_rows=0; timestamps_ok=0

def timebin(s):
    if not s:return None
    try:
        s=s.strip().replace("Z","+00:00")
        d=datetime.datetime.fromisoformat(s)
        if d.tzinfo:d=d.astimezone(datetime.timezone.utc)
        return d.replace(minute=d.minute//5*5,second=0,microsecond=0)
    except ValueError:
        return None

with zipfile.ZipFile(ARCHIVE) as archive:
    f=io.TextIOWrapper(archive.open(PATH),encoding="utf-8-sig",errors="replace",newline="")
    for rec in csv.DictReader(f):
        rows+=1
        proto=(rec.get("protocol") or "").strip()
        attack=(rec.get("attackType") or "").strip()
        p=proto.lower()
        protocol_counts[proto]+=1
        attack_counts[attack]+=1
        is_sip=("sip" in p or "sentrypeer" in p or "sentry" in p)
        if not is_sip:continue
        sip_rows+=1
        sip_attacks[attack]+=1
        sip_ports[(rec.get("dstPort") or "").strip()]+=1
        src=(rec.get("srcIp") or "").strip()
        if src:sip_per_source[src]+=1
        t=timebin((rec.get("timestamp") or "").strip())
        if t:
            timestamps_ok+=1
            sip_by_hour[str(t.hour)]+=1
            sip_by_day[t.date().isoformat()]+=1
            sip_per_5m[t.isoformat()]+=1

    raw=[x.filename for x in archive.infolist() if x.filename.startswith("RawData/") and x.filename.endswith(".json")]
    raw_schema={}
    for name in raw:
        # Only inspect keys from the start of each JSON stream, no values.
        with archive.open(name) as f:
            sample=f.read(160000).decode("utf-8",errors="replace")
        keys=sorted(set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]{0,70})"\s*:',sample)))
        raw_schema[name]={"first_160kb_json_keys":keys[:150],
                          "contains_sip_key":any("sip" in x.lower() for x in keys),
                          "contains_method_key":any("method" in x.lower() for x in keys),
                          "contains_message_key":any("message" in x.lower() for x in keys)}

summary={"source":"MURHCAD Zenodo 10.5281/zenodo.15716735",
 "raw_capture":False,
 "genuine_live_internet_honeypot_events":True,
 "per_event_labels":"honeypot or heuristic categorization; not independently verified attack ground truth",
 "total_rows":rows,
 "sip_rows":sip_rows,
 "sip_fraction":sip_rows/rows if rows else None,
 "protocol_counts":dict(protocol_counts),
 "attack_type_counts_all":dict(attack_counts),
 "sip_event_attack_types":dict(sip_attacks),
 "sip_destination_port_counts":dict(sip_ports),
 "sip_event_day_counts":dict(sorted(sip_by_day.items())),
 "sip_event_hour_utc_counts":dict(sorted(sip_by_hour.items(),key=lambda kv:int(kv[0]))),
 "unique_sip_source_addresses":len(sip_per_source),
 "sip_timestamp_parse_success":timestamps_ok,
 "sip_5min_nonempty_windows":len(sip_per_5m),
 "sip_5min_nonempty_window_min":min(sip_per_5m.values()) if sip_per_5m else None,
 "sip_5min_nonempty_window_max":max(sip_per_5m.values()) if sip_per_5m else None,
 "sip_5min_nonempty_window_mean":sum(sip_per_5m.values())/len(sip_per_5m) if sip_per_5m else None,
 "raw_batch_schema":raw_schema,
 "ticd_full_external_evaluation_possible":False,
 "ticd_recall":None,"ticd_false_positive_rate":None,"ticd_auc":None,
 "limits":"No SIP dialog packets, independent slave reports, zone telemetry, or known-good counterfactual labels."}
(OUT/"murhcad_sip_attack_profile.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
print("MURHCAD_SIP_ATTACK_PROFILE",json.dumps({k:v for k,v in summary.items() if k!="raw_batch_schema"},ensure_ascii=False)[:18000],flush=True)
print("MURHCAD_RAW_SCHEMA",json.dumps(raw_schema)[:16000],flush=True)
if sip_rows<=0:raise SystemExit("No SIP-classified events in protocol field: cannot use as SIP evidence")
