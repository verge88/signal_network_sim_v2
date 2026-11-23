#!/usr/bin/env python3
"""Inspect *raw* SentryPeer SIP request evidence in independently captured MURHCAD.

Streaming parser for the three Elasticsearch JSON export batches. Does not
publish the raw payload, IPs, Call-IDs, phone numbers, or auth credentials.
Do not infer TICD recall/FPR from SentryPeer inbound-only requests.
"""
import collections,hashlib,io,json,re,sys,zipfile
from pathlib import Path
import ijson

DATA=Path(sys.argv[1] if len(sys.argv)>1 else "CloudHoneyNetDataset.zip")
OUT=Path("results_stage_a");OUT.mkdir(exist_ok=True)
methods=collections.Counter();methods_payload=collections.Counter();src_sensors=collections.Counter()
payload_shapes=collections.Counter();raw_keys=collections.Counter()
total=0;sip=0;w_message=0;w_callid=0;w_cseq=0;responses=0;empty_payload=0
callids=set();request_line_types=collections.Counter();file_results=[]
# Never log actual raw SIP data.
CALLID=re.compile(r'(?im)^(?:Call-ID|i)\s*:\s*([^\r\n]+)')
CSEQ=re.compile(r'(?im)^CSeq\s*:\s*(\d+)\s+([A-Za-z]+)')
FIRST_LINE=re.compile(r'^([^\r\n]+)')
VALID_METHOD=re.compile(r'^(INVITE|REGISTER|OPTIONS|ACK|BYE|CANCEL|PRACK|SUBSCRIBE|NOTIFY|REFER|MESSAGE|INFO|UPDATE|PUBLISH)\s+',re.I)

with zipfile.ZipFile(DATA) as archive:
    filenames=[n for n in archive.namelist() if n.startswith("RawData/") and n.endswith(".json")]
    for filename in filenames:
        counts=collections.Counter()
        with archive.open(filename) as reader:
            prefix=reader.read(512).lstrip()[:1]
        if prefix==b'[': expr='item'
        elif prefix==b'{': expr=''
        else:raise SystemExit(f"Unknown JSON root in {filename} prefix={prefix!r}")
        with archive.open(filename) as reader:
            for record in ijson.items(reader,expr,multiple_values=True):
                counts['all']+=1;total+=1
                if not isinstance(record,dict):continue
                node=record.get('_source',record)
                if not isinstance(node,dict):continue
                for key in node:raw_keys[key]+=1
                fields=node.get('fields') if isinstance(node.get('fields'),dict) else {}
                typ=str(node.get('type') or node.get('sensor') or node.get('eventid') or '').lower()
                sensor=str(node.get('sensor') or '').lower()
                method=str(node.get('sip_method') or fields.get('sip_method') or node.get('collected_method') or '').strip().upper()
                payload=node.get('sip_message') or fields.get('sip_message')
                if not method and not payload and not any(x in typ for x in ['sentry','sip']):
                    continue
                counts['possible_sip']+=1
                sip+=1
                if sensor:src_sensors[sensor]+=1
                if method:methods[method]+=1
                if isinstance(payload,str) and payload:
                    w_message+=1
                    first=payload[:500].splitlines()[0] if payload.splitlines() else ''
                    payload_shapes[type(payload).__name__]+=1
                    m=VALID_METHOD.match(first)
                    if m:methods_payload[m.group(1).upper()]+=1
                    elif first.startswith("SIP/2.0"):responses+=1
                    else:request_line_types['unparsed_first_line']+=1
                    c=CALLID.search(payload[:20000])
                    if c:
                        w_callid+=1
                        callids.add(hashlib.sha256(c.group(1).strip().encode()).digest())
                    if CSEQ.search(payload[:20000]):w_cseq+=1
                else:
                    empty_payload+=1
        file_results.append({"file":filename,"all_records":counts['all'],"sip_candidates":counts['possible_sip']})
        print("SIP_RAW_BATCH_DONE",json.dumps(file_results[-1]),flush=True)

r={"dataset":"MURHCAD real-world public-internet SentryPeer data",
   "raw_source":"Zenodo 10.5281/zenodo.15716735 (unmodified ZIP)",
   "json_batch_rows":total,"sip_candidates":sip,
   "method_counts":dict(methods),
   "methods_parsed_from_first_line":dict(methods_payload),
   "sensors":dict(src_sensors),
   "sip_message_raw_payloads":w_message,
   "sip_message_missing_payloads":empty_payload,
   "sip_messages_with_call_id":w_callid,
   "sip_messages_with_cseq":w_cseq,
   "unique_hashed_call_ids":len(callids),
   "sip_response_records_in_raw":responses,
   "batch_counts":file_results,
   "raw_schema_fields":dict(raw_keys),
   "full_ticd_executed":False,
   "ticd_recall":None,"ticd_fpr":None,
   "note":"Independent inbound honeypot event stream. Full SIP transactions or node-reported invariants cannot be established unless corresponding response/ACK/telemetry exists."
}
(OUT/"murhcad_sip_raw_evidence.json").write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n")
print("SIP_RAW_EVIDENCE",json.dumps({k:v for k,v in r.items() if k!="raw_schema_fields"},ensure_ascii=False)[:24000])
if sip==0:raise SystemExit("No actual SIP requests in source log")
