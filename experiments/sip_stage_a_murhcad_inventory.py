#!/usr/bin/env python3
"""Inspect independently collected real-internet SentryPeer events (Zenodo 15716735).

Outputs only aggregated schema and file hashes. Never upload raw IPs or SIP
identifiers. The data are honeypot events, *not* SIP PCAP or TICD node reports.
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

DATASET = Path(sys.argv[1] if len(sys.argv)>1 else "CloudHoneyNetDataset.zip")
OUT = Path("results_stage_a")
OUT.mkdir(parents=True,exist_ok=True)
h1=hashlib.md5()
h2=hashlib.sha256()
with DATASET.open("rb") as f:
    for chunk in iter(lambda:f.read(1024*1024),b""):
        h1.update(chunk);h2.update(chunk)
if h1.hexdigest() != "9902cf514bcc9cf6983a28bf55c7526c":
    raise SystemExit("Archive MD5 differs from Zenodo record: abort")
data={"source":"Zenodo 15716735 MURHCAD (actual public-internet honeypot events)",
      "source_url":"https://zenodo.org/records/15716735",
      "md5":h1.hexdigest(),"sha256":h2.hexdigest(),
      "archive_bytes":DATASET.stat().st_size,"entries":[]}
with ZipFile(DATASET) as z:
    for item in z.infolist():
        if item.is_dir():continue
        entry={"name":item.filename,"uncompressed_bytes":item.file_size}
        if item.filename.lower().endswith(".csv") and item.file_size <= 400*1024*1024:
            with z.open(item) as f:
                reader=csv.reader(io.TextIOWrapper(f,encoding="utf-8-sig",errors="replace"))
                try:header=next(reader)
                except StopIteration:header=[]
            entry["header"]=header[:100]
        data["entries"].append(entry)
(OUT/"murhcad_schema.json").write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
print("MURHCAD_DATASET_INVENTORY",json.dumps(data,ensure_ascii=False)[:30000],flush=True)
