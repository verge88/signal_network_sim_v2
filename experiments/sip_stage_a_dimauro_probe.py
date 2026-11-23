#!/usr/bin/env python3
"""Check availability and ZIP content of independent real LTE SIP traffic.
Do not publish raw trace payloads, IPs, or subscriber identifiers.
"""
import hashlib
import json
import re
from pathlib import Path
import zipfile
import requests
import gdown

OUT=Path("results_stage_a")
OUT.mkdir(exist_ok=True)
data={"source":"Di Mauro et al. LTE-A Linphone dataset","drive_id":"1-r2Xd1VK6r7O_1KaXVYPus1Rcj6TF9DF",
      "status":"unavailable"}
try:
    url="https://drive.google.com/drive/folders/1rGBsHSS3CUQa2ntQkaQfcn-tQggJcm_o"
    r=requests.get(url,timeout=40,headers={"User-Agent":"Mozilla/5.0"},allow_redirects=True)
    embedded=requests.get("https://drive.google.com/embeddedfolderview?id=1rGBsHSS3CUQa2ntQkaQfcn-tQggJcm_o#list",timeout=40)
    x={"source":"Alvares et al. Enterprise VoIP","http_status":r.status_code,"embedded_status":embedded.status_code,
       "html_title":re.findall(r"<title[^>]*>(.*?)</title>",r.text,re.I|re.S)[:1],
       "html_size":len(r.content),"embedded_size":len(embedded.content),
       "file_links":len(re.findall(r"/file/d/",r.text)),"folder_links":len(re.findall(r"/folders/",r.text)),
       "embedded_file_links":len(re.findall(r"/file/d/",embedded.text)),
       "note":"Metadata probe only; zero gdown files is not proof the research dataset has zero contents"}
    (OUT/"alvares_http_probe.json").write_text(json.dumps(x,indent=2)+"\n")
    print("ALVARES_HTTP_PROBE",json.dumps(x),flush=True)
except Exception as e:
    print("ALVARES_HTTP_PROBE_ERROR",repr(e),flush=True)

try:
    target=Path("LTE_Pcap.zip")
    path=gdown.download(id=data["drive_id"],output=str(target),quiet=True,use_cookies=False,timeout=60)
    if not path or not target.exists():
        raise RuntimeError("gdown did not save archive")
    h=hashlib.sha256()
    with target.open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    data["archive_bytes"]=target.stat().st_size
    data["archive_sha256"]=h.hexdigest()
    with zipfile.ZipFile(target) as z:
        info=z.infolist()
        data["total_members"]=len(info)
        data["capture_count"]=sum(i.filename.lower().endswith((".pcap",".pcapng",".cap")) for i in info)
        data["capture_members"]=[{"path":i.filename,"bytes":i.file_size} for i in info if i.filename.lower().endswith((".pcap",".pcapng",".cap"))][:100]
        data["status"]="available"
    print("DIMA URO RESULT",json.dumps(data)[:17000],flush=True)
except Exception as e:
    data["error"]=f"{type(e).__name__}: {str(e)[:900]}"
    print("DIMA URO ERROR",data["error"],flush=True)
(OUT/"dimauro_zip_inventory.json").write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
