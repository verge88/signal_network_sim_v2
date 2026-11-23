#!/usr/bin/env python3
"""Investigate public Drive folder visibility, without logging IP addresses or PCAP contents."""
import json,re,requests
from pathlib import Path
out=Path("results_stage_a");out.mkdir(exist_ok=True)
folder_id="1rGBsHSS3CUQa2ntQkaQfcn-tQggJcm_o"
r=requests.get(f"https://drive.google.com/drive/folders/{folder_id}",timeout=45,headers={"User-Agent":"Mozilla/5.0"},allow_redirects=True)
s=r.text
patterns={
 "named_attack_terms":{w:s.lower().count(w.lower()) for w in ["INVITE","REGISTER","SPIT","BYE","Asterisk","attack","VoIP Enterprise"]},
 "unique_folder_links":list(dict.fromkeys(re.findall(r"(?:https?:)?(?:\\/\\/|//)drive(?:\\.|\.)google(?:\\.|\.)com(?:\\/|/)drive(?:\\/|/)folders(?:\\/|/)([a-zA-Z0-9_-]{20,})",s)))[:35],
 "unique_file_links":list(dict.fromkeys(re.findall(r"/file/d/([a-zA-Z0-9_-]{20,})",s)))[:35],
 "mime_folder_hits":s.count("application/vnd.google-apps.folder"),
 "root_folder_id_hits":s.count(folder_id),
 "possible_files":s.count(".pcap"),
 "google_drive_folder_listening":s.count("AF_initDataCallback"),
 "response_status":r.status_code,
 "title":re.findall(r"<title[^>]*>([^<]+)</title>",s)[:1]
}
for kw in ["Invite Flood","REGISTER","SPIT","Asterisk",".pcap","application/vnd.google-apps.folder"]:
    k=s.lower().find(kw.lower())
    if k>=0:
        patterns.setdefault("snippets",{})[kw]=s[max(0,k-100):k+160].replace("\\n"," ").replace("\\r"," ")
(out/"alvares_detailed_probe.json").write_text(json.dumps(patterns,indent=2)+"\n")
print("ALVARES_METADATA",json.dumps(patterns)[:16000])
