#!/usr/bin/env python3
"""Inventory of upstream research Drive archives, without guessing labels or downloading whole datasets."""
import json
import os
from pathlib import Path
import gdown

OUT = Path("results_stage_a"); OUT.mkdir(exist_ok=True)
sources = {
 "alvares": "https://drive.google.com/drive/folders/1rGBsHSS3CUQa2ntQkaQfcn-tQggJcm_o",
 "dimauro": "https://drive.google.com/file/d/1-r2Xd1VK6r7O_1KaXVYPus1Rcj6TF9DF/view",
}
for label,url in sources.items():
    result={"dataset":label,"url":url,"status":"blocked","entries":[]}
    try:
        if label=="alvares":
            paths=gdown.download_folder(url=url,skip_download=True,remaining_ok=True,quiet=True,use_cookies=False,timeout=40)
        else:
            x=gdown.download(url=url,skip_download=True,quiet=True,use_cookies=False,timeout=40)
            paths=[x] if x is not None else []
        result["entries"]=[{"id":x.id,"path":x.path} for x in paths] if paths else []
        result["status"]="listed" if paths else "empty"
    except Exception as e:
        result["error"]=f"{type(e).__name__}: {str(e)[:600]}"
    (OUT/f"{label}_inventory.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(f"=== {label}: {result['status']} entries={len(result['entries'])} ===",flush=True)
    if "error" in result: print("ERROR:",result["error"],flush=True)
    for item in result["entries"][:120]:print(item["path"],item["id"],flush=True)
