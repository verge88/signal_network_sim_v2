"""Development study: independent digital-twin calibration and score fusion.

No test labels are used in score fusion or threshold construction.
The labelled mixed references are simulated, NOT live commissioning data.
All eight predeclared development worlds are reported; no confirmatory
claim or selected production policy follows from this screening run.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ss7.threshold_transfer_study import world, score
from ss7.incident_calibration_study import (
    elapsed, triple, bucket_peaks, threshold, metrics,
    RAW_BUDGET, INCIDENT_BUDGET, WARMUP,
)

SEEDS = tuple(range(8901, 8909))
VARIANTS = (
    ("strict_local", "benign"),
    ("report_full", "benign"),
    ("strict_local", "twin"),
    ("report_full", "twin"),
    ("report_full", "twin_margin"),
    ("fusion", "twin"),
    ("fusion", "twin_margin"),
)

def fuse(a, b):
    # Equal-weight log-odds pool of two frozen classifiers; no fitted weights.
    def logit(x):
        x = np.clip(x, 1e-9, 1 - 1e-9)
        return np.log(x) - np.log1p(-x)
    return 1 / (1 + np.exp(-0.5 * (logit(a) + logit(b))))

def reference_cutoff(d, s, z, fraction=1.0):
    # Only negative node-windows and entirely negative consecutive triples
    # in the labelled reference. Never filter rows before forming triples.
    neg = d.label.to_numpy() == 0
    zn = triple(d, np.where(neg, s, -np.inf))
    return max(threshold(s[neg], RAW_BUDGET * fraction, elapsed(d)),
               threshold(bucket_peaks(d, zn), INCIDENT_BUDGET * fraction, elapsed(d)))

def test(seed, localpath, frozenpath, out):
    if seed not in SEEDS:
        raise ValueError("unregistered development seed")
    d, sim, eps = world(seed, 1.5, 32, 64, 10)
    eligible = {}
    for i, e in enumerate(eps):
        if e.kind != "compromise" or e.t0 < WARMUP:
            continue
        q = d[(d.episode_id == i) & (d.nid == e.nid)]
        if len(q) == e.t1-e.t0+1 and q.t.nunique() == len(q):
            eligible[i] = e
    if len(eligible) < 12:
        raise ValueError("insufficient complete episodes; no replacement")
    a = world(seed+10000, 3., 0, 128, 20, sim)[0]
    b = world(seed+20000, 3., 0, 192, 32, sim)[0]
    c = world(seed+40000, 3., 64, 128, 20, sim)[0]
    assert not a.label.any() and not b.label.any() and c.label.any()
    assert all(set(x.nid)==set(d.nid) for x in (a,b,c))
    frozen = joblib.load(frozenpath)
    local = joblib.load(localpath)
    models = {"strict_local": local["strict_local"], "report_full": frozen["report_full"]}
    frames = (a,b,c,d)
    scores = {name: [score(model,x) for x in frames] for name,model in models.items()}
    scores["fusion"] = [fuse(x,y) for x,y in zip(scores["strict_local"],scores["report_full"])]
    rows, events = [], []
    for name, policy in VARIANTS:
        ss = scores[name]
        zz = [triple(x,s) for x,s in zip(frames,ss)]
        refs = (0,1) if policy == "benign" else (0,1,2)
        fraction = .5 if policy == "twin_margin" else 1.
        tau = max(reference_cutoff(frames[i],ss[i],zz[i],fraction) for i in refs)
        m, det = metrics(d,ss[3],zz[3],tau,eligible)
        rows.append(dict(seed=seed,model=name,policy=policy,**m))
        for k in eligible:
            events.append(dict(seed=seed,model=name,policy=policy,episode=k,
                detected=int(k in det),delay_seconds=det.get(k)))
        print(seed,name,policy,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(
        seed=seed, model_runs=[45421952080,45440533210],
        benign_seeds=[seed+10000,seed+20000], twin_seed=seed+40000,
        complete=len(eligible), generated=32, fraction_margin=.5,
        scope="development; simulated labelled twin, no production guarantee"),indent=2))

def aggregate(root,out):
    def collect(name):
        paths=sorted(root.rglob(name+".csv"))
        if len(paths)!=len(SEEDS):
            raise ValueError("incomplete registered development worlds")
        return pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    m,e=collect("metrics"),collect("episodes")
    keys=["seed","model","policy"]
    expected={(s,n,p) for s in SEEDS for n,p in VARIANTS}
    if m.duplicated(keys).any() or set(m[keys].itertuples(index=False,name=None))!=expected:
        raise ValueError("invalid factorial")
    if e.duplicated(keys+["episode"]).any():
        raise ValueError("duplicate episodes")
    for seed in SEEDS:
        g=e[e.seed==seed]
        supports=[set(x.episode) for _,x in g.groupby(["model","policy"])]
        if len(supports)!=len(VARIANTS) or not all(x==supports[0] for x in supports):
            raise ValueError("unpaired episode support")
    m["both_budgets"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    summary=m.groupby(["model","policy"]).agg(worlds=("seed","nunique"),
        episodes=("episodes","sum"), detected=("detected","sum"),
        recall=("recall","mean"), fpr=("fpr","mean"),
        raw_day=("false_windows_day","mean"), worst_raw=("false_windows_day","max"),
        notice_day=("false_notifications_day","mean"), worst_notice=("false_notifications_day","max"),
        both_budgets=("both_budgets","sum")).reset_index()
    for name,table in (("all_metrics",m),("episodes",e),("summary",summary)):
        table.to_csv(out/(name+".csv"),index=False)
    print(summary.to_string(index=False))
    print("Development results only: no policy selected on these worlds.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["test","aggregate"])
    p.add_argument("--seed",type=int)
    p.add_argument("--local",type=Path)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--input",type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="test":
        test(a.seed,a.local,a.frozen,a.output)
    else:
        aggregate(a.input,a.output)
