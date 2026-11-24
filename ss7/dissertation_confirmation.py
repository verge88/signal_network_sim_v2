"""Confirmatory SS7 compromise study: frozen model weights, independent worlds.

This is an experiment on the canonical simulator, NOT real SS7 attack validation.
Primary contrast: HGB full vs OWN, at the same validation-only 20 alerts/day
budget. Independent unit for inference is a regenerated world, not a window.
Every invariant family and every registered INV coordinate (+ derivatives) is
ablated; no-loss and report-only versions audit model-state dependencies.
"""
import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7.fix import OWN, INV, SimConfig, build_dataset, rule_baseline, INTERVALS_PER_DAY
from ss7.ss7_training_v7 import assert_no_leakage, resolve_feature_columns

DEV = (8101, 8102, 8103)
EVAL = tuple(range(8201, 8213))
BUDGET = 20.0
GROUPS = {"link": [x for x in INV if x.startswith("recon_")],
          "od": [x for x in INV if x.startswith("od_")],
          "closure": ["closure_resid"], "transit": ["transit_resid"]}
LOSS = GROUPS["link"] + GROUPS["closure"]
MIN_EP = 8
K = 3


def variants(frame):
    full = resolve_feature_columns(frame, OWN + INV)
    def drop(bases):
        bad = set(resolve_feature_columns(frame, bases))
        return [x for x in full if x not in bad]
    out = {"full": full,
           "own": resolve_feature_columns(frame, OWN),
           "inv": resolve_feature_columns(frame, INV),
           "raw": OWN + INV,
           "no_temporal": [c for c in full if not c.endswith(("__mean","__std","__z"))],
           "no_zone": [c for c in full if not c.endswith("__dev_zone")],
           "no_loss": drop(LOSS),
           "report_only_conservative": drop(LOSS + GROUPS["transit"])}
    for name, group in GROUPS.items():
        out["drop_group_" + name] = drop(group)
    for x in INV:
        out["drop_" + x] = drop([x])
    for cols in out.values():
        assert_no_leakage(cols)
    return out


def world(seed, days, n):
    cfg = SimConfig(seed=seed, days=days)
    df, sim, episodes = build_dataset(cfg, episodes_kw={
        "n_compromise":n, "n_normal":2*n, "n_ext":max(8,n//3)})
    return df, cfg, episodes


def fit(out):
    parts = []
    for seed in DEV:
        df, cfg, eps = world(seed, 1.5, 30)
        y = df["label"].to_numpy()
        neg = np.flatnonzero(y == 0)
        keep = np.r_[np.flatnonzero(y == 1),
                     np.random.default_rng(seed).choice(neg, min(30000,len(neg)), replace=False)]
        parts.append(df.iloc[np.sort(keep)])
        print("development", seed, len(keep), flush=True)
    tr = pd.concat(parts, ignore_index=True)
    features = variants(tr)
    models = {}
    y = tr["label"].to_numpy(int)
    w = np.where(y == 1, (y == 0).sum()/max((y == 1).sum(),1), 1.)
    for name, cols in features.items():
        imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(tr[cols])
        X = imp.transform(tr[cols])
        hgb = HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,
              min_samples_leaf=20,l2_regularization=1.,early_stopping=False,
              random_state=8100).fit(X,y,sample_weight=w)
        models["hgb/"+name] = (cols,imp,hgb)
        if name in ("full","own","inv","report_only_conservative"):
            rf = RandomForestClassifier(n_estimators=400,min_samples_leaf=4,
                  class_weight="balanced_subsample",random_state=8100,n_jobs=2).fit(X,y)
            models["rf/"+name] = (cols,imp,rf)
        print("fitted", name, len(cols), flush=True)
    joblib.dump(models,out/"models.joblib",compress=3)
    (out/"training_manifest.json").write_text(json.dumps({
        "development_seeds":DEV,"rows":len(tr),"positive_rows":int(y.sum()),
        "models":{k:v[0] for k,v in models.items()},
        "training_negative_cap_per_world":30000},indent=2))


def threshold(s, alpha):
    # Conservative empirical threshold; ties do not count as exceedances.
    return float(np.quantile(s,1-alpha,method="higher"))


def test_partition(df, cfg, episodes):
    # Calibration first, later contiguous test. Exclude every partially observed
    # or overlapping positive episode rather than count a fragment as a hit.
    cut = cfg.n_intervals//2
    gap = cfg.feat_window + cfg.z_window
    start = cut + gap
    va = df[df.t < cut-gap].copy()
    te = df[df.t >= start].copy()
    eligible = {}
    for i,e in enumerate(episodes):
        if e.kind != "compromise" or e.t0 < start:
            continue
        if any(j != i and o.kind == "compromise" and o.nid == e.nid
               and o.t0 <= e.t1 and e.t0 <= o.t1
               for j,o in enumerate(episodes)):
            continue
        rows = te[(te.episode_id == i) & (te.nid == e.nid)]
        if len(rows)==e.t1-e.t0+1 and rows.t.nunique()==len(rows):
            eligible[i]=e
    te = te[(te.label==0) | te.episode_id.isin(eligible)].copy()
    if len(eligible)<MIN_EP:
        raise RuntimeError(f"Only {len(eligible)} complete test episodes; need {MIN_EP}")
    return va,te,eligible,start


def evaluate(seed, out, modelpath):
    if seed in DEV or seed not in EVAL:
        raise ValueError("Test seed outside preregistered independent set")
    models=joblib.load(modelpath)
    df,cfg,episodes=world(seed,2.,48)
    va,te,eligible,start=test_partition(df,cfg,episodes)
    alpha=min(.01,BUDGET/(va.nid.nunique()*INTERVALS_PER_DAY))
    yva=va.label.to_numpy(int)
    y=te.label.to_numpy(int)
    metrics,events=[],[]
    specs={"rule/full":None,**models}
    for name,spec in specs.items():
        if spec is None:
            sv,st=rule_baseline(va),rule_baseline(te)
        else:
            cols,imp,model=spec
            sv=model.predict_proba(imp.transform(va[cols]))[:,1]
            st=model.predict_proba(imp.transform(te[cols]))[:,1]
        thr=threshold(sv[yva==0],alpha)
        fire=st>thr
        hits=[]
        for ep,e in eligible.items():
            ix=np.flatnonzero(te.episode_id.to_numpy()==ep)
            order=ix[np.argsort(te.t.to_numpy()[ix])]
            times=te.t.to_numpy()[order]
            n=0; hit=None
            for j,k in enumerate(order):
                n=n+1 if fire[k] and (j==0 or times[j]==times[j-1]+1) else int(fire[k])
                if n>=K:
                    hit=int(times[j]-e.t0)
                    break
            hits.append(hit)
            events.append({"seed":seed,"model":name,"episode":ep,"detected":int(hit is not None),
                          "ttd_seconds":None if hit is None else hit*cfg.interval_s,
                          "family":e.family,"hidden_rate":e.hidden_rate})
        detected=[x for x in hits if x is not None]
        neg=y==0
        duration=(cfg.n_intervals-start)/INTERVALS_PER_DAY
        metrics.append({"seed":seed,"model":name,"episodes":len(hits),
            "detected":len(detected),"episode_recall":len(detected)/len(hits),
            "median_ttd_seconds":float(np.median(detected)*cfg.interval_s) if detected else np.nan,
            "auc":roc_auc_score(y,st),"ap":average_precision_score(y,st),
            "test_fpr":float(fire[neg].mean()),"validation_fpr":float((sv[yva==0]>thr).mean()),
            "target_fpr":alpha,"false_windows_per_day":float(fire[neg].sum()/duration),
            "threshold":thr,"test_negative_windows":int(neg.sum())})
    pd.DataFrame(metrics).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)
    (out/"audit.json").write_text(json.dumps({"seed":seed,"days":cfg.days,
        "generated_compromise":sum(e.kind=="compromise" for e in episodes),
        "eligible_test_episodes":len(eligible),"test_start":start,
        "validation_end_exclusive":cfg.n_intervals//2-cfg.feat_window-cfg.z_window,
        "test_windows":len(te),"validation_windows":len(va),"weights_frozen":True},indent=2))
    print("complete",seed,len(eligible),flush=True)


def paired(x,y):
    d=np.asarray(x)-np.asarray(y)
    n=len(d); rng=np.random.default_rng(9861)
    boot=rng.choice(d,size=(20000,n),replace=True).mean(axis=1)
    # exact paired sign-flip randomization; two-sided, topology-level.
    null=np.array([np.mean(d*np.array(s)) for s in itertools.product((-1,1),repeat=n)])
    return {"n_worlds":n,"delta":float(d.mean()),"ci_low":float(np.quantile(boot,.025)),
            "ci_high":float(np.quantile(boot,.975)),
            "p_raw":float(np.mean(np.abs(null)>=abs(d.mean())-1e-12))}


def aggregate(root,out):
    m=pd.concat([pd.read_csv(p) for p in sorted(root.rglob("metrics.csv"))],ignore_index=True)
    e=pd.concat([pd.read_csv(p) for p in sorted(root.rglob("episodes.csv"))],ignore_index=True)
    if set(m.seed)!=set(EVAL) or m.duplicated(["seed","model"]).any():
        raise RuntimeError("Missing or duplicate independent test worlds")
    p=m.pivot(index="seed",columns="model",values="episode_recall")
    if p.isna().any().any():
        raise RuntimeError("Incomplete paired experiment")
    comparisons=[]
    for name in p:
        if name=="hgb/full": continue
        comparisons.append({"reference":"hgb/full","comparison":name,
                            **paired(p["hgb/full"],p[name])})
    # Holm correction for all predeclared pairwise contrasts.
    ranked=sorted(comparisons,key=lambda x:x["p_raw"])
    running=0.
    for i,r in enumerate(ranked):
        running=max(running,min(1.,(len(ranked)-i)*r["p_raw"]))
        r["p_holm"]=running
    pd.DataFrame(comparisons).to_csv(out/"paired_tests.csv",index=False)
    m.to_csv(out/"all_metrics.csv",index=False)
    e.to_csv(out/"all_episodes.csv",index=False)
    s=m.groupby("model").agg(worlds=("seed","nunique"),episodes=("episodes","sum"),
          detected=("detected","sum"),mean_recall=("episode_recall","mean"),
          mean_fpr=("test_fpr","mean"),max_fpr=("test_fpr","max"),
          mean_auc=("auc","mean"),mean_ap=("ap","mean"),
          mean_false_windows_day=("false_windows_per_day","mean")).reset_index()
    s["pooled_recall"]=s.detected/s.episodes
    s.to_csv(out/"summary.csv",index=False)
    print(s.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["train","test","aggregate"])
    p.add_argument("--seed",type=int)
    p.add_argument("--models",type=Path)
    p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="train": fit(a.output)
    elif a.mode=="test": evaluate(a.seed,a.output,a.models)
    else: aggregate(a.input,a.output)


if __name__=="__main__":
    main()
