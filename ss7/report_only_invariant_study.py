"""Defensive measurement-consistency study on independent simulated worlds.
The report-only residuals below use observed counters, not latent link loss.
The canonical simulator counts offered MSUs at both ends, NOT delivered MSUs;
this study does not validate an operational physical-layer loss model.
"""
from pathlib import Path
import argparse
import itertools
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from ss7 import fix as F

DEV=(8301,8302,8303)
TEST=tuple(range(8401,8409))
BUDGET=20.0
NEW=["pair_mean","pair_min","pair_absmax","pair_rms","od_pair_mean",
     "od_pair_min","od_pair_absmax","relay_balance"]
SUFFIX=("","__mean","__std","__z","__dev_zone")
LEGACY=F.all_invariants


def report_only(reps,topo):
    link={n:[] for n in reps}
    od={n:[] for n in reps}
    def residual(x,y):
        # Symmetric relative discrepancy, dimensionless, bounded by two.
        return 2.*(x-y)/max(x+y,1.)
    for a,b in topo.linksets:
        if a not in reps or b not in reps:
            continue
        k=(a,b); A=reps[a].state; B=reps[b].state
        for s,d,x,y in ((a,b,A.link_out.get(k,0),B.link_in.get(k,0)),
                        (b,a,B.link_out.get(k,0),A.link_in.get(k,0))):
            if x+y==0: continue
            r=residual(x,y)
            link[s].append(r)
            link[d].append(-r)
    pairs=set()
    for a,r in reps.items():
        for b in r.state.dest:
            if b in reps and a!=b:
                pairs.add(tuple(sorted((a,b))))
    for a,b in sorted(pairs):
        x=reps[a].state.dest.get(b,0); y=reps[b].state.dest.get(a,0)
        if x+y==0: continue
        r=residual(x,y);od[a].append(r);od[b].append(-r)
    out={}
    for n,r in reps.items():
        a=np.asarray(link[n],float); b=np.asarray(od[n],float)
        st=r.state
        x=sum(st.link_in.values());y=sum(st.link_out.values())
        relay=(topo.nodes[n].ntype in (F.NodeType.STP,F.NodeType.IGW))
        out[n]=dict(pair_mean=a.mean() if len(a) else np.nan,
                    pair_min=a.min() if len(a) else np.nan,
                    pair_absmax=np.abs(a).max() if len(a) else np.nan,
                    pair_rms=np.sqrt((a*a).mean()) if len(a) else np.nan,
                    od_pair_mean=b.mean() if len(b) else np.nan,
                    od_pair_min=b.min() if len(b) else np.nan,
                    od_pair_absmax=np.abs(b).max() if len(b) else np.nan,
                    relay_balance=residual(x,y) if relay and x+y else np.nan)
    return out


def both(reps,topo,lstate):
    old=LEGACY(reps,topo,lstate)
    new=report_only(reps,topo)
    for n in old: old[n].update(new[n])
    return old


def generate(seed,days,n,*,template=None):
    cfg=F.SimConfig(seed=seed,days=days)
    F.all_invariants=both
    try:
        df,sim,eps=F.build_dataset(cfg,episodes_kw={
            "n_compromise":n,"n_normal":max(24,2*n),"n_ext":max(8,n//3)},
            topo=template.topo if template is not None else None,
            sim_template=template)
    finally:
        F.all_invariants=LEGACY
    df=F.temporal_block(df,NEW,cfg.feat_window)
    df=F.zone_deviation(df,NEW,{n:v.zone for n,v in sim.topo.nodes.items()})
    return df,sim,eps


def columns(df,bases):
    return [b+s for b in bases for s in SUFFIX if b+s in df]


def train(out):
    data=[]
    for seed in DEV:
        df,_,_=generate(seed,1.5,30)
        pos=np.flatnonzero(df.label.to_numpy()==1)
        neg=np.flatnonzero(df.label.to_numpy()==0)
        ix=np.sort(np.r_[pos,np.random.default_rng(seed).choice(neg,min(30000,len(neg)),replace=False)])
        data.append(df.iloc[ix])
        print("dev",seed,len(ix),flush=True)
    d=pd.concat(data,ignore_index=True)
    specs={"own":F.OWN,"legacy_inv":F.INV,"legacy_full":F.OWN+F.INV,
           "report_inv":NEW,"report_full":F.OWN+NEW,
           "report_no_od":F.OWN+[b for b in NEW if not b.startswith("od_")],
           "report_link_only":[b for b in NEW if b.startswith("pair_")]}
    y=d.label.to_numpy();w=np.where(y==1,(y==0).sum()/max(1,(y==1).sum()),1.)
    models={}
    for name,base in specs.items():
        candidates=columns(d,base)
        X=d[candidates].replace([np.inf,-np.inf],np.nan)
        # Fit the preprocessing exclusively on development data, uniformly
        # for all variants. HGB binning cannot handle all-missing/constant
        # coordinates reliably in the pinned library version.
        med=X.median()
        X=X.fillna(med).fillna(0.)
        c=[x for x in candidates if X[x].nunique(dropna=False)>1]
        if not c:
            raise ValueError(f"No varying development features: {name}")
        dropped=[x for x in candidates if x not in c]
        print("preprocessing",name,"dropped",dropped,flush=True)
        model=HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,
           min_samples_leaf=20,l2_regularization=1.,early_stopping=False,
           random_state=8300).fit(X[c],y,sample_weight=w)
        models[name]=(c,med[c].fillna(0.),model)
        print("fitted",name,len(c),flush=True)
    joblib.dump(models,out/"models.joblib",compress=3)
    (out/"manifest.json").write_text(json.dumps({"dev":DEV,"test":TEST,
          "features":{k:v[0] for k,v in models.items()}},indent=2))


def test(seed,out,modelpath):
    if seed not in TEST:raise ValueError("not a fixed evaluation seed")
    # Fixed weights. A separate benign-only compromise-free world with the
    # same topology, different time/random stream supplies threshold data.
    d,sim,eps=generate(seed,1.5,32)
    cal,_,_=generate(seed+10000,1.,0,template=sim)
    assert not cal.label.any()
    start=sim.cfg.feat_window+sim.cfg.z_window
    d=d[d.t>=start].copy()
    cal=cal[cal.t>=start].copy()
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<start:continue
        rows=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(rows)==e.t1-e.t0+1 and rows.t.nunique()==len(rows):eligible[i]=e
    if len(eligible)<12:raise RuntimeError(f"Only {len(eligible)} full episodes")
    d=d[(d.label==0)|d.episode_id.isin(eligible)].copy()
    alpha=BUDGET/(cal.nid.nunique()*F.INTERVALS_PER_DAY)
    y=d.label.to_numpy(int);neg=y==0
    rows=[];events=[]
    models=joblib.load(modelpath)
    for name,(cols,med,model) in models.items():
        cv=cal[cols].replace([np.inf,-np.inf],np.nan).fillna(med)
        tv=d[cols].replace([np.inf,-np.inf],np.nan).fillna(med)
        sv=model.predict_proba(cv)[:,1]
        st=model.predict_proba(tv)[:,1]
        thr=float(np.quantile(sv,1-alpha,method="higher"))
        fire=st>thr
        hits=[]
        for ep,e in eligible.items():
            ix=np.flatnonzero(d.episode_id.to_numpy()==ep)
            ix=ix[np.argsort(d.t.to_numpy()[ix])]
            times=d.t.to_numpy()[ix];run=0;hit=None
            for j,k in enumerate(ix):
                run=run+1 if fire[k] and (j==0 or times[j]==times[j-1]+1) else int(fire[k])
                if run>=3:hit=int(times[j]-e.t0);break
            hits.append(hit)
            events.append(dict(seed=seed,model=name,episode=ep,detected=int(hit is not None),
                  delay_seconds=None if hit is None else hit*sim.cfg.interval_s,
                  hidden_rate=e.hidden_rate,mask_family=e.family))
        found=[h for h in hits if h is not None]
        span=(sim.cfg.n_intervals-start)/F.INTERVALS_PER_DAY
        rows.append(dict(seed=seed,model=name,episodes=len(hits),detected=len(found),
          recall=len(found)/len(hits),auc=roc_auc_score(y,st),ap=average_precision_score(y,st),
          calibration_fpr=float((sv>thr).mean()),test_fpr=float(fire[neg].mean()),
          target_fpr=alpha,false_windows_day=float(fire[neg].sum()/span),
          median_delay_seconds=float(np.median(found)*sim.cfg.interval_s) if found else np.nan))
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def aggregate(root,out):
    m=pd.concat([pd.read_csv(p) for p in sorted(root.rglob("metrics.csv"))],ignore_index=True)
    if set(m.seed)!=set(TEST) or m.duplicated(["seed","model"]).any():raise ValueError("incomplete worlds")
    p=m.pivot(index="seed",columns="model",values="recall")
    if p.isna().any().any():raise ValueError("incomplete paired results")
    tests=[];rng=np.random.default_rng(8321)
    for a,b in (("report_full","own"),("report_inv","legacy_inv"),("report_full","legacy_full")):
        d=(p[a]-p[b]).to_numpy();n=len(d)
        null=np.array([np.mean(d*np.array(s)) for s in itertools.product((-1,1),repeat=n)])
        boot=rng.choice(d,size=(20000,n),replace=True).mean(1)
        tests.append(dict(a=a,b=b,n=n,delta=d.mean(),low=np.quantile(boot,.025),
             high=np.quantile(boot,.975),p_raw=np.mean(np.abs(null)>=abs(d.mean())-1e-12)))
    running=0
    for i,r in enumerate(sorted(tests,key=lambda r:r["p_raw"])):
        running=max(running,min(1.,(len(tests)-i)*r["p_raw"]));r["p_holm"]=running
    s=m.groupby("model").agg(worlds=("seed","nunique"),episodes=("episodes","sum"),
         detected=("detected","sum"),recall=("recall","mean"),fpr=("test_fpr","mean"),
         auc=("auc","mean"),ap=("ap","mean"),false_windows_day=("false_windows_day","mean")).reset_index()
    m.to_csv(out/"all_metrics.csv",index=False)
    s.to_csv(out/"summary.csv",index=False)
    pd.DataFrame(tests).to_csv(out/"paired_tests.csv",index=False)
    print(s.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["train","test","aggregate"])
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--models",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--seed",type=int);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="train":train(a.output)
    elif a.mode=="test":test(a.seed,a.output,a.models)
    else:aggregate(a.input,a.output)

if __name__=="__main__":main()
