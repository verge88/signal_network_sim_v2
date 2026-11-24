"""Incident-based calibration and strictly local baselines.
All test labels are used only to evaluate a previously frozen decision rule.
One notification per node per fixed 30-minute bucket; no ranking or tuning
on test labels. A confirmed notification needs three consecutive high scores.
The bucket definition differs from uninterrupted-streak incident counts in
the previous study; report both metrics rather than equating them.
"""
import argparse
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from ss7 import fix as F
from ss7.report_only_invariant_study import NEW,DEV,generate,columns
from ss7.threshold_transfer_study import world,score,incident_starts

SEEDS=tuple(range(8701,8709))
MODELS=("strict_local","strict_local_plus_inv","own","report_full","report_inv","legacy_full")
POLICIES=("raw_matched","raw_robust","incident_matched","incident_robust","joint_robust")
RAW_BUDGET=20.
INCIDENT_BUDGET=2.
BUCKET=60
WARMUP=50

def train(out):
    # Exactly the same development seeds, simulation parameters, subsampling
    # and hyperparameters as the existing frozen comparators.
    parts=[]
    for seed in DEV:
        df,_,_=generate(seed,1.5,30)
        y=df.label.to_numpy()
        pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        ix=np.sort(np.r_[pos,np.random.default_rng(seed).choice(
            neg,min(30000,len(neg)),replace=False)])
        parts.append(df.iloc[ix])
    d=pd.concat(parts,ignore_index=True)
    y=d.label.to_numpy()
    w=np.where(y==1,(y==0).sum()/max(1,(y==1).sum()),1.)
    result={}
    for name,bases in (("strict_local",F.OWN),("strict_local_plus_inv",F.OWN+NEW)):
        cand=[c for c in columns(d,bases) if not c.endswith("__dev_zone")]
        x=d[cand].replace([np.inf,-np.inf],np.nan)
        med=x.median().fillna(0.)
        x=x.fillna(med)
        cols=[c for c in cand if x[c].nunique(dropna=False)>1]
        assert cols
        est=HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,
            min_samples_leaf=20,l2_regularization=1.,early_stopping=False,
            random_state=8300).fit(x[cols],y,sample_weight=w)
        result[name]=(cols,med[cols],est)
        print(name,len(cols),flush=True)
    joblib.dump(result,out/"local_models.joblib",compress=3)
    (out/"local_manifest.json").write_text(json.dumps(
        dict(dev=DEV,features={k:v[0] for k,v in result.items()}),indent=2))

def elapsed(d):
    return (int(d.t.max())-int(d.t.min())+1)/F.INTERVALS_PER_DAY

def triple(d,s):
    # Frames are ordered by node and time. A gap always breaks a streak.
    n=d.nid.to_numpy();t=d.t.to_numpy()
    z=np.full(len(d),-np.inf)
    valid=(n[2:]==n[1:-1])&(n[1:-1]==n[:-2])&(t[2:]==t[1:-1]+1)&(t[1:-1]==t[:-2]+1)
    z[2:]=np.where(valid,np.minimum(np.minimum(s[2:],s[1:-1]),s[:-2]),-np.inf)
    return z

def bucket_peaks(d,z):
    v=pd.DataFrame(dict(nid=d.nid.to_numpy(),bucket=d.t.to_numpy()//BUCKET,z=z))
    return v.groupby(["nid","bucket"],sort=False).z.max().to_numpy()

def threshold(x,budget,days):
    # Strict inequality handles ties and guarantees no more than floor(B*d)
    # exceedances in this calibration sample. No finite-sample transfer claim.
    x=np.sort(x[np.isfinite(x)])[::-1]
    if not len(x):raise ValueError("empty calibration sample")
    k=min(len(x)-1,int(np.floor(budget*days)))
    return float(x[k])

def notify(d,z,thr):
    idx=np.flatnonzero(z>thr)
    selected=pd.DataFrame(dict(i=idx,nid=d.nid.to_numpy()[idx],
                   bucket=d.t.to_numpy()[idx]//BUCKET))
    return selected.drop_duplicates(["nid","bucket"],keep="first").i.to_numpy(int)

def metrics(d,s,z,thr,eligible=None):
    fire=s>thr
    ix=notify(d,z,thr)
    y=d.label.to_numpy();eid=d.episode_id.to_numpy();t=d.t.to_numpy()
    # Classify notifications by all three constituent windows; mixed onset
    # alerts are reported separately. Test labels do not affect notification.
    false=0;mixed=0;detected={}
    for j in ix:
        prev=y[j-2:j+1];ep=eid[j-2:j+1]
        if np.all(prev==0):false+=1
        elif np.all(prev==1) and np.all(ep==ep[-1]):
            if eligible is not None and int(ep[-1]) in eligible:
                k=int(ep[-1])
                detected.setdefault(k,(int(t[j])-eligible[k].t0)*30)
        else:mixed+=1
    days=elapsed(d)
    neg=y==0
    result=dict(threshold=thr,days=days,notifications=len(ix),
        false_windows_day=float(fire[neg].sum()/days),
        false_notifications_day=float(false/days),
        mixed_notifications=int(mixed),
        streak_incidents_day=float(incident_starts(d,fire)/days),
        fpr=float(fire[neg].mean()))
    if eligible is not None:
        result.update(episodes=len(eligible),detected=len(detected),
            recall=len(detected)/len(eligible),
            auc=roc_auc_score(y,s),ap=average_precision_score(y,s),
            median_delay_seconds=float(np.median(list(detected.values()))) if detected else np.nan)
    return result,detected

def test(seed,localpath,frozenpath,out):
    assert seed in SEEDS
    # Independent test and commissioning worlds. Calibration matched and
    # stress have 3 days each, an independent 1.5-day benign validation
    # does not influence thresholds or model choice.
    d,sim,eps=world(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:raise ValueError("too few complete episodes")
    # Retain every window for actual operational decisions, including
    # incomplete positives; assess recall on eligible complete episodes.
    # Window AUC includes all labelled compromise windows in this study.
    a=world(seed+10000,3.,0,128,20,sim)[0]
    b=world(seed+20000,3.,0,192,32,sim)[0]
    v=world(seed+30000,1.5,0,80,13,sim)[0]
    assert all(not x.label.any() for x in (a,b,v))
    assert all(set(x.nid)==set(d.nid) for x in (a,b,v))
    models=joblib.load(frozenpath);models.update(joblib.load(localpath))
    rows=[];controls=[];events=[];retro=[]
    for name in MODELS:
        sa,sb,sv,st=[score(models[name],x) for x in (a,b,v,d)]
        za,zb,zv,zt=[triple(x,s) for x,s in zip((a,b,v,d),(sa,sb,sv,st))]
        ra=threshold(sa,RAW_BUDGET,elapsed(a))
        rb=threshold(sb,RAW_BUDGET,elapsed(b))
        ia=threshold(bucket_peaks(a,za),INCIDENT_BUDGET,elapsed(a))
        ib=threshold(bucket_peaks(b,zb),INCIDENT_BUDGET,elapsed(b))
        th={"raw_matched":ra,"raw_robust":max(ra,rb),
            "incident_matched":ia,"incident_robust":max(ia,ib),
            "joint_robust":max(ra,rb,ia,ib)}
        for policy,tau in th.items():
            m,det=metrics(d,st,zt,tau,eligible)
            rows.append(dict(seed=seed,model=name,policy=policy,**m))
            for phase,x,s,z in (("matched",a,sa,za),("stress",b,sb,zb),("validation",v,sv,zv)):
                q,_=metrics(x,s,z,tau)
                controls.append(dict(seed=seed,model=name,policy=policy,phase=phase,**q))
            for i,e in eligible.items():
                events.append(dict(seed=seed,model=name,policy=policy,episode=i,
                  detected=int(i in det),delay_seconds=det.get(i),
                  hidden_rate=e.hidden_rate,mask_family=e.family))
        # Retrospective raw-window comparison only, using true negatives:
        # not a deployable decision rule and never selected as such.
        for budget in (20.,40.,100.):
            tau=threshold(st[d.label.to_numpy()==0],budget,elapsed(d))
            q,_=metrics(d,st,zt,tau,eligible)
            retro.append(dict(seed=seed,model=name,budget=budget,**q))
        print("evaluated",seed,name,flush=True)
    for name,data in (("metrics",rows),("controls",controls),("episodes",events),("retrospective",retro)):
        pd.DataFrame(data).to_csv(out/(name+".csv"),index=False)
    (out/"manifest.json").write_text(json.dumps(dict(seed=seed,
       complete_episodes=len(eligible),generated=32,raw_budget=RAW_BUDGET,
       notification_budget=INCIDENT_BUDGET,bucket_intervals=BUCKET,
       frozen_run=45421952080,calibration_days=[3.,3.],validation_days=1.5,
       test_days=1.5,scope="simulated compromise detection; no real traffic"),indent=2))

def collect(root,name):
    ps=sorted(root.rglob(name+".csv"))
    if len(ps)!=len(SEEDS):raise ValueError(f"{name}: incomplete seeds")
    d=pd.concat([pd.read_csv(p) for p in ps],ignore_index=True)
    if set(d.seed)!=set(SEEDS):raise ValueError("seed mismatch")
    return d

def aggregate(root,out):
    m=collect(root,"metrics");c=collect(root,"controls")
    e=collect(root,"episodes");r=collect(root,"retrospective")
    if len(m)!=len(SEEDS)*len(MODELS)*len(POLICIES) or m.duplicated(["seed","model","policy"]).any():
        raise ValueError("incomplete matrix")
    def summary(d,groups):
        return d.groupby(groups).agg(worlds=("seed","nunique"),episodes=("episodes","sum"),
          detected=("detected","sum"),recall=("recall","mean"),fpr=("fpr","mean"),
          raw_day=("false_windows_day","mean"),notifications_day=("false_notifications_day","mean"),
          streak_day=("streak_incidents_day","mean"),
          worst_notifications=("false_notifications_day","max"),
          raw_within_budget=("false_windows_day",lambda x:int((x<=RAW_BUDGET).sum())),
          notification_within_budget=("false_notifications_day",lambda x:int((x<=INCIDENT_BUDGET).sum()))).reset_index()
    s=summary(m,["model","policy"]);rs=summary(r,["model","budget"])
    tests=[];rng=np.random.default_rng(8700)
    # Fixed comparisons. The same policy and same synthetic worlds are used.
    contrasts=[("report_full","incident_robust","report_full","raw_matched"),
       ("report_full","incident_robust","strict_local","incident_robust"),
       ("strict_local_plus_inv","incident_robust","strict_local","incident_robust"),
       ("report_full","joint_robust","strict_local","joint_robust")]
    for a,p,b,q in contrasts:
        x=m[(m.model==a)&(m.policy==p)].set_index("seed").sort_index()
        y=m[(m.model==b)&(m.policy==q)].set_index("seed").sort_index()
        assert x.index.equals(y.index)
        for metric in ("recall","false_notifications_day"):
            z=(x[metric]-y[metric]).to_numpy()
            null=np.array([np.mean(z*np.array(h)) for h in itertools.product((-1,1),repeat=len(z))])
            boot=rng.choice(z,size=(20000,len(z)),replace=True).mean(1)
            tests.append(dict(a=a,policy_a=p,b=b,policy_b=q,metric=metric,delta=z.mean(),
                 low=np.quantile(boot,.025),high=np.quantile(boot,.975),
                 p_raw=np.mean(np.abs(null)>=abs(z.mean())-1e-12)))
    running=0.
    for i,t in enumerate(sorted(tests,key=lambda x:x["p_raw"])):
        running=max(running,min(1.,(len(tests)-i)*t["p_raw"]));t["p_holm"]=running
    for name,table in (("all_metrics",m),("summary",s),("controls",c),
       ("episodes",e),("retrospective",r),("retrospective_summary",rs),("paired_tests",pd.DataFrame(tests))):
        table.to_csv(out/(name+".csv"),index=False)
    print(s.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--local",type=Path)
    p.add_argument("--frozen",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="train":train(a.output)
    elif a.mode=="test":test(a.seed,a.local,a.frozen,a.output)
    else:aggregate(a.input,a.output)
