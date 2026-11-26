"""Dual-channel compromise detection on unchanged corrected threat model.

No attack-generation change, no authorization oracle, no true hidden state in
predictors. Behavior uses measured claimed typed reports, measured boundary
counts and causal history. These are aggregate traffic statistics, not
transaction authorization. Hidden activity sampled from legitimate flow support
may be indistinguishable. Results on balanced scenarios are exploratory until
larger independent confirmation. Expanded is frozen; refit controls training mix.
"""
import argparse,json,itertools
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ss7 import fix as F
from ss7 import observability_audit_development as O
from ss7 import factorial_model_comparison as B
from ss7 import episode_ranking_development as R
from ss7 import cusum_cascade_development as C
from ss7.weak_signal_confirmation import fit_model,weak_predict
from ss7.incident_calibration_study import elapsed,triple,metrics,notify,RAW_BUDGET,INCIDENT_BUDGET,WARMUP

TRAIN=tuple(range(15001,15005))
VAL=tuple(range(15101,15105))
TEST=tuple(range(15201,15209))
BASE=("expanded","role_expanded","refit_expanded","behavior","behavior_unmasked")
CANDIDATES=BASE[2:]+("dual_frozen_25","dual_frozen_50","dual_frozen_75",
    "dual_refit_25","dual_refit_50","dual_refit_75","dual_unmasked_50","joint")
SHARES={25:.25,50:.5,75:.75}
FRACTION=.5
WINDOWS=(10,60,240)
KINDS=("typed","matrix","border")
PROFILE=("log_total","entropy","top1","support","code_entropy","link_entropy","map_share","isup_share")
MAP={F.MT_IDX[m] for m in ("SRI_SM","MT_FSM","MO_FSM","SRI","PRN","UL","ISD","ATI","PSI","SAI","CL","PURGE")}
ISUP={F.MT_IDX[m] for m in ("IAM","ACM","ANM","REL","RLC")}
BEHAVIOR=tuple("beh_"+k+"_"+p for k in KINDS for p in PROFILE)+("beh_link_in_entropy","beh_link_out_entropy",
    "beh_direction_ratio","beh_dest_log_total","beh_origin_log","beh_term_log","beh_transit_log",
    "beh_dialog_log","beh_dialog_end_share")
ORIGINAL=O.telemetry

def entropy(v):
    a=np.asarray(v,float);a=a[a>0]
    if not len(a):return 0.
    p=a/a.sum();return float(-np.sum(p*np.log(p)))

def profile(counts):
    # keys are (direction,link,code) or (input link,output link,code).
    vals=np.asarray(list(counts.values()),float)
    vals=vals[vals>0];total=float(vals.sum())
    if not total:return (0.,0.,0.,0.,0.,0.,0.,0.)
    code={};link={}
    for k,v in counts.items():
        if v<=0:continue
        code[k[-1]]=code.get(k[-1],0.)+v
        link[k[:-1]]=link.get(k[:-1],0.)+v
    return (np.log1p(total),entropy(vals),vals.max()/total,np.log1p(len(vals)),
        entropy(list(code.values())),entropy(list(link.values())),
        sum(v for j,v in code.items() if j in MAP)/total,
        sum(v for j,v in code.items() if j in ISUP)/total)

def behavior_snapshot(reps,typed,mat,sensors,topo):
    out={}
    for n,r in reps.items():
        z={}
        for kind,counts in (("typed",typed[n]),("matrix",mat[n]),("border",sensors.get(n))):
            values=profile(counts) if counts is not None else (np.nan,)*len(PROFILE)
            z.update({"beh_"+kind+"_"+k:v for k,v in zip(PROFILE,values)})
        st=r.state
        si=sum(st.link_in.values());so=sum(st.link_out.values())
        z.update(beh_link_in_entropy=entropy(list(st.link_in.values())),
            beh_link_out_entropy=entropy(list(st.link_out.values())),
            beh_direction_ratio=(si-so)/max(si+so,1),
            beh_dest_log_total=np.log1p(sum(st.dest.values())),beh_origin_log=np.log1p(st.originated),
            beh_term_log=np.log1p(st.terminated),beh_transit_log=np.log1p(st.transited),
            beh_dialog_log=np.log1p(st.dlg_begin),beh_dialog_end_share=st.dlg_end/max(st.dlg_begin,1))
        out[n]=z
    return out

def telemetry(reps,typed,mat,sensors,topo):
    out=ORIGINAL(reps,typed,mat,sensors,topo)
    b=behavior_snapshot(reps,typed,mat,sensors,topo)
    for n in out:out[n].update(b[n])
    return out

def history(d):
    # Strictly past rolling mean/std and current deviation, reset on gaps.
    d=d.sort_values(["nid","t"]).reset_index(drop=True)
    n=d.nid.to_numpy();t=d.t.to_numpy()
    g=np.cumsum(np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)])
    out={}
    # Exclude labels, attack attributes, hidden state, and identity features.
    cols=list(BEHAVIOR)+list(F.OWN)
    for c in cols:
        x=d[c].astype(float)
        for k in WINDOWS:
            roll=x.groupby(g,sort=False).rolling(k,min_periods=max(5,k//4),closed="left")
            mu=roll.mean().reset_index(level=0,drop=True).reindex(d.index)
            sd=roll.std().reset_index(level=0,drop=True).reindex(d.index)
            delta=x-mu
            out[f"hist_{c}_delta{k}"]=delta.to_numpy()
            # Variance floor avoids huge ratios on constant/sparse streams.
            out[f"hist_{c}_z{k}"]=(delta/(sd+.05*np.maximum(mu.abs(),1.))).clip(-25,25).to_numpy()
    return pd.concat([d,pd.DataFrame(out,index=d.index)],axis=1)

def generate(seed,balanced=True,template=None,comp=0,normal=128,external=20):
    O.telemetry=telemetry
    try:d,sim,eps=B.generate("corrected",seed,balanced,template,comp,normal,external)
    finally:O.telemetry=ORIGINAL
    return history(d),sim,eps

def eligible_episodes(d,eps):
    e={i:x for i,x in enumerate(eps) if x.kind=="compromise"}
    assert len(e)==len(list(B.scenarios("corrected")))*len(B.RATES)*len(B.ROLES)
    for i,x in e.items():
        q=d[(d.episode_id==i)&(d.nid==x.nid)]
        if len(q)!=x.t1-x.t0+1 or q.t.nunique()!=len(q):raise ValueError("incomplete factorial episode")
    return e

def behavior_cols(d):
    return list(F.OWN)+list(BEHAVIOR)+[c for c in d.columns if c.startswith("hist_")]

def columns(d,n):
    if n=="refit_expanded":return O.feature_names(d,"expanded")
    b=behavior_cols(d)
    if n=="joint":return list(dict.fromkeys(O.feature_names(d,"expanded")+b))
    return b

def fit(d,n):
    # All positive windows retained for all-scenario models; unmasked
    # specialist excludes masked positives instead of treating them as normal.
    use=np.ones(len(d),bool)
    if n=="behavior_unmasked":use=(d.label.to_numpy()==0)|d.is_unmasked.to_numpy()
    d=d.loc[use];y=d.label.to_numpy();p=y==1;ep=d.uid
    counts=ep[p].value_counts();w=np.ones(len(d));w[p]=1./ep[p].map(counts).to_numpy()
    for k in (0,1):w[y==k]*=len(y)/(2*w[y==k].sum())
    names=columns(d,n);return fit_model(d[names].to_numpy(float),y,w,names)

def train(out):
    parts=[]
    for seed in TRAIN:
        d,sim,eps=generate(seed)
        eligible=eligible_episodes(d,eps)
        d["uid"]=str(seed)+":"+d.episode_id.astype(str)
        d["is_unmasked"]=d.episode_id.isin([i for i,e in eligible.items() if e.sophistication==F.Sophistication.NAIVE])
        y=d.label.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        rng=np.random.default_rng(seed);ix=np.sort(np.r_[pos,rng.choice(neg,min(50000,len(neg)),replace=False)])
        parts.append(d.iloc[ix][list(dict.fromkeys(columns(d,"joint")+["uid","label","is_unmasked"]))].copy())
        print("train",seed,len(eligible),len(pos),len(ix),flush=True)
    d=pd.concat(parts,ignore_index=True)
    models={n:fit(d,n) for n in ("refit_expanded","behavior","behavior_unmasked","joint")}
    joblib.dump(models,out/"models.joblib",compress=3)
    (out/"manifest.json").write_text(json.dumps(dict(train=TRAIN,val=VAL,test=TEST,
        features={n:m["names"] for n,m in models.items()},threat="unchanged corrected balanced"),indent=2))

def evaluate(seed,frozen,models,out,selection=None):
    assert seed in VAL+TEST
    val=seed in VAL
    if val!=(selection is None):raise ValueError("test requires frozen validation selection")
    d,sim,eps=generate(seed);eligible=eligible_episodes(d,eps)
    refs=(generate(seed+10000,False,sim,0,128,20)[0],
          generate(seed+20000,False,sim,0,192,32)[0],
          generate(seed+40000,False,sim,64,128,20)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    assert all(set(x.nid)==set(d.nid) for x in refs)
    frames=(*refs,d);ss={}
    for n in ("expanded","role_expanded"):ss[n]=[O.predict(x,frozen,n) for x in frames]
    for n,m in models.items():ss[n]=[weak_predict(m,x[m["names"]].to_numpy(float)) for x in frames]
    requested=tuple(dict.fromkeys(BASE+(CANDIDATES if val else (selection["selected"],))))
    for n in requested:
        if not n.startswith("dual_"):continue
        _,kind,share=n.split("_")
        a=ss["refit_expanded" if kind=="refit" else "expanded"]
        b=ss["behavior_unmasked" if kind=="unmasked" else "behavior"]
        tx,ty,ex=R.union(refs,a[:3],b[:3],SHARES[int(share)])
        ss[n]=[np.maximum(p-tx,q-ty)-ex for p,q in zip(a,b)]
    rows=[];events=[];controls=[];diagnostics=[]
    y=d.label.to_numpy();t=d.t.to_numpy();nid=d.nid.to_numpy();ids=d.episode_id.to_numpy()
    unmasked=np.isin(ids,[i for i,e in eligible.items() if e.sophistication==F.Sophistication.NAIVE])
    neg=y==0
    for n in requested:
        s=ss[n];tau=0. if n.startswith("dual_") else C.joint_cut(refs,s[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION)
        u=s[3];z=triple(d,u);m,det=metrics(d,u,z,tau,eligible)
        rows.append(dict(seed=seed,model=n,**m))
        for i,e in eligible.items():
            ix=(ids==i)&(nid==e.nid)
            events.append(dict(seed=seed,model=n,episode=i,detected=int(i in det),
                sophistication=e.sophistication.value,attack=B.scenario_name("corrected",e),
                hidden_rate=e.hidden_rate,ntype=sim.topo.nodes[e.nid].ntype.value,
                delay_seconds=det.get(i),raw_windows=int((u[ix]>tau).sum()),
                max_margin=float(u[ix].max()-tau),confirmed_candidate=int((z[ix]>tau).any())))
        for typ in B.ROLES:
            ix=(d.ntype.to_numpy()==typ)&(unmasked|neg)
            yy=y[ix]
            if len(np.unique(yy))==2:
                diagnostics.append(dict(seed=seed,model=n,ntype=typ,auc_unmasked=roc_auc_score(yy,u[ix]),
                    positive_windows=int(yy.sum()),negative_windows=int((yy==0).sum())))
        alerts=notify(d,z,tau)
        false_alerts=np.array([j for j in alerts if (y[j-2:j+1]==0).all()],dtype=int)
        for i,e in enumerate(eps):
            if e.kind=="compromise":continue
            scope=(t>=max(e.t0,WARMUP))&(t<=e.t1)
            if e.kind!="ext_attack" and e.nid is not None:scope&=nid==e.nid
            mask=scope&neg
            if mask.any():controls.append(dict(seed=seed,model=n,episode=i,event=e.kind,
                negative_windows=int(mask.sum()),false_windows=int((mask&(u>tau)).sum()),
                false_notifications=int(scope[false_alerts].sum())))
        print(seed,n,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    for name,q in (("metrics",rows),("episodes",events),("controls",controls),("unmasked_auc",diagnostics)):
        pd.DataFrame(q).to_csv(out/(name+".csv"),index=False)

def tables(m,e):
    return e.groupby(["model","sophistication","attack"]).agg(episodes=("episode","size"),
        detected=("detected","sum"),recall=("detected","mean"),delay=("delay_seconds","median")).reset_index()

def select(root,out):
    m,e,s=R.collect(root,VAL,tuple(dict.fromkeys(BASE+CANDIDATES)))
    a=tables(m,e);u=e[e.sophistication=="naive"].groupby("model").detected.mean().rename("unmasked_recall")
    q=s.merge(u,on="model")
    # Compliance first, then overall recall, then unmasked recall and false-window maximum.
    ranked=q[q.model.isin(CANDIDATES)].sort_values(["both_budgets","recall","unmasked_recall","worst_raw","model"],
        ascending=[False,False,False,True,True])
    chosen=str(ranked.iloc[0].model)
    (out/"selection.json").write_text(json.dumps(dict(selected=chosen,val=VAL,
        rule="compliant worlds, overall recall, unmasked recall, worst raw, name"),indent=2))
    q.to_csv(out/"validation_summary.csv",index=False);a.to_csv(out/"validation_attacks.csv",index=False)
    m.to_csv(out/"validation_metrics.csv",index=False);e.to_csv(out/"validation_episodes.csv",index=False)
    print(q.to_string(index=False));print(a.to_string(index=False));print("SELECTED",chosen,flush=True)

def aggregate(root,out,selection):
    chosen=selection["selected"];m,e,s=R.collect(root,TEST,tuple(dict.fromkeys(BASE+(chosen,))))
    def read(n):
        p=sorted(root.rglob(n+".csv"))
        if len(p)!=len(TEST):raise ValueError("missing diagnostics")
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    a=tables(m,e);cells=e.groupby(["model","sophistication","attack","hidden_rate","ntype"]).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    u=read("unmasked_auc");c=read("controls")
    cs=c.groupby(["model","event"]).agg(negative_windows=("negative_windows","sum"),
        false_windows=("false_windows","sum"),false_notifications=("false_notifications","sum")).reset_index()
    cs["fpr"]=cs.false_windows/cs.negative_windows
    p=m.pivot(index="seed",columns="model",values="recall")
    un=e[e.sophistication=="naive"].groupby(["seed","model"]).detected.mean().unstack()
    tests=[];rng=np.random.default_rng(15200)
    for endpoint,piv in (("all",p),("unmasked",un)):
        for baseline in ("expanded","refit_expanded"):
            if chosen==baseline:continue
            v=(piv[chosen]-piv[baseline]).to_numpy();boot=rng.choice(v,size=(20000,len(v)),replace=True).mean(1)
            null=np.array([np.mean(v*np.array(sign)) for sign in itertools.product((-1,1),repeat=len(v))])
            tests.append(dict(endpoint=endpoint,a=chosen,b=baseline,delta=v.mean(),lo=np.quantile(boot,.025),
                hi=np.quantile(boot,.975),p_raw=np.mean(np.abs(null)>=abs(v.mean())-1e-12)))
    prev=0.
    for i,x in enumerate(sorted(tests,key=lambda z:z["p_raw"])):
        prev=max(prev,min(1.,(len(tests)-i)*x["p_raw"]));x["p_holm"]=prev
    ua=u.groupby(["model","ntype"]).auc_unmasked.mean().reset_index()
    for n,q in (("summary",s),("attacks",a),("full_factorial",cells),("paired",pd.DataFrame(tests)),
        ("unmasked_auc",u),("unmasked_auc_summary",ua),("controls",cs),("control_events",c),("all_metrics",m),("episodes",e)):
        q.to_csv(out/(n+".csv"),index=False)
    print("VALIDATION-SELECTED",chosen);print(s.to_string(index=False));print(a.to_string(index=False))
    print("UNMASKED AUC",ua.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False))
    print("No authorization evidence; behavior detection of excess traffic is not proof of compromise.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","validate","select","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--frozen",type=Path);p.add_argument("--models",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--selection",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    selection=json.loads(a.selection.read_text()) if a.selection else None
    if a.mode=="train":train(a.output)
    elif a.mode=="select":select(a.input,a.output)
    elif a.mode=="aggregate":aggregate(a.input,a.output,selection)
    else:evaluate(a.seed,joblib.load(a.frozen),joblib.load(a.models),a.output,selection)
