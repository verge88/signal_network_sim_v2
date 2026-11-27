"""Five specialist detectors on unchanged integrated CU simulation.

The extra profile features are derived only from already received CU payloads,
including cached reports when the current CU is lost. No new sensor, attack
generation, hidden ledger, or authorization information. Existing integrated
models are frozen controls. All heads share one network-level alarm budget.
"""
import argparse,itertools,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from ss7 import integrated_cu_five_class_development as I
from ss7 import dual_channel_behavior_development as D
from ss7 import fix as F
from ss7.weak_signal_confirmation import fit_model,weak_predict
from ss7.cusum_cascade_development import raw_cut,joint_cut
from ss7.episode_ranking_development import union,collect
from ss7.incident_calibration_study import elapsed,triple,metrics,notify,RAW_BUDGET,INCIDENT_BUDGET,WARMUP

TRAIN=tuple(range(17001,17005))
VAL=tuple(range(17101,17105))
TEST=tuple(range(17201,17209))
CLASSES=I.CLASSES
FRACTION=.5
BASE=("article_cascade","expanded_cu","dual_75","refit_general","refit_context_general")
CANDIDATES=("special_equal","special_compromise","special_context_equal","special_context_compromise",
            "hybrid_context_equal","hybrid_context_compromise")
WEIGHTS={"equal":np.full(5,.2),"compromise":np.array([.4,.15,.15,.15,.15])}
ORIGINAL=D.behavior_snapshot
CODE_GROUPS={"sms":("SRI_SM","MT_FSM","MO_FSM"),"mobility":("UL","ISD","CL"),
             "location":("ATI","PSI"),"voice":("IAM","ACM","ANM","REL","RLC")}
CONTEXT=("ctx_external_dest_count","ctx_external_dest_share","ctx_dest_count","ctx_external_peer_count",
         "ctx_external_peer_share")+tuple("ctx_count_"+k for k in F.MSG_TYPES)+tuple(
         "ctx_group_"+k for k in CODE_GROUPS)+tuple("ctx_"+k+"_"+direction for k in CODE_GROUPS for direction in ("in","out"))
TEMPORAL=("ctx_external_dest_count","ctx_external_dest_share","ctx_count_IAM","ctx_count_ATI","ctx_count_PSI",
          "ctx_group_sms","ctx_group_mobility","ctx_group_location","ctx_group_voice",
          "total_messages","cu_staleness","cu_delivered")
LAGS=(10,60)
def snapshot(reps,typed,mat,sensors,topo):
    out=ORIGINAL(reps,typed,mat,sensors,topo)
    for n,r in reps.items():
        st=r.state;ty=typed[n];v=out[n]
        ext=sum(c for k,c in st.dest.items() if k in topo.nodes and topo.nodes[k].external)
        total=sum(st.dest.values())
        border=sum(c for (_,k,j),c in ty.items() if topo.nodes[topo.peer(k,n)].external)
        v.update(ctx_external_dest_count=np.log1p(ext),ctx_external_dest_share=ext/max(total,1),
                 ctx_dest_count=np.log1p(total),ctx_external_peer_count=np.log1p(border),
                 ctx_external_peer_share=border/max(sum(ty.values()),1))
        v.update({"ctx_count_"+k:np.log1p(st.mtype[j]) for j,k in enumerate(F.MSG_TYPES)})
        for k,codes in CODE_GROUPS.items():
            js={F.MT_IDX[x] for x in codes}
            v["ctx_group_"+k]=np.log1p(sum(st.mtype[j] for j in js))
            for flag,direction in enumerate(("in","out")):
                v["ctx_"+k+"_"+direction]=np.log1p(sum(c for (f,l,j),c in ty.items() if f==flag and j in js))
    return out

def context_history(d):
    # Causal, gap-aware, and no use of node identifiers as predictors.
    d=d.sort_values(["nid","t"]).reset_index(drop=True)
    n=d.nid.to_numpy();t=d.t.to_numpy()
    g=np.cumsum(np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)])
    out={}
    for c in TEMPORAL:
        x=d[c].astype(float)
        for k in LAGS:
            roll=x.groupby(g,sort=False).rolling(k,min_periods=max(5,k//4),closed="left")
            mu=roll.mean().reset_index(level=0,drop=True).reindex(d.index)
            sd=roll.std().reset_index(level=0,drop=True).reindex(d.index)
            out[f"ctxhist_{c}_delta{k}"]=(x-mu).to_numpy()
            out[f"ctxhist_{c}_z{k}"]=((x-mu)/(sd+.05*np.maximum(mu.abs(),1.))).clip(-25,25).to_numpy()
    return pd.concat([d,pd.DataFrame(out,index=d.index)],axis=1)

def generate(seed,balanced=True,template=None,stress=False):
    D.behavior_snapshot=snapshot
    try:d,sim,eps,tr=I.generate(seed,balanced,template,stress)
    finally:D.behavior_snapshot=ORIGINAL
    for c in CONTEXT:
        if c not in d:d[c]=np.nan
    return context_history(d),sim,eps,tr

def cols(d,context=False):
    base=I.cols(d,"expanded_cu")
    return list(dict.fromkeys(base+(list(CONTEXT)+[c for c in d.columns if c.startswith("ctxhist_")] if context else [])))

def fit(d,cls,context):
    names=cols(d,context)
    y=d.label.to_numpy() if cls=="all" else (d.attack==cls).to_numpy(int)
    # Equal weight per positive episode and world. Other attack classes are
    # negatives for attribution; all their positive windows are retained.
    p=y==1;ep=d.uid;w=np.ones(len(d));counts=ep[p].value_counts()
    w[p]=1./ep[p].map(counts).to_numpy()
    for k in (0,1):
        if not (y==k).any():raise ValueError("missing class "+cls)
        w[y==k]*=len(y)/(2*w[y==k].sum())
    return fit_model(d[names].to_numpy(float),y,w,names)

def train(out):
    parts=[]
    for seed in TRAIN:
        d,sim,eps,tr=generate(seed)
        d["uid"]=str(seed)+":"+d.episode_id.astype(str)
        pos=np.flatnonzero(d.label.to_numpy()==1);neg=np.flatnonzero(d.label.to_numpy()==0)
        ix=np.sort(np.r_[pos,np.random.default_rng(seed).choice(neg,min(50000,len(neg)),replace=False)])
        parts.append(d.iloc[ix][list(dict.fromkeys(cols(d,True)+["uid","label","attack"]))].copy())
        print("train",seed,d.attack.value_counts().to_dict(),flush=True)
    d=pd.concat(parts,ignore_index=True);models={}
    for context in (False,True):
        kind="context" if context else "base"
        for cls in ("all",)+CLASSES:
            models[kind+"_"+cls]=fit(d,cls,context)
            print("trained",kind,cls,len(models[kind+"_"+cls]["names"]),flush=True)
    joblib.dump(models,out/"models.joblib",compress=3)
    (out/"manifest.json").write_text(json.dumps(dict(train=TRAIN,val=VAL,test=TEST,
        features={k:v["names"] for k,v in models.items()},classes=CLASSES),indent=2))

def score(d,m):return weak_predict(m,d[m["names"]].to_numpy(float))

def fuse(refs,channels,weights):
    # All channels are probabilities. Split a conservative raw budget,
    # then calibrate the joint maximum on independent commissioning data.
    # This never spends the full budget separately for each specialist.
    assert len(channels)==len(weights) and np.isclose(sum(weights),1.)
    cuts=np.array([raw_cut(refs,a[:3],RAW_BUDGET*FRACTION*w) for a,w in zip(channels,weights)])
    margins=[np.stack([a[j]-t for a,t in zip(channels,cuts)],axis=1) for j in range(4)]
    maxima=[x.max(axis=1) for x in margins]
    extra=max(0.,joint_cut(refs,maxima[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION))
    return [x-extra for x in margins],cuts,extra

def confirm(d,margins):
    # Detection requires three consecutive windows from the SAME head.
    # Attribution uses the winning confirmed margin, not the true label.
    z=np.stack([triple(d,x) for x in margins.T],axis=1)
    return z.max(axis=1),z.argmax(axis=1)

def evaluate(seed,frozen,models,out,selection=None):
    val=seed in VAL
    if seed not in VAL+TEST or val!=(selection is None):raise ValueError("invalid phase")
    d,sim,eps,tr=generate(seed);eligible={i:e for i,e in enumerate(eps) if e.kind in CLASSES}
    assert len(eligible)==200
    for i,e in eligible.items():
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        assert len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q)
    refs=(generate(seed+10000,False,sim)[0],generate(seed+20000,False,sim,True)[0],
          generate(seed+40000,True,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    assert all(set(q.nid)==set(d.nid) for q in refs)
    frames=(*refs,d);ss={}
    for n in ("article_rf","article_special","expanded_cu","expanded_special"):
        ss[n]=[I.predict(q,frozen,n) for q in frames]
    ss["article_cascade"]=[np.maximum(a,b) for a,b in zip(ss["article_rf"],ss["article_special"])]
    a,b=ss["expanded_cu"],ss["expanded_special"]
    tx,ty,ex=union(refs,a[:3],b[:3],.75)
    ss["dual_75"]=[np.maximum(x-tx,y-ty)-ex for x,y in zip(a,b)]
    for k,m in models.items():ss[k]=[score(q,m) for q in frames]
    ss["refit_general"]=ss["base_all"];ss["refit_context_general"]=ss["context_all"]
    names=BASE+(CANDIDATES if val else (selection["selected"],))
    rows=[];events=[];controls=[];diag=[];cal=[]
    y=d.label.to_numpy();t=d.t.to_numpy();nid=d.nid.to_numpy();ids=d.episode_id.to_numpy()
    for key in ("base","context"):
        for cls in CLASSES:
            u=ss[key+"_"+cls][3];yy=(d.attack.to_numpy()==cls)
            diag.append(dict(seed=seed,features=key,attack=cls,auc=roc_auc_score(yy,u),
                ap=average_precision_score(yy,u),auc_vs_normal=roc_auc_score(yy[yy|(y==0)],u[yy|(y==0)])))
    for n in names:
        labels=None;winner=None
        if n.startswith(("special_","hybrid_")):
            key="context" if "context" in n else "base"
            weights=WEIGHTS[n.split("_")[-1]]
            ch=[ss[key+"_"+cls] for cls in CLASSES];labels=list(CLASSES)
            if n.startswith("hybrid_"):
                # Include frozen article RF as an untyped safety-net alarm.
                # All six heads compete under a single shared budget.
                ch.append(ss["article_rf"]);labels.append("unspecified")
                weights=np.r_[weights*.8,.2]
            mm,cut,extra=fuse(refs,ch,weights)
            u=mm[3].max(axis=1);z,winner=confirm(d,mm[3]);tau=0.
            for cls,c,w in zip(labels,cut,weights):cal.append(dict(seed=seed,model=n,head=cls,cut=c,weight=w,joint_extra=extra))
        else:
            s=ss[n];tau=0. if n=="dual_75" else joint_cut(refs,s[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION)
            u=s[3];z=triple(d,u)
        m,det=metrics(d,u,z,tau,eligible)
        alerts=notify(d,z,tau);first={}
        for j in alerts:
            ep=ids[j-2:j+1]
            if (y[j-2:j+1]==1).all() and (ep==ep[-1]).all() and int(ep[-1]) in eligible:
                first.setdefault(int(ep[-1]),j)
        assert set(first)==set(det)
        correct=0
        for i,e in eligible.items():
            hit=i in det
            predicted=labels[winner[first[i]]] if hit and labels is not None else "unspecified" if hit else "missed"
            attributed=int(hit and predicted==e.kind);correct+=attributed
            events.append(dict(seed=seed,model=n,episode=i,detected=int(hit),attack=e.kind,
                predicted=predicted,correct_type=attributed,delay_seconds=det.get(i),
                ntype=sim.topo.nodes[e.nid].ntype.value,sophistication=e.sophistication.value if e.kind=="compromise" else "na",
                hidden_rate=e.hidden_rate,mask_family=I.B.scenario_name("corrected",e) if e.kind=="compromise" else "na"))
        rows.append(dict(seed=seed,model=n,**m,correct_type=correct))
        false_alerts=np.array([j for j in alerts if (y[j-2:j+1]==0).all()],dtype=int)
        for i,e in enumerate(eps):
            if e.kind in CLASSES:continue
            scope=(t>=max(e.t0,WARMUP))&(t<=e.t1)
            if e.kind!="ext_attack" and e.nid is not None:scope&=nid==e.nid
            neg=scope&(y==0)
            if neg.any():controls.append(dict(seed=seed,model=n,episode=i,event=e.kind,
                negative_windows=int(neg.sum()),false_windows=int((neg&(u>tau)).sum()),
                false_notifications=int(scope[false_alerts].sum())))
        print(seed,n,m["recall"],m["false_windows_day"],m["false_notifications_day"],"correct_type",correct,flush=True)
    for n,q in (("metrics",rows),("episodes",events),("controls",controls),("head_diagnostics",diag),("calibration",cal)):
        pd.DataFrame(q).to_csv(out/(n+".csv"),index=False)

def attack_table(e):
    return e.groupby(["model","attack"]).agg(episodes=("episode","size"),detected=("detected","sum"),
        recall=("detected","mean"),correct_type=("correct_type","sum"),typed_recall=("correct_type","mean"),
        delay=("delay_seconds","median")).reset_index()

def summary(m,e,s):
    a=attack_table(e);g=a.groupby("model").agg(macro_recall=("recall","mean"),
        worst_class_recall=("recall","min"),macro_typed_recall=("typed_recall","mean")).reset_index()
    return s.merge(g,on="model"),a

def select(root,out):
    m,e,s=collect(root,VAL,BASE+CANDIDATES);q,a=summary(m,e,s)
    # Preregistered selection: compliance, worst-class recall, class-balanced
    # recall, correct attribution, overall recall. No use of final test.
    rank=q[q.model.isin(CANDIDATES)].sort_values(["both_budgets","worst_class_recall","macro_recall",
        "macro_typed_recall","recall","worst_raw","model"],ascending=[False,False,False,False,False,True,True])
    chosen=str(rank.iloc[0].model);(out/"selection.json").write_text(json.dumps(dict(selected=chosen,
        rule="compliant worlds, worst class, macro recall, macro attribution, recall, worst raw",val=VAL),indent=2))
    q.to_csv(out/"validation_summary.csv",index=False);a.to_csv(out/"validation_attacks.csv",index=False)
    print(q.to_string(index=False));print(a.to_string(index=False));print("SELECTED",chosen,flush=True)

def aggregate(root,out,selection):
    chosen=selection["selected"];m,e,s=collect(root,TEST,BASE+(chosen,));s,a=summary(m,e,s)
    def read(n):
        p=sorted(root.rglob(n+".csv"))
        if len(p)!=len(TEST):raise ValueError("missing "+n)
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    diagnostics=read("head_diagnostics");dg=diagnostics.groupby(["features","attack"])[["auc","ap","auc_vs_normal"]].mean().reset_index()
    c=read("controls").groupby(["model","event"])[["negative_windows","false_windows","false_notifications"]].sum().reset_index()
    c["fpr"]=c.false_windows/c.negative_windows
    comp=e[e.attack=="compromise"].groupby(["model","sophistication","mask_family","hidden_rate","ntype"]).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    conf=e[e.model==chosen].groupby(["attack","predicted"]).size().rename("episodes").reset_index()
    tests=[];rng=np.random.default_rng(17200)
    # Three predeclared primary contrasts. Class-specific comparisons
    # descriptive, not a significance fishing exercise.
    for baseline in ("dual_75","refit_general","refit_context_general"):
        p=e.groupby(["seed","model","attack"]).detected.mean().groupby(["seed","model"]).mean().unstack()
        v=(p[chosen]-p[baseline]).to_numpy();boot=rng.choice(v,size=(20000,len(v)),replace=True).mean(1)
        null=np.array([np.mean(v*np.array(z)) for z in itertools.product((-1,1),repeat=len(v))])
        tests.append(dict(endpoint="macro_class_recall",a=chosen,b=baseline,delta=v.mean(),lo=np.quantile(boot,.025),
            hi=np.quantile(boot,.975),p_raw=np.mean(np.abs(null)>=abs(v.mean())-1e-12)))
    prev=0.
    for i,x in enumerate(sorted(tests,key=lambda z:z["p_raw"])):
        prev=max(prev,min(1.,(len(tests)-i)*x["p_raw"]));x["p_holm"]=prev
    tables={"summary":s,"attacks":a,"compromise_factorial":comp,"attribution_confusion":conf,
        "head_diagnostics":diagnostics,"head_diagnostics_summary":dg,"paired":pd.DataFrame(tests),
        "controls":c,"calibration":read("calibration"),"all_metrics":m,"episodes":e}
    for n,q in tables.items():q.to_csv(out/(n+".csv"),index=False)
    print("VALIDATION-SELECTED",chosen);print(s.to_string(index=False));print(a.to_string(index=False))
    print("HEAD DIAGNOSTICS",dg.to_string(index=False));print("ATTRIBUTION",conf.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))
    print("Same integrated offered-traffic simulator; classification of synthetic profiles, not verified attack semantics.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","validate","select","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--frozen",type=Path);p.add_argument("--models",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--selection",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sel=json.loads(a.selection.read_text()) if a.selection else None
    if a.mode=="train":train(a.output)
    elif a.mode=="select":select(a.input,a.output)
    elif a.mode=="aggregate":aggregate(a.input,a.output,sel)
    else:evaluate(a.seed,joblib.load(a.frozen),joblib.load(a.models),a.output,sel)
