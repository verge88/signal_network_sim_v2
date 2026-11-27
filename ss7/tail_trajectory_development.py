"""Tail-focused causal trajectory specialists on unchanged integrated SS7 simulation.

Compare XGB and HGB on identical features/training rows; mine difficult benign
windows by cross-fitting pilots on disjoint training worlds. Neyman-Pearson
inspired operating point, empirical network-day budgets, not iid NP guarantee.
No subscriber authorization or new sensors. No test tuning.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from xgboost import XGBClassifier
from ss7 import five_specialist_development as S
from ss7 import integrated_cu_five_class_development as I
from ss7.cusum_cascade_development import joint_cut
from ss7.episode_ranking_development import union,collect
from ss7.incident_calibration_study import elapsed,triple,metrics,notify,RAW_BUDGET,INCIDENT_BUDGET,WARMUP

TRAIN=(19001,19002,19003,19004)
VAL=(19101,19102,19103,19104)
TEST=tuple(range(19201,19209))
CLASSES=S.CLASSES
BASE=("dual_75","frozen_special","hgb_random","hgb_tail")
CANDIDATES=("hgb_tail","xgb_random","xgb_tail","xgb_tail_smooth5","xgb_tail_smooth15","hybrid_tail")
WINDOWS=(3,10,30,60)
ANCHOR=240
FRACTION=.5
COMMON=("total_messages","mtype_entropy","map_share","in_out_ratio","cu_delivered","cu_staleness","cu_rtt_delay","cu_combined_loss_prob")
CONTEXT=tuple(S.CONTEXT)
INV=tuple(x for x in list(I.NEW)+list(I.DIRECTION)+list(I.PEER)+list(I.O.EXTRA) if any(v in x for v in ("abs","rms","gap","balance","excess","res","mean","max","deficit")))
SIGNALS=tuple(dict.fromkeys(COMMON+tuple(S.TEMPORAL)+("ctx_count_SRI_SM","ctx_count_MT_FSM","ctx_count_UL","ctx_count_IAM","ctx_location_in","ctx_location_out","ctx_voice_in","ctx_voice_out")+INV))
ROLES=("SP","STP","SCP","IGW")
def groups(d):
    n=d.nid.to_numpy();t=d.t.to_numpy()
    return np.cumsum(np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)])
def roll(x,g,w,fn="mean"):
    r=x.groupby(g,sort=False).rolling(w,min_periods=max(2,w//3))
    return getattr(r,fn)().reset_index(level=0,drop=True).reindex(x.index)
def trajectories(d):
    d=d.sort_values(["nid","t"]).reset_index(drop=True);g=groups(d);out={}
    for c in SIGNALS:
        if c not in d:continue
        x=d[c].astype(float).replace([np.inf,-np.inf],np.nan)
        # Compare short-window mean to a preceding non-overlapping baseline.
        # No future sample, no across-node or across-gap history.
        for w in WINDOWS:
            anchor=x.groupby(g,sort=False).shift(w)
            mu=roll(anchor,g,ANCHOR);sd=roll(anchor,g,ANCHOR,"std")
            short=roll(x,g,w);delta=short-mu
            out[f"traj_{c}_mean{w}"]=short.to_numpy(np.float32)
            out[f"traj_{c}_z{w}"]=(delta/(sd+.03*np.maximum(mu.abs(),1.))).clip(-40,40).to_numpy(np.float32)
    for r in ROLES:out["role_"+r]=(d.ntype==r).to_numpy(np.float32)
    return pd.concat([d,pd.DataFrame(out,index=d.index)],axis=1)
def generate(seed,balanced=True,template=None,stress=False):
    d,sim,eps,tr=S.generate(seed,balanced,template,stress)
    return trajectories(d),sim,eps,tr

def cols(d,cls):
    # Predeclared task-specific context; include shared measurements and
    # invariant evidence for every head to avoid hard role-based gating.
    extra={
       "compromise":CONTEXT+INV,
       "sms_intercept":tuple(c for c in CONTEXT if any(k in c.lower() for k in ("sms","sri","fsm","ul","mobility","dest","peer"))),
       "location_track":tuple(c for c in CONTEXT if any(k in c.lower() for k in ("ati","psi","location","dest","peer"))),
       "signaling_dos":CONTEXT,
       "irsf":tuple(c for c in CONTEXT if any(k in c.lower() for k in ("iam","voice","external","dest","peer","acm","anm","rel","rlc")))
    }[cls]
    selected=set(COMMON+INV+extra)
    base=list(dict.fromkeys(list(I.ARTICLE)+list(I.F.OWN)+sorted(selected)+["role_"+r for r in ROLES]))
    names=[c for c in base if c in d]+[c for c in d if c.startswith("traj_") and any(c.startswith("traj_"+b+"_") for b in selected)]
    return list(dict.fromkeys(names))

def matrix(d,names):
    x=d[names].to_numpy(dtype=np.float32,copy=True);x[~np.isfinite(x)]=np.nan;return x
def fit(d,cls,algorithm,hard_weight=1.,pilot=False):
    names=cols(d,cls);x=matrix(d,names);y=(d.attack==cls).to_numpy(int)
    assert y.any() and (~y.astype(bool)).any()
    w=np.ones(len(d));p=y==1;counts=d.uid[p].value_counts()
    w[p]=1./d.uid[p].map(counts).to_numpy()
    # Hard negatives are benign training windows only, independently
    # selected by a pilot fitted on other training seeds.
    hard=(d["hard_"+cls].to_numpy(bool) if "hard_"+cls in d else np.zeros(len(d),bool)) & (d.label.to_numpy()==0)
    w[hard]*=hard_weight
    for k in (0,1):w[y==k]*=len(y)/(2*w[y==k].sum())
    if algorithm=="xgb":
        est=XGBClassifier(n_estimators=260,max_depth=5,learning_rate=.045,min_child_weight=10,
            subsample=.85,colsample_bytree=.8,reg_lambda=8.,reg_alpha=.15,tree_method="hist",
            n_jobs=2,random_state=19000,eval_metric="logloss")
    else:
        est=HistGradientBoostingClassifier(max_iter=100 if pilot else 200,max_leaf_nodes=15,
            learning_rate=.05,min_samples_leaf=40,l2_regularization=5.,early_stopping=False,random_state=19000)
    est.fit(x,y,sample_weight=w)
    return dict(names=names,model=est)
def predict(d,m):
    x=matrix(d,m["names"]);p=m["model"].predict_proba(x)[:,1]
    if not np.isfinite(p).all():raise ValueError("nonfinite predictions")
    return p
def sample(d,seed,hard_models=None):
    rng=np.random.default_rng(seed);pos=np.flatnonzero(d.label.to_numpy()==1);neg=np.flatnonzero(d.label.to_numpy()==0)
    # Identical random baseline sample for all arms, all positive rows kept.
    random=np.sort(np.r_[pos,rng.choice(neg,min(50000,len(neg)),replace=False)])
    hard={}
    if hard_models:
        for cls,m in hard_models.items():
            p=predict(d,m)
            # Peak of confirmed 3-window score catches rare sustained false alarms.
            s=np.maximum(p,triple(d,p))
            top=neg[np.argsort(-s[neg],kind="stable")[:6000]]
            # Retain hard window neighbors within the same node/contiguous segment
            # to train on both onset and sustained normal bursts.
            g=groups(d);near=np.clip(top[:,None]+np.arange(-2,3)[None,:],0,len(d)-1)
            same=g[near]==g[top,None]
            near=np.unique(near[same & (d.label.to_numpy()[near]==0)])
            hard[cls]=np.unique(np.r_[top,near])
    ix=np.unique(np.concatenate([random]+list(hard.values())))
    names=list(dict.fromkeys(sum([cols(d,c) for c in CLASSES],[])+["label","attack","episode_id"]))
    q=d.iloc[ix][names].copy();q["uid"]=str(seed)+":"+q.episode_id.astype(str)
    q["random"]=np.isin(ix,random)
    for cls in CLASSES:q["hard_"+cls]=np.isin(ix,hard.get(cls,[]))
    return q
def train(out):
    parts=[]
    for seed in TRAIN:
        d,sim,eps,tr=generate(seed);q=sample(d,seed);parts.append(q)
        print("pilot data",seed,len(q),flush=True)
    # Two-fold cross-fit: a pilot never sees the world on which it mines.
    pilots=[]
    for ids in ((0,1),(2,3)):
        d=pd.concat([parts[i] for i in ids],ignore_index=True)
        pilots.append({c:fit(d,c,"hgb",pilot=True) for c in CLASSES})
        del d
    del parts
    parts=[]
    for i,seed in enumerate(TRAIN):
        d,sim,eps,tr=generate(seed);q=sample(d,seed,pilots[1 if i<2 else 0]);parts.append(q)
        print("crossfit mined",seed,len(q),{c:int(q["hard_"+c].sum()) for c in CLASSES},flush=True)
    d=pd.concat(parts,ignore_index=True);models={}
    for alg in ("hgb","xgb"):
        for arm in ("random","tail"):
            q=d[d.random] if arm=="random" else d
            for c in CLASSES:
                models[alg+"_"+arm+"_"+c]=fit(q,c,alg,4. if arm=="tail" else 1.)
                print("trained",alg,arm,c,len(models[alg+"_"+arm+"_"+c]["names"]),flush=True)
    joblib.dump(models,out/"models.joblib",compress=3)
    (out/"manifest.json").write_text(json.dumps(dict(train=TRAIN,val=VAL,test=TEST,
        crossfit=[[19001,19002],[19003,19004]],features={k:v["names"] for k,v in models.items()},windows=WINDOWS,anchor=ANCHOR),indent=2))
def smooth(d,p,w):
    # Average logits causally, without crossing gaps. No extra learned parameters.
    z=pd.Series(np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1.)),index=d.index)
    v=roll(z,groups(d),w).fillna(z).clip(-20,20).to_numpy()
    return 1/(1+np.exp(-v))

def evaluate(seed,frozen,special,models,out,selection=None):
    val=seed in VAL
    if seed not in VAL+TEST or val!=(selection is None):raise ValueError("invalid phase")
    d,sim,eps,tr=generate(seed);eligible={i:e for i,e in enumerate(eps) if e.kind in CLASSES}
    assert len(eligible)==200
    for i,e in eligible.items():
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        assert len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q)
    refs=(generate(seed+10000,False,sim)[0],generate(seed+20000,False,sim,True)[0],generate(seed+40000,True,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    frames=(*refs,d);ss={}
    a=[I.predict(q,frozen,"expanded_cu") for q in frames];b=[I.predict(q,frozen,"expanded_special") for q in frames]
    tx,ty,ex=union(refs,a[:3],b[:3],.75);dual=[np.maximum(x-tx,y-ty)-ex for x,y in zip(a,b)]
    for key in ("hgb_random","hgb_tail","xgb_random","xgb_tail"):
        ss[key]=[[predict(q,models[key+"_"+c]) for q in frames] for c in CLASSES]
    ss["frozen_special"]=[[S.score(q,special["context_"+c]) for q in frames] for c in CLASSES]
    for w in (5,15):ss[f"xgb_tail_smooth{w}"]=[[smooth(q,p,w) for q,p in zip(frames,s)] for s in ss["xgb_tail"]]
    ss["hybrid_tail"]=[[np.maximum(x,y) for x,y in zip(a,b)] for a,b in zip(ss["xgb_tail"],ss["frozen_special"])]
    names=tuple(dict.fromkeys(BASE+(CANDIDATES if val else (selection["selected"],))))
    rows=[];events=[];diag=[];controls=[];cal=[]
    y=d.label.to_numpy();ids=d.episode_id.to_numpy();t=d.t.to_numpy();nid=d.nid.to_numpy()
    for n in names:
        if n=="dual_75":u=dual[3];z=triple(d,u);win=None
        else:
            margins,cuts,extra=S.fuse(refs,ss[n],np.full(5,.2))
            u=margins[3].max(1);z,win=S.confirm(d,margins[3])
            for c,cut,p in zip(CLASSES,cuts,ss[n]):
                yy=d.attack.to_numpy()==c
                diag.append(dict(seed=seed,model=n,attack=c,auc=roc_auc_score(yy,p[3]),ap=average_precision_score(yy,p[3]),
                    partial_auc=roc_auc_score(yy,p[3],max_fpr=.001)))
                cal.append(dict(seed=seed,model=n,head=c,cut=cut,extra=extra))
        m,det=metrics(d,u,z,0.,eligible);alerts=notify(d,z,0.);first={}
        for j in alerts:
            ep=ids[j-2:j+1]
            if (y[j-2:j+1]==1).all() and (ep==ep[-1]).all() and int(ep[-1]) in eligible:first.setdefault(int(ep[-1]),j)
        assert set(first)==set(det)
        for i,e in eligible.items():
            hit=i in det;pred=CLASSES[win[first[i]]] if hit and win is not None else "unspecified" if hit else "missed"
            events.append(dict(seed=seed,model=n,episode=i,detected=int(hit),attack=e.kind,predicted=pred,
                correct_type=int(pred==e.kind),delay_seconds=det.get(i),ntype=sim.topo.nodes[e.nid].ntype.value,
                sophistication=e.sophistication.value if e.kind=="compromise" else "na",
                hidden_rate=e.hidden_rate,mask_family=I.B.scenario_name("corrected",e) if e.kind=="compromise" else "na"))
        rows.append(dict(seed=seed,model=n,**m))
        false=np.array([j for j in alerts if (y[j-2:j+1]==0).all()],dtype=int)
        for i,e in enumerate(eps):
            if e.kind in CLASSES:continue
            scope=(t>=max(e.t0,WARMUP))&(t<=e.t1)
            if e.kind!="ext_attack" and e.nid is not None:scope&=nid==e.nid
            neg=scope&(y==0)
            if neg.any():controls.append(dict(seed=seed,model=n,episode=i,event=e.kind,negative_windows=int(neg.sum()),
                false_windows=int((neg&(u>0)).sum()),false_notifications=int(scope[false].sum())))
        print(seed,n,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    for n,q in (("metrics",rows),("episodes",events),("diagnostics",diag),("controls",controls),("calibration",cal)):
        pd.DataFrame(q).to_csv(out/(n+".csv"),index=False)
def summarize(e,s):
    a=e.groupby(["model","attack"]).agg(episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean"),
        typed_recall=("correct_type","mean"),delay=("delay_seconds","median")).reset_index()
    return s.merge(a.groupby("model").agg(macro_recall=("recall","mean"),worst_class_recall=("recall","min"),
        macro_typed_recall=("typed_recall","mean")).reset_index(),on="model"),a
def select(root,out):
    m,e,s=collect(root,VAL,tuple(dict.fromkeys(BASE+CANDIDATES)));s,a=summarize(e,s)
    rank=s[s.model.isin(CANDIDATES)].sort_values(["both_budgets","worst_class_recall","macro_recall","recall","worst_raw","model"],
        ascending=[False,False,False,False,True,True])
    chosen=str(rank.iloc[0].model);(out/"selection.json").write_text(json.dumps(dict(selected=chosen,val=VAL),indent=2))
    s.to_csv(out/"validation_summary.csv",index=False);a.to_csv(out/"validation_attacks.csv",index=False)
    print(s.to_string(index=False));print(a.to_string(index=False));print("SELECTED",chosen,flush=True)
def aggregate(root,out,selection):
    chosen=selection["selected"];m,e,s=collect(root,TEST,tuple(dict.fromkeys(BASE+(chosen,))));s,a=summarize(e,s)
    def read(n):
        p=sorted(root.rglob(n+".csv"))
        if len(p)!=len(TEST):raise ValueError("missing "+n)
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    tests=[]
    for baseline in ("dual_75","frozen_special","hgb_random","hgb_tail"):
        if chosen==baseline:continue
        p=e.groupby(["seed","model","attack"]).detected.mean().groupby(["seed","model"]).mean().unstack()
        v=(p[chosen]-p[baseline]).to_numpy();rng=np.random.default_rng(19200)
        boot=rng.choice(v,(20000,len(v)),replace=True).mean(1)
        null=np.array([np.mean(v*np.array(z)) for z in itertools.product((-1,1),repeat=len(v))])
        tests.append(dict(endpoint="macro_recall",a=chosen,b=baseline,delta=v.mean(),lo=np.quantile(boot,.025),
            hi=np.quantile(boot,.975),p_raw=np.mean(np.abs(null)>=abs(v.mean())-1e-12)))
    prev=0.
    for i,x in enumerate(sorted(tests,key=lambda x:x["p_raw"])):
        prev=max(prev,min(1.,(len(tests)-i)*x["p_raw"]));x["p_holm"]=prev
    diag=read("diagnostics");dg=diag.groupby(["model","attack"])[["auc","ap","partial_auc"]].mean().reset_index()
    comp=e[e.attack=="compromise"].groupby(["model","ntype","sophistication","mask_family","hidden_rate"]).detected.agg(["size","sum","mean"]).reset_index()
    ctrl=read("controls").groupby(["model","event"])[["negative_windows","false_windows","false_notifications"]].sum().reset_index()
    ctrl["fpr"]=ctrl.false_windows/ctrl.negative_windows
    for n,q in (("summary",s),("attacks",a),("paired",pd.DataFrame(tests)),("all_metrics",m),("episodes",e),
        ("head_diagnostics",diag),("head_summary",dg),("controls",ctrl),("compromise",comp),("calibration",read("calibration"))):
        q.to_csv(out/(n+".csv"),index=False)
    print("SELECTED",chosen);print(s.to_string(index=False));print(a.to_string(index=False));print(dg.to_string(index=False));print(pd.DataFrame(tests).to_string(index=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","validate","select","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--frozen",type=Path);p.add_argument("--special",type=Path);p.add_argument("--models",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--selection",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sel=json.loads(a.selection.read_text()) if a.selection else None
    if a.mode=="train":train(a.output)
    elif a.mode=="select":select(a.input,a.output)
    elif a.mode=="aggregate":aggregate(a.input,a.output,sel)
    else:evaluate(a.seed,joblib.load(a.frozen),joblib.load(a.special),joblib.load(a.models),a.output,sel)
