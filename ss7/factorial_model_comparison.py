"""Balanced, paired model x compromise scenario benchmark.

All weights are frozen from successful earlier runs. Legacy and corrected
threat arms are separate estimands. New telemetry models cannot run on
legacy data lacking those measurements. Legacy report-only models on the
corrected arm are explicitly out-of-domain transfers.
External floods and normal events are negative controls for compromise.
"""
import argparse,itertools,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7 import weak_signal_confirmation as W
from ss7 import meta_episode_development as M
from ss7 import episode_ranking_development as R
from ss7 import multiscale_role_development as U
from ss7 import observability_audit_development as O
from ss7 import cusum_cascade_development as C
from ss7.threshold_transfer_study import score
from ss7.incident_calibration_study import elapsed,triple,metrics,notify,RAW_BUDGET,INCIDENT_BUDGET,WARMUP

SEEDS=tuple(range(14001,14009))
ROLES=("SP","SCP","STP","IGW")
RATES=(.05,.15,.4,1.,2.5)
ARMS=("legacy","corrected")
DAYS=3.
OLD_MAKE=F.make_episodes

def scenarios(arm):
    for soph in F.Sophistication:
        if soph==F.Sophistication.NAIVE:families=("F1_uniform",)
        elif arm=="legacy":families=F.MASK_FAMILIES
        else:families=("F1_uniform","F4_temporal")
        for family in families:
            name="unmasked" if soph==F.Sophistication.NAIVE else (
                family if arm=="legacy" else "coherent_temporal" if family=="F4_temporal" else "coherent_fixed")
            yield soph,family,name

def scenario_name(arm,e):
    return next(name for soph,family,name in scenarios(arm) if soph==e.sophistication and family==e.family)

def balanced(arm,sim,cfg,rng,**kw):
    # Schedule all cells once per topology; no overlap between compromise
    # episodes on the same node; other benign controls can overlap.
    controls=OLD_MAKE(sim,cfg,rng,**dict(kw,n_compromise=0))
    cells=list(itertools.product(list(scenarios(arm)),RATES,ROLES))
    rng.shuffle(cells);occupied={n:[] for n in sim.topo.slaves};eps=[]
    for (soph,family,_),rate,role in cells:
        nodes=[n for n in sim.topo.slaves if sim.topo.nodes[n].ntype.value==role]
        if not nodes:raise ValueError("missing node type")
        length=int(rng.integers(60,121))
        for _ in range(5000):
            n=int(rng.choice(nodes));start=int(rng.integers(WARMUP+5,cfg.n_intervals-length-5))
            end=start+length-1
            if all(end+3<a or start>b+3 for a,b in occupied[n]):break
        else:raise ValueError("cannot schedule registered cell")
        occupied[n].append((start,end))
        eps.append(F.Episode(kind="compromise",nid=n,t0=start,t1=end,hidden_rate=rate,
            family=family,sophistication=soph,budget=float(rng.uniform(.7,1.)),params=rng.random(2)))
    return eps+controls

def generate(arm,seed,balanced_test=False,template=None,comp=0,normal=128,external=20):
    gen=W.generate if arm=="legacy" else O.generate
    if balanced_test:F.make_episodes=lambda sim,cfg,rng,**kw:balanced(arm,sim,cfg,rng,**kw)
    try:return gen(seed,DAYS,comp,normal,external,template)
    finally:F.make_episodes=OLD_MAKE

def load_all(root):
    read=lambda p:joblib.load(root/p)
    return dict(local=read("local/local_models.joblib"),frozen=read("frozen/models.joblib"),
        specialist=read("specialist/specialist.joblib"),meta=read("meta/meta.joblib"),weak=read("weak/weak.joblib"),
        ranks=read("ranks/ranks.joblib"),multi=read("multi/multiscale.joblib"),
        corrected=read("corrected/models.joblib"))

def all_scores(arm,frames,models):
    refs=frames[:3];ss={};margin=set()
    def assign(n,m):ss[n]=[score(m,d) for d in frames]
    def fusion(name,a,b,share=.5,role=False):
        x,y=ss[a],ss[b]
        if role:
            x=U.role_tail(refs,x[:3],list(zip(frames,x)))
            y=U.role_tail(refs,y[:3],list(zip(frames,y)))
        tx,ty,extra=R.union(refs,x[:3],y[:3],share)
        ss[name]=[np.maximum(p-tx,q-ty)-extra for p,q in zip(x,y)];margin.add(name)
    assign("strict_local",models["local"]["strict_local"])
    for n in ("own","report_inv","report_full","report_no_od","report_link_only"):
        assign(n,models["frozen"][n])
    for n,m in models["specialist"].items():assign(n,m)
    a=ss["report_full"];b=ss["peer_only"]
    ss["max_pool"]=[np.maximum(x,y) for x,y in zip(a,b)]
    tx,ty,ex=M.union(refs,a[:3],b[:3])
    ss["budget_union"]=[np.maximum(x-tx,y-ty)-ex for x,y in zip(a,b)];margin.add("budget_union")
    for n,m in models["meta"].items():
        if n=="meta_route" and arm!="legacy":continue
        ss[n]=[M.predict(m,M.features(d,x,y,n=="meta_route")) for d,x,y in zip(frames,a,b)]
    h=ss["meta_hgb"]
    for n,m in models["weak"].items():
        if n=="weak_route" and arm!="legacy":continue
        ss[n]=[W.weak_predict(m,W.matrix(d,x,y,z,n=="weak_route")[0]) for d,x,y,z in zip(frames,a,b,h)]
    for name,bn in (("union_weak","weak"),("union_route","weak_route")):
        if bn not in ss:continue
        x,y,ex=W.combine(refs,h[:3],ss[bn][:3])
        ss[name]=[np.maximum(p-x,q-y)-ex for p,q in zip(h,ss[bn])];margin.add(name)
    for n,m in models["ranks"].items():
        ss[n]=[R.rank_predict(m,R.ranking_features(x,y,z,w)) for x,y,z,w in zip(a,b,h,ss["weak"])]
    for bname in ("weak","rank8"):
        for share in R.SHARES:fusion(f"union_{bname}_{share}","meta_hgb",bname,share)
    for n,m in models["multi"].items():
        ss[n]=[W.weak_predict(m,U.inputs(d,({"primary":models["frozen"]["report_full"],
            "peer":models["specialist"]["peer_only"]},models["meta"]["meta_hgb"],models["weak"]["weak"]),n=="multiscale")[0]) for d in frames]
    for bname,suffix in (("rank2","rank"),("multiscale","multi")):
        fusion("union_weak_"+suffix,"weak",bname)
        fusion("role_weak_"+suffix,"weak",bname,role=True)
    for v in C.VARIANTS:
        n,k,q,share=v
        if not k:continue
        s=ss[n];rule=C.fit(refs,s[:3],k,q,share);name=C.name(v)
        ss[name]=[C.apply(d,z,rule)[0] for d,z in zip(frames,s)];margin.add(name)
    if arm=="corrected":
        for n in O.MODES:ss["corrected_"+n]=[O.predict(d,models["corrected"],n) for d in frames]
    else:
        for n in ("local","report"):ss["corrected_"+n]=[O.predict(d,models["corrected"],n) for d in frames]
    for n,s in ss.items():
        if not all(np.isfinite(x).all() for x in s):raise ValueError(f"nonfinite {n}")
    return ss,margin

def evaluate(seed,arm,models,out):
    assert seed in SEEDS and arm in ARMS
    d,sim,eps=generate(arm,seed,True)
    eligible={i:e for i,e in enumerate(eps) if e.kind=="compromise"}
    expected=len(list(scenarios(arm)))*len(RATES)*len(ROLES)
    assert len(eligible)==expected
    for i,e in eligible.items():
        ix=(d.episode_id==i)&(d.nid==e.nid)
        if ix.sum()!=e.t1-e.t0+1 or d.loc[ix,"t"].nunique()!=ix.sum():
            raise ValueError(f"incomplete factorial episode {i}")
    refs=(generate(arm,seed+10000,template=sim,normal=128,external=20)[0],
          generate(arm,seed+20000,template=sim,normal=192,external=32)[0],
          generate(arm,seed+40000,template=sim,comp=64,normal=128,external=20)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    assert all(set(x.nid)==set(d.nid) for x in refs)
    frames=(*refs,d);ss,margin=all_scores(arm,frames,models)
    rows=[];events=[];controls=[]
    y=d.label.to_numpy();t=d.t.to_numpy();nid=d.nid.to_numpy();ids=d.episode_id.to_numpy()
    for n,s in ss.items():
        tau=0. if n in margin else C.joint_cut(refs,s[:3],RAW_BUDGET*.5,INCIDENT_BUDGET*.5)
        u=s[3];z=triple(d,u);m,det=metrics(d,u,z,tau,eligible)
        rows.append(dict(arm=arm,seed=seed,model=n,**m))
        for i,e in eligible.items():
            ix=(ids==i)&(nid==e.nid)
            events.append(dict(arm=arm,seed=seed,model=n,episode=i,detected=int(i in det),
                sophistication=e.sophistication.value,attack=scenario_name(arm,e),hidden_rate=e.hidden_rate,
                ntype=sim.topo.nodes[e.nid].ntype.value,delay_seconds=det.get(i),
                peak_margin=float(u[ix].max()-tau),raw_windows=int((u[ix]>tau).sum()),
                confirmed_candidate=int((z[ix]>tau).any())))
        alerts=notify(d,z,tau)
        false_alerts=np.array([j for j in alerts if (y[j-2:j+1]==0).all()],dtype=int)
        for i,e in enumerate(eps):
            if e.kind=="compromise":continue
            # External controls use all monitored domestic nodes, normal
            # controls use affected node. Association is not causal attribution.
            scope=(t>=max(e.t0,WARMUP))&(t<=e.t1)
            if e.kind!="ext_attack" and e.nid is not None:scope&=nid==e.nid
            neg=scope&(y==0)
            if not neg.any():continue
            controls.append(dict(arm=arm,seed=seed,model=n,episode=i,event=e.kind,
                negative_windows=int(neg.sum()),false_windows=int((neg&(u>tau)).sum()),
                false_notifications=int(scope[false_alerts].sum())))
        print(arm,seed,n,m["recall"],m["false_windows_day"],flush=True)
    for n,data in (("metrics",rows),("episodes",events),("controls",controls)):
        pd.DataFrame(data).to_csv(out/(n+".csv"),index=False)
    (out/"manifest.json").write_text(json.dumps(dict(arm=arm,seed=seed,models=list(ss),complete=len(eligible),
        days=DAYS,reference_seeds=[seed+10000,seed+20000,seed+40000],
        scope="frozen models; same world per arm; external controls not compromise"),indent=2))

def aggregate(root,out):
    def read(n):
        p=sorted(root.rglob(n+".csv"))
        if len(p)!=len(ARMS)*len(SEEDS):raise ValueError("incomplete registered arm/world matrix")
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    m,e,c=[read(n) for n in ("metrics","episodes","controls")]
    for arm in ARMS:
        sub=m[m.arm==arm];ep=e[e.arm==arm]
        names=set(sub.model)
        if sub.duplicated(["seed","model"]).any() or set(sub[["seed","model"]].itertuples(index=False,name=None))!={(s,n) for s in SEEDS for n in names}:
            raise ValueError("missing model/world")
        cells={(s.value,n,r,t) for s,_,n in scenarios(arm) for r in RATES for t in ROLES}
        for (seed,name),q in ep.groupby(["seed","model"]):
            if q.episode.duplicated().any() or set(q[["sophistication","attack","hidden_rate","ntype"]].itertuples(index=False,name=None))!=cells or len(q)!=len(cells):
                raise ValueError("incomplete attack factorial")
        for seed,q in ep.groupby("seed"):
            support=[set(x.episode) for _,x in q.groupby("model")]
            if len(support)!=len(names) or not all(x==support[0] for x in support):raise ValueError("unpaired episodes")
    m["both_budgets"]=(m.false_windows_day<=RAW_BUDGET)&(m.false_notifications_day<=INCIDENT_BUDGET)
    summary=m.groupby(["arm","model"]).agg(worlds=("seed","nunique"),episodes=("episodes","sum"),
        detected=("detected","sum"),recall=("recall","mean"),recall_sd=("recall","std"),auc=("auc","mean"),
        ap=("ap","mean"),fpr=("fpr","mean"),raw_day=("false_windows_day","mean"),
        worst_raw=("false_windows_day","max"),notice_day=("false_notifications_day","mean"),
        worst_notice=("false_notifications_day","max"),both_budgets=("both_budgets","sum"),
        median_world_delay=("median_delay_seconds","median")).reset_index()
    def group(keys):
        return e.groupby(["arm","model"]+keys).agg(episodes=("episode","size"),detected=("detected","sum"),
            recall=("detected","mean"),delay_seconds=("delay_seconds","median")).reset_index()
    attack=group(["sophistication","attack"]);cells=group(["sophistication","attack","hidden_rate","ntype"])
    controls=c.groupby(["arm","model","event"]).agg(episodes=("episode","size"),negative_windows=("negative_windows","sum"),
        false_windows=("false_windows","sum"),false_notifications=("false_notifications","sum"),
        affected_events=("false_notifications",lambda x:int((x>0).sum()))).reset_index()
    controls["window_fpr"]=controls.false_windows/controls.negative_windows
    tests=[];rng=np.random.default_rng(14000)
    for arm in ARMS:
        p=m[m.arm==arm].pivot(index="seed",columns="model",values="recall")
        baseline="strict_local" if arm=="legacy" else "corrected_local"
        for n in p.columns:
            if n==baseline:continue
            v=(p[n]-p[baseline]).to_numpy();boot=rng.choice(v,size=(20000,len(v)),replace=True).mean(1)
            null=np.array([np.mean(v*np.array(q)) for q in itertools.product((-1,1),repeat=len(v))])
            tests.append(dict(arm=arm,a=n,b=baseline,delta=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),
                p_raw=np.mean(np.abs(null)>=abs(v.mean())-1e-12)))
        subset=sorted([x for x in tests if x["arm"]==arm],key=lambda x:x["p_raw"]);prev=0.
        for i,x in enumerate(subset):
            prev=max(prev,min(1.,(len(subset)-i)*x["p_raw"]));x["p_holm"]=prev
    tables={"summary":summary,"attack_summary":attack,"full_factorial":cells,
        "rate_summary":group(["hidden_rate"]),"role_summary":group(["ntype"]),"controls":controls,
        "paired":pd.DataFrame(tests),"all_metrics":m,"episodes":e,"control_events":c}
    for n,t in tables.items():t.to_csv(out/(n+".csv"),index=False)
    for arm in ARMS:
        print("ARM",arm,flush=True)
        print(summary[summary.arm==arm].sort_values("recall",ascending=False).to_string(index=False),flush=True)
        print("ATTACK MATRIX (pooled recall)",flush=True)
        print(attack[attack.arm==arm].pivot(index="model",columns=["sophistication","attack"],values="recall").to_string(),flush=True)
    print("Paired comparisons to arm local baseline, Holm within arm",flush=True)
    print(pd.DataFrame(tests).to_string(index=False),flush=True)
    print("External events are negative controls; overlapping control intervals may repeat observations.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--arm",choices=ARMS);p.add_argument("--models",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="test":evaluate(a.seed,a.arm,load_all(a.models),a.output)
    else:aggregate(a.input,a.output)
