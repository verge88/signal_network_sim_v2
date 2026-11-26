"""Corrected role-consistent SS7 simulation and explicit telemetry assumptions.

Legacy simulator remains unchanged. This arm changes the hidden workload and
mask mechanism, so legacy/new recall is not a causal model comparison.
Counts are offered MSUs, not delivered MSUs; monitoring is an optional,
independent simulated sensor on domestic side of external interconnect.
No true state, hidden ledger, attack rate or label is passed to inference.
"""
from __future__ import annotations
import argparse,json,itertools
from typing import Dict,List,Tuple
from collections import defaultdict
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7.fix import Episode,NodeState,Service,MSG_TYPES,MSG_BYTES,SERVICE_TEMPLATE,INTERVALS_PER_DAY
from ss7.report_feature_specialist import augmented,add_history,DIRECTION,PEER,generate as legacy_generate
from ss7.report_only_invariant_study import NEW,columns
from ss7.weak_signal_confirmation import fit_model,weak_predict
from ss7.incident_calibration_study import elapsed,triple,metrics,RAW_BUDGET,INCIDENT_BUDGET,WARMUP
from ss7.cusum_cascade_development import joint_cut
from ss7.episode_ranking_development import collect

TRAIN=tuple(range(13001,13007))
TEST=tuple(range(13101,13109))
MODES=("local","report","typed","boundary","expanded","role_expanded")
EXTRA=("typed_mean","typed_abs","typed_max","typed_support","matrix_in_abs","matrix_out_abs",
       "matrix_in_max","matrix_out_max","matrix_support","boundary_mean","boundary_abs","boundary_max","boundary_support")
TYPED=EXTRA[:9];BOUNDARY=EXTRA[9:]
FRACTION=.5

def add_record(st,typed,mat,path,i,service,cnt,closed):
    tmpl=F.SERVICE_TEMPLATE[service];per=sum(tmpl.values())
    prev=path[i-1] if i>0 else None;nxt=path[i+1] if i+1<len(path) else None
    n=path[i];ki=tuple(sorted((prev,n))) if prev is not None else None
    ko=tuple(sorted((n,nxt))) if nxt is not None else None
    for m,c in tmpl.items():
        j=F.MT_IDX[m];v=cnt*c
        st.mtype[j]+=v*((ki is not None)+(ko is not None))
        if ki is not None:typed[(0,ki,j)]+=v
        if ko is not None:typed[(1,ko,j)]+=v
        if ki is not None and ko is not None:mat[(ki,ko,j)]+=v
    if ki is not None:st.link_in[ki]=st.link_in.get(ki,0)+cnt*per
    if ko is not None:st.link_out[ko]=st.link_out.get(ko,0)+cnt*per
    if i==0:
        st.originated+=cnt;st.dest[path[-1]]=st.dest.get(path[-1],0)+cnt
    elif i==len(path)-1:
        st.terminated+=cnt;st.dest[path[0]]=st.dest.get(path[0],0)+cnt
    else:st.transited+=cnt
    if (i==0 or i==len(path)-1) and service in F.TCAP_SERVICES:
        st.dlg_begin+=cnt;st.dlg_end+=closed

def extra_rng(seed,domain,t):
    return np.random.default_rng(np.random.SeedSequence(int(seed),spawn_key=(1000,int(domain),int(t))))

def empty(n,t):
    return F.NodeState(n,t,np.zeros(len(F.MSG_TYPES),np.int64))

def noisy(counts,rng,sigma):
    # Independent aggregation/measurement channel, same noise law for all.
    return {k:int(max(0,round(v*(1+rng.normal(0,sigma))))) for k,v in counts.items() if v>0}

class CorrectedSim(F.Ss7Sim):
    def candidates(self,n):
        if not hasattr(self,"_candidate_cache"):self._candidate_cache={}
        if n in self._candidate_cache:return self._candidate_cache[n]
        typ=self.topo.nodes[n].ntype;out=[]
        for f in self.flows:
            if typ==F.NodeType.SP and f.src!=n:continue
            if typ==F.NodeType.SCP and f.dst!=n:continue
            if typ==F.NodeType.IGW and not (self.topo.nodes[f.src].external or self.topo.nodes[f.dst].external):continue
            for path in self.topo.route(f.src,f.dst).paths:
                if len(path)!=len(set(path)):continue
                if typ in (F.NodeType.STP,F.NodeType.IGW) and n not in path[1:-1]:continue
                if n not in path:continue
                out.append((f,path))
        self._candidate_cache[n]=out
        return out

    def step_true(self, t: int, episodes: List[Episode]
                  ) -> Tuple[Dict[int, NodeState], Dict[int, Dict], Dict]:
        cfg, topo = self.cfg, self.topo
        rng = self.book.child("traffic", t)
        dt = cfg.interval_s
        phase = 2.0 * np.pi * (t % INTERVALS_PER_DAY) / INTERVALS_PER_DAY
        diurnal = 1.0 + self.diurnal_amp * np.sin(phase - 1.2)

        # AR(1) по узлам — общий механизм для всех, без веток
        for n in topo.nodes:
            self._ar_x[n] = (self._ar_phi[n] * self._ar_x[n]
                             + rng.normal(0.0, self._ar_sig[n]))
        node_mult = {n: float(np.exp(self._ar_x[n])) for n in topo.nodes}

        # нормальные события (негативные контроли)
        ev_mult = {n: 1.0 for n in topo.nodes}
        failed_links = set()
        for ep in episodes:
            if not (ep.t0 <= t <= ep.t1) or ep.kind == "compromise":
                continue
            if ep.kind in ("traffic_spike", "traffic_dip", "scale_in", "campaign"):
                for n in (topo.nodes if ep.nid is None else [ep.nid]):
                    ev_mult[n] *= ep.mult
            elif ep.kind == "link_failure" and ep.nid is not None:
                for l in topo.links_of(ep.nid)[:1]:
                    failed_links.add(l)

        offered: Dict[Tuple[int, int], float] = {}
        pending: List[Tuple] = []

        # --- первый проход: раскладка транзакций по маршрутам
        for f in self.flows:
            lam = f.rate * dt * diurnal * node_mult[f.src] * ev_mult[f.src]
            # внешние атаки на границе (SRI-SM flood / ATI tracking / IRSF)
            for ep in episodes:
                if (ep.kind == "ext_attack" and ep.t0 <= t <= ep.t1
                        and f.src == ep.nid and f.service in
                        (Service.ROAM_SRI, Service.SMS_MT, Service.LOCATION)):
                    lam *= ep.mult
            n_txn = int(rng.poisson(max(lam, 0.0)))
            if n_txn <= 0:
                continue
            r = topo.route(f.src, f.dst)
            if not r.paths:
                continue
            route_shares = self._effective_route_shares(
                r, f.src, f.dst, t, episodes
            )
            split = rng.multinomial(n_txn, route_shares)
            for path, cnt in zip(r.paths, split):
                if cnt <= 0 or any((min(u, v), max(u, v)) in failed_links
                                   for u, v in zip(path[:-1], path[1:])):
                    continue
                pending.append((f, path, int(cnt), None))
                tmpl = SERVICE_TEMPLATE[f.service]
                bits = 8.0 * sum(MSG_BYTES[m] * c for m, c in tmpl.items())
                for u, v in zip(path[:-1], path[1:]):
                    k = (min(u, v), max(u, v))
                    offered[k] = offered.get(k, 0.0) + cnt * bits / dt


        # Hidden events use a separate RNG, explicit provenance and legal
        # shortest paths. No artificial detours through endpoint nodes.
        arng=extra_rng(self.cfg.seed,1,t)
        active={e.nid:e for e in episodes if e.kind=="compromise" and e.t0<=t<=e.t1}
        for n,e in active.items():
            h=int(arng.poisson(max(e.hidden_rate*dt,0.)))
            if h<=0:continue
            opts=self.candidates(n)
            if not opts:continue
            f,path=opts[int(arng.integers(len(opts)))]
            if any(tuple(sorted((u,v))) in failed_links for u,v in zip(path[:-1],path[1:])):continue
            pending.append((f,path,h,n))
            bits=8.*sum(F.MSG_BYTES[m]*c for m,c in F.SERVICE_TEMPLATE[f.service].items())
            for u,v in zip(path[:-1],path[1:]):
                k=tuple(sorted((u,v)));offered[k]=offered.get(k,0.)+h*bits/dt
        lstate={k:F.link_state(offered.get(k,0.),ls,self.mean_msg_bits) for k,ls in topo.linksets.items()}
        records=defaultdict(list)
        for f,path,cnt,owner in pending:
            hops=[tuple(sorted((u,v))) for u,v in zip(path[:-1],path[1:])]
            p=float(np.prod([1.-lstate[k][2] for k in hops]))**2
            p=1.-(1.-p)**(1+self.max_retry)
            crng=rng if owner is None else arng
            closed=int(crng.binomial(cnt,np.clip(p,0.,1.)))
            for i,n in enumerate(path):records[n].append((path,i,f.service,cnt,closed,owner))
        true={};claimed={};typed={};mat={};boundary={};audit={}
        for n in topo.nodes:
            st=empty(n,t);cl=empty(n,t);tt=defaultdict(int);tm=defaultdict(int)
            ct=defaultdict(int);cm=defaultdict(int);hidden=0;removed=0;observable=0
            e=active.get(n)
            for path,i,service,cnt,closed,owner in records[n]:
                add_record(st,tt,tm,path,i,service,cnt,closed)
                remain=cnt;cc=closed
                if owner==n:
                    per=sum(F.SERVICE_TEMPLATE[service].values())
                    hidden+=cnt*per*((i>0)+(i<len(path)-1))
                    # Corrected threat arm: coherent whole-transaction
                    # suppression, not legacy F1/F2/F3/F5 counter editing.
                    k=0. if e.sophistication==F.Sophistication.NAIVE else e.budget
                    if e.sophistication==F.Sophistication.STATISTICAL:k*=.6
                    if e.family=="F4_temporal":k*=float(arng.uniform(.4,1.))
                    cut=int(round(cnt*np.clip(k,0.,1.)))
                    remain=cnt-cut;cc=min(remain,int(round(closed*remain/max(cnt,1))))
                    removed+=cut*per*((i>0)+(i<len(path)-1))
                    observable+=cut*per*sum(not topo.nodes[p].external for p in
                        ([path[i-1]] if i>0 else [])+([path[i+1]] if i<len(path)-1 else []))
                if remain:add_record(cl,ct,cm,path,i,service,remain,cc)
            st.check_identities();cl.check_identities()
            true[n]=st;claimed[n]=cl;typed[n]=ct;mat[n]=cm
            # Explicit additional sensor: true offered counters sampled on
            # the domestic side of an external link, independently of node report.
            # Only sensor measurements, never this latent dictionary, reach features.
            boundary[n]={k:v for k,v in tt.items() if topo.nodes[topo.peer(k[1],n)].external}
            audit[n]=(hidden,removed,observable,max(st.total()-hidden,0))
        return true,claimed,typed,mat,boundary,audit,lstate

def discrepancy(x,y):
    return 2.*(x-y)/max(x+y,1.)

def stats(vals):
    if not vals:return (np.nan,np.nan,np.nan,0.)
    a=np.asarray(vals);return (float(a.mean()),float(np.abs(a).mean()),float(np.abs(a).max()),float(len(a)))

def telemetry(reps,typed,mat,sensors,topo):
    out={}
    for n,r in reps.items():
        tc=typed[n];mc=mat[n];cross=[];boundary=[];mi=[];mo=[]
        for k in topo.links_of(n):
            p=topo.peer(k,n)
            for direction in (0,1):
                for j in range(len(F.MSG_TYPES)):
                    x=tc.get((direction,k,j),0)
                    if p in typed:
                        y=typed[p].get((1-direction,k,j),0)
                        if x+y:cross.append(discrepancy(x,y))
                    if (direction,k,j) in sensors.get(n,{}) or (topo.nodes[p].external and n in sensors):
                        y=sensors[n].get((direction,k,j),0)
                        if x+y:boundary.append(discrepancy(x,y))
        # Compare relay matrix marginals to opposite endpoint's independently
        # reported typed counts, not to the relay's own self-consistent totals.
        if topo.nodes[n].ntype in (F.NodeType.STP,F.NodeType.IGW):
            marg=defaultdict(int)
            for (ki,ko,j),v in mc.items():
                marg[(0,ki,j)]+=v;marg[(1,ko,j)]+=v
            for k in topo.links_of(n):
                p=topo.peer(k,n)
                if p not in typed:continue
                for direction,dest in ((0,mi),(1,mo)):
                    for j in range(len(F.MSG_TYPES)):
                        x=marg.get((direction,k,j),0);y=typed[p].get((1-direction,k,j),0)
                        if x+y:dest.append(discrepancy(x,y))
        _,ia,im,ic=stats(mi);_,oa,om,oc=stats(mo)
        out[n]=dict(zip(TYPED[:4],stats(cross)))
        out[n].update(dict(matrix_in_abs=ia,matrix_out_abs=oa,matrix_in_max=im,matrix_out_max=om,matrix_support=ic+oc))
        out[n].update(dict(zip(BOUNDARY,stats(boundary))))
    return out

def generate(seed,days,comp,normal,external,template=None,force_hidden=None):
    cfg=F.SimConfig(seed=seed,days=days);book=F.RngBook(seed)
    sim=CorrectedSim(cfg,book,topo=template.topo if template else None,template=template)
    eps=F.make_episodes(sim,cfg,book["episode"],n_compromise=comp,n_normal=normal,n_ext=external,force_hidden=force_hidden)
    ids={id(e):i for i,e in enumerate(eps)};rows=[]
    for t in range(cfg.n_intervals):
        truth,claimed,typed,mat,boundary,audit,lstate=sim.step_true(t,eps)
        rng=book.child("measure",t);trng=extra_rng(seed,2,t);brng=extra_rng(seed,3,t)
        active={e.nid:e for e in eps if e.kind=="compromise" and e.t0<=t<=e.t1}
        reps={};tc={};mc={};sensors={}
        for n,st in claimed.items():
            if sim.topo.nodes[n].external:continue
            e=active.get(n);keys=sim.topo.links_of(n)
            rho=float(np.mean([lstate[k][0] for k in keys])) if keys else 0.
            cap=float(np.mean([sim.topo.linksets[k].capacity_bps for k in keys])) if keys else 64000.
            reps[n]=F.render_report(st,rel_sigma=sim.rel_sigma,rho_path=rho,mean_msg_bits=sim.mean_msg_bits,
                lsx_cap_bps=cap,custody=cfg.key_custody,compromised=e is not None,t_compromise=e.t0 if e else None,
                rotation=sim.rotation,background=sim.background,stale=False,rng=rng)
            tc[n]=noisy(typed[n],trng,sim.rel_sigma);mc[n]=noisy(mat[n],trng,sim.rel_sigma)
            if sim.topo.nodes[n].ntype==F.NodeType.IGW:sensors[n]=noisy(boundary[n],brng,sim.rel_sigma)
        inv=augmented(reps,sim.topo,lstate);extra=telemetry(reps,tc,mc,sensors,sim.topo)
        for n,r in reps.items():
            node=sim.topo.nodes[n]
            if node.role==F.NodeRole.MASTER:continue
            e=active.get(n);h,cut,obs,legit=audit[n]
            rows.append(dict(nid=n,t=t,ntype=node.ntype.value,zone=node.zone,**r.feats,**inv[n],**extra[n],
                label=int(e is not None),episode_id=ids[id(e)] if e else -1,hidden_msgs=h,
                audit_removed=cut,audit_visible=obs,audit_legit=legit,audit_sigma=sim.rel_sigma,
                integrity_fail=float(not r.integrity_ok),rtt_ms=r.rtt_ms))
    d=pd.DataFrame(rows)
    d=F.temporal_block(d,F.OWN+F.INV+NEW+list(EXTRA),cfg.feat_window)
    d=F.zone_deviation(d,F.OWN+F.INV+NEW+list(EXTRA),{n:v.zone for n,v in sim.topo.nodes.items()})
    d=add_history(d);d=d[d.t>=WARMUP].reset_index(drop=True)
    return d,sim,eps

def feature_names(d,mode):
    base=columns(d,F.OWN)
    if mode!="local":base+=columns(d,NEW)+[c+s for c in DIRECTION+PEER for s in ("","__delta1","__delta5")]
    if mode in ("typed","expanded","role_expanded"):base+=columns(d,TYPED)
    if mode in ("boundary","expanded","role_expanded"):base+=columns(d,BOUNDARY)
    return base

def fit(d,mode):
    cols=feature_names(d,mode);y=d.label.to_numpy();ep=d.uid
    counts=ep[y==1].value_counts();w=np.ones(len(d));p=y==1
    w[p]=1./ep[p].map(counts).to_numpy()
    for k in (0,1):
        if not (y==k).any():raise ValueError("missing class")
        w[y==k]*=len(y)/(2*w[y==k].sum())
    return fit_model(d[cols].to_numpy(float),y,w,cols)

def train(out):
    parts=[]
    for seed in TRAIN:
        d,_,_=generate(seed,1.5,64,96,16)
        d["uid"]=str(seed)+":"+d.episode_id.astype(str)
        pos=np.flatnonzero(d.label.to_numpy()==1);neg=np.flatnonzero(d.label.to_numpy()==0)
        rng=np.random.default_rng(seed)
        ix=np.sort(np.r_[pos,rng.choice(neg,min(50000,len(neg)),replace=False)])
        parts.append(d.iloc[ix]);print("train",seed,len(pos),len(ix),flush=True)
    d=pd.concat(parts,ignore_index=True)
    models={n:fit(d,n) for n in MODES if n!="role_expanded"}
    models["role_expanded"]={r:fit(d[d.ntype==r],"expanded") for r in ("STP","IGW") if
        d[d.ntype==r].label.nunique()==2 and d[(d.ntype==r)&(d.label==1)].uid.nunique()>=10}
    joblib.dump(models,out/"models.joblib",compress=3)
    print("ROLE MODELS",list(models["role_expanded"]),flush=True)

def predict(d,models,n):
    if n!="role_expanded":
        m=models[n];return weak_predict(m,d[m["names"]].to_numpy(float))
    # Global expanded fallback for SP/SCP and insufficient role support.
    s=predict(d,models,"expanded")
    for role,m in models[n].items():
        ix=d.ntype.to_numpy()==role
        if ix.any():s[ix]=weak_predict(m,d.loc[ix,m["names"]].to_numpy(float))
    return s

def test(seed,models,out):
    if seed not in TEST:raise ValueError("unregistered test")
    d,sim,eps=generate(seed,1.5,32,64,10)
    eligible={}
    for i,e in enumerate(eps):
        if e.kind!="compromise" or e.t0<WARMUP:continue
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q):eligible[i]=e
    if len(eligible)<12:raise ValueError("insufficient complete episodes")
    refs=(generate(seed+10000,3.,0,128,20,sim)[0],
          generate(seed+20000,3.,0,192,32,sim)[0],
          generate(seed+40000,3.,64,128,20,sim)[0])
    rows=[];events=[]
    for n in MODES:
        s=[predict(x,models,n) for x in (*refs,d)]
        tau=joint_cut(refs,s[:3],RAW_BUDGET*FRACTION,INCIDENT_BUDGET*FRACTION)
        u=s[3];z=triple(d,u);m,det=metrics(d,u,z,tau,eligible)
        rows.append(dict(seed=seed,model=n,**m))
        for i,e in eligible.items():
            q=d[(d.episode_id==i)&(d.nid==e.nid)]
            # Latent diagnostic, explicitly excluded from all training/prediction.
            snr=q.audit_visible/(np.maximum(q.audit_legit,1)*q.audit_sigma)
            events.append(dict(seed=seed,model=n,episode=i,detected=int(i in det),ntype=sim.topo.nodes[e.nid].ntype.value,
                hidden_rate=e.hidden_rate,mask_family=e.family,duration_windows=len(q),delay_seconds=det.get(i),
                hidden_mean=q.hidden_msgs.mean(),removed_mean=q.audit_removed.mean(),visible_mean=q.audit_visible.mean(),
                legit_mean=q.audit_legit.mean(),snr_proxy=float(snr.median()),active_fraction=float((q.hidden_msgs>0).mean())))
        print(seed,n,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)

def legacy_audit(source,out):
    old=pd.read_csv(source/"episodes.csv")
    old=old[old.model=="weak"].copy()
    expected=tuple(range(11201,11209))
    if set(old.seed)!=set(expected) or old.duplicated(["seed","episode"]).any():raise ValueError("unexpected legacy support")
    rows=[]
    for seed in expected:
        d,sim,eps=legacy_generate(seed,1.5,32,64,10)
        for e in old[old.seed==seed].itertuples():
            ep=eps[int(e.episode)]
            q=d[(d.episode_id==e.episode)&(d.nid==ep.nid)]
            if len(q)!=ep.t1-ep.t0+1 or sim.topo.nodes[ep.nid].ntype.value!=e.ntype:
                raise ValueError("legacy replay mismatch")
            # Original hidden ledger is itself under audit. Its ratio to
            # reported total is a proxy, not measured SNR or true deficit.
            proxy=q.hidden_msgs/(np.maximum(q.total_messages,1)*sim.rel_sigma)
            rows.append(dict(seed=seed,episode=e.episode,ntype=e.ntype,detected=e.detected,
                hidden_rate=e.hidden_rate,mask_family=e.mask_family,
                hidden_mean=q.hidden_msgs.mean(),reported_mean=q.total_messages.mean(),
                sigma=sim.rel_sigma,snr_proxy=float(proxy.median()),active_fraction=float((q.hidden_msgs>0).mean())))
        print("legacy replay",seed,flush=True)
    t=pd.DataFrame(rows);t.to_csv(out/"legacy_snr_episodes.csv",index=False)
    summary=t.groupby(["ntype","detected"]).agg(episodes=("episode","size"),
        snr=("snr_proxy","median"),hidden=("hidden_mean","median"),
        reported=("reported_mean","median"),active=("active_fraction","median")).reset_index()
    summary.to_csv(out/"legacy_snr_summary.csv",index=False);print(summary.to_string(index=False))
    print("Descriptive replay; old hidden counter and reported denominator do not measure physical SNR.")

def aggregate(root,out):
    m,e,s=collect(root,TEST,MODES)
    strata=e.groupby(["model","ntype"]).agg(episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    diag=e[e.model=="report"].groupby(["ntype","detected"]).agg(episodes=("episode","size"),
        snr=("snr_proxy","median"),hidden=("hidden_mean","median"),legit=("legit_mean","median"),
        visible=("visible_mean","median"),active=("active_fraction","median")).reset_index()
    p=m.pivot(index="seed",columns="model",values="recall")
    rng=np.random.default_rng(13100);contrasts=[]
    for a,b in (("typed","report"),("boundary","report"),("expanded","report"),("role_expanded","expanded"),("report","local")):
        v=(p[a]-p[b]).to_numpy();boot=rng.choice(v,size=(20000,len(v)),replace=True).mean(1)
        null=np.array([np.mean(v*np.array(q)) for q in itertools.product((-1,1),repeat=len(v))])
        contrasts.append(dict(a=a,b=b,delta=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),
            p_raw=np.mean(np.abs(null)>=abs(v.mean())-1e-12)))
    prev=0.
    for i,x in enumerate(sorted(contrasts,key=lambda z:z["p_raw"])):
        prev=max(prev,min(1.,(len(contrasts)-i)*x["p_raw"]));x["p_holm"]=prev
    for name,q in (("summary",s),("all_metrics",m),("episodes",e),("roles",strata),("snr_diagnostic",diag),("paired",pd.DataFrame(contrasts))):
        q.to_csv(out/(name+".csv"),index=False);print(name,q.to_string(index=False) if name not in ("all_metrics","episodes") else q.shape,flush=True)
    print("Corrected synthetic workload, additional telemetry is conditional; no direct legacy performance comparison.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","test","aggregate","legacy_audit"])
    p.add_argument("--seed",type=int);p.add_argument("--models",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="train":train(a.output)
    elif a.mode=="test":test(a.seed,joblib.load(a.models),a.output)
    elif a.mode=="legacy_audit":legacy_audit(a.input,a.output)
    else:aggregate(a.input,a.output)
