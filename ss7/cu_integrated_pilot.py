"""Integrated CU + corrected-threat pilot (synthetic data only).
The canonical corrected transaction attribution is preserved from commit
ca1cc9390e91bddf50db28ba0fa94ec47eea0a61; only the flow input and link
load calculation are extended. No network I/O or protocol message generation.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from ss7 import fix as F
from ss7 import observability_audit_development as O
from ss7 import dual_channel_behavior_development as D
from ss7.report_only_invariant_study import report_only, NEW
from ss7.incident_calibration_study import elapsed, triple, metrics, RAW_BUDGET, INCIDENT_BUDGET, WARMUP
from ss7.cusum_cascade_development import joint_cut

Episode=F.Episode
NodeState=F.NodeState
Service=F.Service
MSG_TYPES=F.MSG_TYPES
MSG_BYTES=F.MSG_BYTES
SERVICE_TEMPLATE=F.SERVICE_TEMPLATE
INTERVALS_PER_DAY=F.INTERVALS_PER_DAY
add_record=O.add_record
empty=O.empty
extra_rng=O.extra_rng
TRAIN=(16001,16002)
CAL=(16101,16102)
TEST=(16201,16202)
DAYS=.5
KINDS=("sms_interception","location_tracking","signalling_dos","irsf","compromise")
CU=("cu_delivered","cu_rtt_delay","cu_req_delay","cu_resp_delay","cu_combined_loss_prob",
    "integrity_check","consecutive_cu_losses","cu_processing_delay","cu_response_time_jitter",
    "cu_staleness","cu_consistency_error")
MAP=tuple(m for m in F.MSG_TYPES if F.MT_IDX[m]>=5)
SUB=tuple("obs_ratio_"+m for m in MAP)
TEMP=("cv_total","cv_entropy","cv_io","cv_dominance","ar1_total","integrity_fail_rate","cv_dest")
ZONE_BASE=("total_messages","map_share","isup_share","in_out_ratio","integrity_fail_rate","cv_total","cu_delivered")
ZONE=tuple("zone_dev_"+c for c in ZONE_BASE)+("zone_io_ratio_rank",)
BASE=list(CU)+list(F.OWN)+list(SUB)+list(TEMP)+list(ZONE)+["node_type_enc"]
SPECIAL=list(CU)+list(TEMP)+list(ZONE[:3])
assert len(BASE)==50 and len(SPECIAL)==21
MODELS=("local","article_cascade","cu_invariant_cascade","integrated_cascade")
SOURCE_REV="ca1cc9390e91bddf50db28ba0fa94ec47eea0a61"

class CUExchange:
    """One explicit request and conditional response per slave per interval.
    Planned CU demand determines the provisional finite-buffer load; realized
    per-hop attempted bits (including lost packets) determine final link load.
    Interval-aggregated queue approximation, not a discrete MTP2 frame emulator.
    50 B request/base response; extended counters have a modeled wire cost.
    """
    def __init__(self,sim,extended=True):
        self.sim=sim;self.extended=extended;topo=sim.topo
        self.master={}
        for n in topo.slaves:
            nd=topo.nodes[n]
            masters=[m for m in topo.masters if topo.nodes[m].zone==nd.zone] or list(topo.masters)
            self.master[n]=min(masters,key=lambda m:(len(topo.route(m,n).paths[0]),m))
        self.paths={n:[tuple(p) for p in topo.route(m,n).paths] for n,m in self.master.items()}
        self.payload={}
        for n in self.master:
            deg=len(topo.links_of(n))
            # Dense fixed-width count envelope; includes ordinary counters,
            # typed directional counters and relay matrix, no free telemetry.
            self.payload[n]=50+(8*(17+2*deg+len(topo.nodes)+8+2*deg*17+deg*deg*17) if extended else 0)
        rng=extra_rng(sim.cfg.seed,20,0)
        self.prop={k:float(rng.uniform(2,5)) for k in topo.linksets}
        self.tot=defaultdict(float);self.per_link=defaultdict(float)
        self.reserved=defaultdict(float)
    @staticmethod
    def hops(path):
        return [tuple(sorted((a,b))) for a,b in zip(path[:-1],path[1:])]
    def exchange(self,t,offered,failed):
        topo=self.sim.topo;dt=self.sim.cfg.interval_s
        rng=extra_rng(self.sim.cfg.seed,21,t)
        path={};planned=defaultdict(float);actual=defaultdict(float)
        for n,opts in self.paths.items():
            good=[p for p in opts if not any(k in failed for k in self.hops(p))]
            p=(good or opts)[0];path[n]=p
            for k in self.hops(p):planned[k]+=8*(50+self.payload[n])
        provisional={k:F.link_state(offered.get(k,0)+planned[k]/dt,ls,self.sim.mean_msg_bits)
                     for k,ls in topo.linksets.items()}
        def send(p,sz):
            delay=0.
            for k in self.hops(p):
                actual[k]+=8*sz
                rho,q,loss=provisional[k];ls=topo.linksets[k]
                if k in failed:return False,delay
                # M/G/1 queue + propagation + actual packet serialization.
                delay+=max(0.,q-self.sim.mean_msg_bits/ls.capacity_bps*1000)
                delay+=self.prop[k]+8*sz/ls.capacity_bps*1000
                delay+=float(rng.uniform(0,.8))
                if rng.random()<1-(1-.001)*(1-loss):return False,delay
            return delay<dt*1000,delay
        polls={}
        for n,p in path.items():
            rq,req=send(p,50)
            proc=float(rng.uniform(.8,1.2)) # same law for all nodes
            rs,resp=send(p[::-1],self.payload[n]) if rq else (False,np.nan)
            ok=bool(rq and rs and req+resp+proc<dt*1000)
            polls[n]=(ok,req if rq else np.nan,resp if rs else np.nan,proc if rq else np.nan)
            self.tot["polls"]+=1;self.tot["requests_delivered"]+=rq;self.tot["responses_delivered"]+=ok
        for k,b in actual.items():
            assert b<=planned[k]
            self.per_link[k]+=b
        for k,b in planned.items():self.reserved[k]+=b
        self.tot["wire_bytes"]+=sum(actual.values())/8
        self.tot["reserved_bytes"]+=sum(planned.values())/8
        return {k:F.link_state(offered.get(k,0)+actual[k]/dt,ls,self.sim.mean_msg_bits)
                for k,ls in topo.linksets.items()},polls
    def report(self):
        caps={str(k):self.per_link[k]/(ls.capacity_bps*self.sim.cfg.n_intervals*self.sim.cfg.interval_s)
              for k,ls in self.sim.topo.linksets.items()}
        return dict(self.tot,max_mean_link_fraction=max(caps.values(),default=0),
                    link_fraction=caps,extended=self.extended,
                    response_bytes_min=min(self.payload.values()),response_bytes_max=max(self.payload.values()))

class IntegratedSim(O.CorrectedSim):
    def __init__(self,cfg,book,extended=True):
        super().__init__(cfg,book)
        self.cu=CUExchange(self,extended)
    def flows_for(self,t,episodes):
        """Abstract synthetic service-mix perturbations, not protocol exploits.
        Keep normal routing and background process; generate no live packets."""
        active={e.nid:e for e in episodes if e.kind in KINDS[:-1] and e.t0<=t<=e.t1}
        for f in self.flows:
            e=active.get(f.src)
            if e is None:yield f;continue
            # Reweight aggregate service volumes within the simulated traffic.
            # Total volume and class mix are not guaranteed article-identical.
            k=e.kind
            if k=="signalling_dos":mult=e.mult
            elif k=="sms_interception":mult=e.mult if f.service==Service.SMS_MT else 1.
            elif k=="irsf":mult=e.mult if f.service==Service.VOICE_INT else 1.
            else:mult=1.
            yield F.Flow(f.src,f.dst,f.service,f.rate*mult)
        for n,e in active.items():
            if e.kind=="location_tracking":
                # Aggregate LOCATION category absent in normal flow generator.
                # Add a small synthetic rate to a pre-existing endpoint pair.
                dst=self.topo.hlr[0]
                yield F.Flow(n,dst,Service.LOCATION,e.mult)

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
        for f in self.flows_for(t,episodes):
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
        lstate,self.polls=self.cu.exchange(t,offered,failed_links)
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

def schedule(sim,rng):
    # Every test topology has all classes; role/sophistication balanced for compromise.
    cfg=sim.cfg
    eps=F.make_episodes(sim,cfg,rng,n_compromise=0,n_normal=20,n_ext=0)
    occ=defaultdict(list)
    def place(nodes,kind,length,**kwargs):
        for _ in range(5000):
            n=int(rng.choice(nodes));a=int(rng.integers(WARMUP+5,cfg.n_intervals-length-5));b=a+length-1
            if all(b+3<x or a>y+3 for x,y in occ[n]):
                occ[n].append((a,b));eps.append(Episode(kind=kind,nid=n,t0=a,t1=b,**kwargs));return
        raise RuntimeError("no nonoverlapping slot")
    sp=[n for n in sim.topo.slaves if sim.topo.nodes[n].ntype==F.NodeType.SP]
    for k in KINDS[:-1]:
        for _ in range(4):
            mult={"sms_interception":(2.,2.5),"location_tracking":(1.5,1.8),
                  "signalling_dos":(30.,40.),"irsf":(3.,4.)}[k]
            place(sp,k,int(rng.integers(30,61)),mult=float(rng.uniform(*mult)))
    for role in F.NodeType:
        nodes=[n for n in sim.topo.slaves if sim.topo.nodes[n].ntype==role]
        for soph in F.Sophistication:
            place(nodes,"compromise",int(rng.integers(60,91)),
                  sophistication=soph,hidden_rate=float(rng.choice((.05,.15,.4,1.,2.5))),
                  family=str(rng.choice(("F1_uniform","F4_temporal"))),
                  budget=float(rng.uniform(.7,1.)),params=rng.random(2))
    return eps

def features(d):
    d=d.sort_values(["nid","t"]).reset_index(drop=True)
    for name in TEMP:d[name]=np.nan
    for _,g in d.groupby("nid",sort=False):
        for base,name in zip(("total_messages","mtype_entropy","in_out_ratio","dest_top1_ratio","n_unique_dest"),
                            ("cv_total","cv_entropy","cv_io","cv_dominance","cv_dest")):
            rr=g[base].rolling(30,min_periods=5)
            d.loc[g.index,name]=(rr.std()/(rr.mean().abs()+1e-9)).to_numpy()
        d.loc[g.index,"ar1_total"]=g.total_messages.rolling(30,min_periods=5).corr(g.total_messages.shift(1)).to_numpy()
        d.loc[g.index,"integrity_fail_rate"]=(1-g.integrity_check).rolling(30,min_periods=5).mean().to_numpy()
    for c in ZONE_BASE:
        # Contemporaneous zone peer median is observable at the coordinator.
        d["zone_dev_"+c]=d[c]-d.groupby(["zone","t"])[c].transform("median")
    d["zone_io_ratio_rank"]=d.groupby(["zone","t"]).in_out_ratio.rank(pct=True)
    inv=list(NEW)+list(O.EXTRA)
    d=F.temporal_block(d,inv,30)
    d=F.zone_deviation(d,inv,{int(n):int(g.zone.iloc[0]) for n,g in d.groupby("nid")})
    d=D.history(d)
    return d.replace([np.inf,-np.inf],np.nan)

def generate(seed,days=DAYS,extended=True):
    cfg=F.SimConfig(seed=seed,days=days);book=F.RngBook(seed)
    sim=IntegratedSim(cfg,book,extended);eps=schedule(sim,book["episode"])
    ids={id(e):i for i,e in enumerate(eps)}
    last={};losses=defaultdict(int);age=defaultdict(int);history=defaultdict(lambda:deque(maxlen=30))
    jitter=defaultdict(lambda:deque(maxlen=10))
    rows=[]
    for t in range(cfg.n_intervals):
        truth,claimed,typed,mat,boundary,audit,lstate=sim.step_true(t,eps)
        rng=book.child("measure",t);trng=extra_rng(seed,2,t);brng=extra_rng(seed,3,t)
        active={e.nid:e for e in eps if e.kind in KINDS and e.t0<=t<=e.t1}
        fresh={};tc={};mc={};sensors={};obs={}
        for n,st in claimed.items():
            node=sim.topo.nodes[n]
            if node.external:continue
            e=active.get(n);comp=e is not None and e.kind=="compromise"
            # Shared honest/adaptive report path, no compromise-conditioned delay/noise.
            # The actual CU timing is supplied by CUExchange, not renderer RTT.
            keys=sim.topo.links_of(n);rho=float(np.mean([lstate[k][0] for k in keys]))
            cap=float(np.mean([sim.topo.linksets[k].capacity_bps for k in keys]))
            rep=F.render_report(st,rel_sigma=sim.rel_sigma,rho_path=rho,mean_msg_bits=sim.mean_msg_bits,
                 lsx_cap_bps=cap,custody=cfg.key_custody,compromised=comp,t_compromise=e.t0 if comp else None,
                 rotation=sim.rotation,background=sim.background,stale=False,rng=rng)
            a=O.noisy(typed[n],trng,sim.rel_sigma);b=O.noisy(mat[n],trng,sim.rel_sigma)
            if node.ntype==F.NodeType.IGW:sensors[n]=O.noisy(boundary[n],brng,sim.rel_sigma)
            if node.role==F.NodeRole.MASTER:
                fresh[n]=rep;tc[n]=a;mc[n]=b;continue
            delivered,req,resp,proc=sim.polls[n]
            valid=delivered and rep.integrity_ok
            hist=history[n]
            loss_est=float(np.mean(hist)) if hist else np.nan # observable past losses
            hist.append(int(not delivered))
            losses[n]=0 if delivered else losses[n]+1
            age[n]=0 if valid else age[n]+1
            if delivered:jitter[n].append(resp+proc)
            if valid:
                last[n]=rep;fresh[n]=rep;tc[n]=a;mc[n]=b
            r=last.get(n)
            vals={c:np.nan for c in F.OWN+list(SUB)}
            if r is not None:
                vals.update(r.feats);st0=r.state;ms=sum(st0.mtype[5:])
                vals.update({"obs_ratio_"+m:float(st0.mtype[F.MT_IDX[m]])/max(ms,1) for m in MAP})
            err=(abs(r.state.total()-sum(r.state.link_in.values())-sum(r.state.link_out.values()))
                 /max(r.state.total(),1) if r is not None else np.nan)
            vals.update(dict(cu_delivered=float(delivered),cu_rtt_delay=req+resp+proc if delivered else np.nan,
                cu_req_delay=req if delivered else np.nan,cu_resp_delay=resp if delivered else np.nan,
                cu_combined_loss_prob=loss_est,integrity_check=float(rep.integrity_ok) if delivered else np.nan,
                consecutive_cu_losses=losses[n],cu_processing_delay=proc if delivered else np.nan,
                cu_response_time_jitter=float(np.std(jitter[n])) if len(jitter[n])>=3 else np.nan,
                cu_staleness=age[n],cu_consistency_error=err))
            obs[n]=vals
        # Compare only current authenticated observations. Missing report != zero.
        inv=report_only(fresh,sim.topo);extra=O.telemetry(fresh,tc,mc,sensors,sim.topo)
        behavior=D.behavior_snapshot(fresh,tc,mc,sensors,sim.topo)
        for n,vals in obs.items():
            nd=sim.topo.nodes[n];e=active.get(n)
            row={c:np.nan for c in list(NEW)+list(O.EXTRA)+list(D.BEHAVIOR)}
            row.update(inv.get(n,{}));row.update(extra.get(n,{}));row.update(behavior.get(n,{}))
            # Independent boundary sensor remains observable even when CU lost.
            if n in sensors and n not in fresh:
                row.update({"beh_border_"+k:v for k,v in zip(D.PROFILE,D.profile(sensors[n]))})
            rows.append(dict(nid=n,t=t,zone=nd.zone,ntype=nd.ntype.value,
                  node_type_enc=list(F.NodeType).index(nd.ntype),label=int(e is not None),
                  episode_id=ids[id(e)] if e else -1,attack=e.kind if e else "normal",
                  sophistication=e.sophistication.value if e and e.kind=="compromise" else "",
                  **vals,**row))
    d=features(pd.DataFrame(rows));d=d[d.t>=WARMUP].reset_index(drop=True)
    assert not d.duplicated(["nid","t"]).any()
    eligible={i:e for i,e in enumerate(eps) if e.kind in KINDS}
    for i,e in eligible.items():
        assert (d.episode_id==i).sum()==e.t1-e.t0+1
    assert set(d.attack.unique())==set(KINDS)|{"normal"}
    return d,eligible,sim.cu.report()

def cols(d,mode):
    if mode=="local":return list(F.OWN)+list(SUB)+[x for x in TEMP if x!="integrity_fail_rate"]+["node_type_enc"]
    if mode=="base":return BASE
    if mode=="special":return SPECIAL
    if mode=="invariant":return BASE+[c for c in d.columns if any(c==b or c.startswith(b+"__") for b in list(NEW)+list(O.EXTRA))]
    if mode=="behavior":return list(F.OWN)+list(D.BEHAVIOR)+[c for c in d if c.startswith("hist_")]+list(CU)
    raise ValueError(mode)

def matrix(d,names):
    forbidden={"label","episode_id","attack","sophistication","nid","t","zone","ntype","uid"}
    assert not forbidden.intersection(names) and not any(c.startswith(("audit_","hidden_","true_","gt_")) for c in names)
    return d[names].to_numpy(float)

def fit(d,mode,target,kind):
    names=cols(d,mode);y=(d.attack=="compromise").to_numpy(int) if target=="comp" else d.label.to_numpy(int)
    w=np.ones(len(d));p=y==1;counts=d.loc[p,"uid"].value_counts()
    w[p]=1/d.loc[p,"uid"].map(counts).to_numpy()
    for k in (0,1):w[y==k]*=len(y)/(2*w[y==k].sum())
    if kind=="rf":est=RandomForestClassifier(n_estimators=100,min_samples_leaf=4,n_jobs=2,random_state=42)
    elif kind=="gb":est=GradientBoostingClassifier(n_estimators=80,max_depth=2,random_state=42)
    else:est=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=25,early_stopping=False,random_state=42)
    model=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),est)
    model.fit(matrix(d,names),y,**{model.steps[-1][0]+"__sample_weight":w})
    return names,model

def train(root,out):
    data=[]
    for seed in TRAIN:
        d,e,cu=joblib.load(root/f"world-{seed}"/"world.joblib")
        d["uid"]=str(seed)+":"+d.episode_id.astype(str)
        pos=np.flatnonzero(d.label.to_numpy()==1);neg=np.flatnonzero(d.label.to_numpy()==0)
        ix=np.sort(np.r_[pos,np.random.default_rng(seed).choice(neg,min(25000,len(neg)),replace=False)])
        data.append(d.iloc[ix])
    d=pd.concat(data,ignore_index=True)
    spec={"local":("local","all","rf"),"general":("base","all","rf"),
          "article_rf":("special","comp","rf"),"article_gb":("special","comp","gb"),
          "invariant":("invariant","comp","hgb"),"behavior":("behavior","comp","hgb")}
    models={n:fit(d,*s) for n,s in spec.items()}
    joblib.dump(models,out/"models.joblib")
    (out/"features.json").write_text(json.dumps({n:v[0] for n,v in models.items()},indent=2))
    print("TRAIN",len(d),d.attack.value_counts().to_dict(),flush=True)

def scores(d,models):
    s={n:m.predict_proba(matrix(d,c))[:,1] for n,(c,m) in models.items()}
    a=np.maximum(s["general"],.5*(s["article_rf"]+s["article_gb"]))
    b=np.maximum(s["general"],s["invariant"])
    return dict(local=s["local"],article_cascade=a,cu_invariant_cascade=b,
                integrated_cascade=np.maximum(b,s["behavior"]))

def evaluate(root,modelpath,out):
    models=joblib.load(modelpath/"models.joblib")
    refs=[joblib.load(root/f"world-{s}"/"world.joblib")[0] for s in CAL]
    ss=[scores(d,models) for d in refs]
    thresholds={n:joint_cut(refs,[s[n] for s in ss],RAW_BUDGET*.5,INCIDENT_BUDGET*.5) for n in MODELS}
    rows=[];episodes=[];overhead=[]
    for seed in TEST:
        d,eligible,cu=joblib.load(root/f"world-{seed}"/"world.joblib");overhead.append(dict(seed=seed,**cu))
        allscore=scores(d,models)
        for n in MODELS:
            for policy,thr in (("matched_budget",thresholds[n]),("article_fixed_0.5",.5)):
                s=allscore[n];fire=s>thr;m,det=metrics(d,s,triple(d,s),thr,eligible)
                m.update(seed=seed,model=n,policy=policy,window_precision=precision_score(d.label,fire,zero_division=0),
                    window_recall=recall_score(d.label,fire,zero_division=0),
                    window_f1=f1_score(d.label,fire,zero_division=0),
                    budget_ok=(m["false_windows_day"]<=RAW_BUDGET and m["false_notifications_day"]<=INCIDENT_BUDGET))
                rows.append(m)
                for i,e in eligible.items():
                    mask=d.episode_id.to_numpy()==i
                    episodes.append(dict(seed=seed,model=n,policy=policy,episode=i,attack=e.kind,
                      role=d.loc[mask,"ntype"].iloc[0],sophistication=e.sophistication.value if e.kind=="compromise" else "",
                      detected=int(i in det),delay=det.get(i,np.nan),window_recall=float(fire[mask].mean())))
    m=pd.DataFrame(rows);e=pd.DataFrame(episodes)
    m.to_csv(out/"metrics.csv",index=False);e.to_csv(out/"episodes.csv",index=False)
    summary=m.groupby(["policy","model"]).agg(worlds=("seed","count"),recall=("recall","mean"),
        fpr=("fpr","mean"),false_windows_day=("false_windows_day","mean"),
        false_notifications_day=("false_notifications_day","mean"),budget_worlds=("budget_ok","sum"),
        window_f1=("window_f1","mean")).reset_index()
    by=e.groupby(["policy","model","attack"]).agg(episodes=("episode","count"),
        detected=("detected","sum"),recall=("detected","mean"),window_recall=("window_recall","mean")).reset_index()
    detail=e[e.attack=="compromise"].groupby(["policy","model","role","sophistication"]).detected.agg(["count","sum","mean"]).reset_index()
    summary.to_csv(out/"summary.csv",index=False);by.to_csv(out/"by_attack.csv",index=False);detail.to_csv(out/"compromise_detail.csv",index=False)
    (out/"cu_overhead.json").write_text(json.dumps(overhead,indent=2))
    (out/"protocol.json").write_text(json.dumps(dict(train=TRAIN,cal=CAL,test=TEST,days=DAYS,thresholds=thresholds,
        source_revision=SOURCE_REV,raw_budget=RAW_BUDGET,notice_budget=INCIDENT_BUDGET,
        calibration_fraction=.5,pilot=True,baseline="50/21-dimensional functional reconstruction, not original weights",
        target="attacked node; effects on other nodes counted conservatively as negatives"),indent=2))
    print("SUMMARY\n",summary.to_string(index=False),"\nBY ATTACK\n",by.to_string(index=False),flush=True)

def selftest():
    assert len(set(TRAIN+CAL+TEST))==len(TRAIN+CAL+TEST)
    assert len(BASE)==len(set(BASE))==50 and len(SPECIAL)==21
    cfg=F.SimConfig(seed=TRAIN[0],days=DAYS);sim=IntegratedSim(cfg,F.RngBook(cfg.seed))
    e=schedule(sim,sim.book["episode"])
    assert set(x.kind for x in e if x.kind in KINDS)==set(KINDS)
    st=sim.step_true(0,[])
    for n,r in st[0].items():
        r.check_identities();st[1][n].check_identities();assert r==st[1][n]
    assert len(sim.polls)==len(sim.topo.slaves)
    assert all(sim.cu.per_link[k]<=sim.cu.reserved[k] for k in sim.cu.per_link)
    # Force all links failed: no response, no accepted report possible.
    _,p=sim.cu.exchange(1,{},set(sim.topo.linksets));assert not any(x[0] for x in p.values())
    print("CU accounting, failed paths, identity at zero, seed separation, class coverage passed",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("selftest","generate","train","evaluate"))
    p.add_argument("--seed",type=int);p.add_argument("--output",default="results")
    p.add_argument("--root",default="results/worlds");p.add_argument("--models",default="results/trained")
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.mode=="selftest":selftest()
    elif a.mode=="generate":
        assert a.seed in TRAIN+CAL+TEST
        d,e,cu=generate(a.seed);joblib.dump((d,e,cu),out/"world.joblib",compress=2)
        print("WORLD",a.seed,len(d),d.attack.value_counts().to_dict(),cu,flush=True)
    elif a.mode=="train":train(Path(a.root),out)
    else:evaluate(Path(a.root),Path(a.models),out)
if __name__=="__main__":main()
