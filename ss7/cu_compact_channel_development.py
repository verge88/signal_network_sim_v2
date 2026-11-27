"""Paired dense/compact CU transport and channel-wise calibration screen.
Synthetic interval-aggregate queues; no real network traffic or protocol I/O.
Dense and compact share topology, episode schedule and underlying traffic seeds.
Compact transmits grouped directional counts, relay marginals and local profiles.
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
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from ss7 import fix as F
from ss7 import observability_audit_development as O
from ss7 import dual_channel_behavior_development as D
from ss7.report_only_invariant_study import report_only, NEW
from ss7.incident_calibration_study import elapsed, triple, metrics, notify, bucket_peaks, threshold, RAW_BUDGET, INCIDENT_BUDGET, WARMUP
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
TRAIN=(21001,21002,21003)
RANK=(21101,21102)
CAL=(21201,21202)
TEST=(21301,21302)
DAYS=1.
# Anti-overfit protocol: temporally thin highly autocorrelated windows before fitting.
# Train/cal/test remain different simulator worlds; CAL is used only for diagnostics and thresholds.
TRAIN_STRIDE=3
TRAIN_NORMAL_PER_WORLD=12000
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

ARMS=("dense","compact")
GROUPS=(("IAM","ACM","ANM","REL","RLC"),("SRI_SM","MT_FSM","MO_FSM"),
        ("ATI","PSI","UL","ISD","CL","PURGE"),("SRI","PRN","SAI"))
GROUP={F.MT_IDX[m]:g for g,ms in enumerate(GROUPS) for m in ms}
assert len(GROUP)==len(F.MSG_TYPES)
def wire_bytes(size):
    return int(size+12*math.ceil(size/256))
def payload_bytes(topo,n,compact):
    d=len(topo.links_of(n));m=len(F.MSG_TYPES);N=len(topo.nodes)
    relay=topo.nodes[n].ntype in (F.NodeType.STP,F.NodeType.IGW)
    if not compact:return 50+8*(m+2*d+N+8+2*d*m+d*d*m)
    # Fixed-layout envelope. Base state, four-group directional counts,
    # four-group matrix marginals for relays, two eight-float64 profiles.
    # uint32 counters checked before forwarding; no omitted matrix cells are read.
    return 50+4*(m+2*d+N+8+2*d*len(GROUPS)+(2*d*len(GROUPS) if relay else 0))+16*8

def group_counts(a):
    out=defaultdict(int)
    for (direction,k,j),v in a.items():out[(direction,k,GROUP[j])]+=int(v)
    return dict(out)

def marginals(b):
    out=defaultdict(int)
    for (ki,ko,j),v in b.items():
        out[(0,ki,GROUP[j])]+=int(v);out[(1,ko,GROUP[j])]+=int(v)
    return dict(out)

def encode_compact(rep,a,b):
    # The slave computes these summaries locally; only their transmitted
    # representation is used by the coordinator. No claim of cell recovery.
    x=group_counts(a);y=marginals(b)
    st=rep.state
    for v in list(x.values())+list(y.values())+list(st.mtype)+list(st.dest.values())+list(st.link_in.values())+list(st.link_out.values())+[st.originated,st.terminated,st.transited,st.dlg_begin,st.dlg_end]:
        assert 0<=int(v)<2**32,"compact counter overflow"
    z={"beh_"+kind+"_"+k:v for kind,c in (("typed",a),("matrix",b)) for k,v in zip(D.PROFILE,D.profile(c))}
    return x,y,z

def compact_telemetry(reps,typed,mat,sensors,topo):
    grouped={n:group_counts(a) for n,a in sensors.items()};out={}
    for n,r in reps.items():
        tc=typed[n];mc=mat[n];cross=[];boundary=[];mi=[];mo=[]
        relay=topo.nodes[n].ntype in (F.NodeType.STP,F.NodeType.IGW)
        for k in topo.links_of(n):
            p=topo.peer(k,n)
            for direction,dest in ((0,mi),(1,mo)):
                for j in range(len(GROUPS)):
                    x=tc.get((direction,k,j),0)
                    if p in typed:
                        y=typed[p].get((1-direction,k,j),0)
                        if x+y:cross.append(O.discrepancy(x,y))
                        if relay:
                            z=mc.get((direction,k,j),0)
                            if z+y:dest.append(O.discrepancy(z,y))
                    if topo.nodes[p].external and n in grouped:
                        y=grouped[n].get((direction,k,j),0)
                        if x+y:boundary.append(O.discrepancy(x,y))
        _,ia,im,ic=O.stats(mi);_,oa,om,oc=O.stats(mo)
        out[n]=dict(zip(O.TYPED[:4],O.stats(cross)))
        out[n].update(dict(matrix_in_abs=ia,matrix_out_abs=oa,matrix_in_max=im,matrix_out_max=om,matrix_support=ic+oc))
        out[n].update(dict(zip(O.BOUNDARY,O.stats(boundary))))
    return out

class CUExchange:
    """One explicit request and conditional response per slave per interval.
    Planned CU demand determines the provisional finite-buffer load; realized
    per-hop attempted bits (including lost packets) determine final link load.
    Interval-aggregated queue approximation, not a discrete MTP2 frame emulator.
    50 B request/base response; extended counters have a modeled wire cost.
    """
    def __init__(self,sim,arm="compact"):
        self.sim=sim;self.arm=arm;assert arm in ARMS;topo=sim.topo
        self.master={}
        for n in topo.slaves:
            nd=topo.nodes[n]
            masters=[m for m in topo.masters if topo.nodes[m].zone==nd.zone] or list(topo.masters)
            self.master[n]=min(masters,key=lambda m:(len(topo.route(m,n).paths[0]),m))
        self.paths={n:[tuple(p) for p in topo.route(m,n).paths] for n,m in self.master.items()}
        self.payload={n:payload_bytes(topo,n,arm=="compact") for n in self.master}
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
            choices=good or opts;p=choices[(t+n)%len(choices)];path[n]=p
            for k in self.hops(p):planned[k]+=8*(wire_bytes(50)+wire_bytes(self.payload[n]))
        provisional={k:F.link_state(offered.get(k,0)+planned[k]/dt,ls,self.sim.mean_msg_bits)
                     for k,ls in topo.linksets.items()}
        def send(p,sz,rng):
            delay=0.;fragments=math.ceil(sz/256);sz=wire_bytes(sz)
            for k in self.hops(p):
                actual[k]+=8*sz
                rho,q,loss=provisional[k];ls=topo.linksets[k]
                if k in failed:return False,delay
                # M/G/1 queue + propagation + actual packet serialization.
                delay+=max(0.,q-self.sim.mean_msg_bits/ls.capacity_bps*1000)
                delay+=self.prop[k]+8*sz/ls.capacity_bps*1000
                delay+=float(rng.uniform(0,.8))
                if rng.random()<1-((1-.001)*(1-loss))**fragments:return False,delay
            return delay<dt*1000,delay
        polls={}
        for n,p in path.items():
            rq,req=send(p,50,extra_rng(self.sim.cfg.seed,22,t*1000+n))
            proc=float(extra_rng(self.sim.cfg.seed,23,t*1000+n).uniform(.8,1.2)) # same law for all nodes
            rs,resp=send(p[::-1],self.payload[n],extra_rng(self.sim.cfg.seed,24,t*1000+n)) if rq else (False,np.nan)
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
                    link_fraction=caps,arm=self.arm,
                    response_bytes_min=min(self.payload.values()),response_bytes_max=max(self.payload.values()))

class IntegratedSim(O.CorrectedSim):
    def __init__(self,cfg,book,arm="compact"):
        super().__init__(cfg,book)
        self.cu=CUExchange(self,arm)
    def _build_flows(self,rng):
        fl=super()._build_flows(rng)
        # Legitimate location traffic, rather than a code seen only in attacks.
        for sp in (n for ns in self.topo.zone_sp.values() for n in ns):
            for h in self.topo.hlr:fl.append(F.Flow(sp,h,Service.LOCATION,float(rng.gamma(1.5,.04))))
        return fl
    def flows_for(self,t,episodes):
        """Synthetic service-volume profiles on the existing simulated graph."""
        active={e.nid:e for e in episodes if e.kind in KINDS[:-1] and e.t0<=t<=e.t1}
        if not active:
            yield from self.flows
            return
        grouped=defaultdict(list)
        for f in self.flows:
            if f.src not in active:yield f
            else:grouped[f.src].append(f)
        for n,fs in grouped.items():
            e=active[n];kind=e.kind
            weight=lambda f: f.rate*sum(F.SERVICE_TEMPLATE[f.service].values())
            total=sum(weight(f) for f in fs);target=total*e.mult
            if kind=="signalling_dos":
                for f in fs:yield F.Flow(f.src,f.dst,f.service,f.rate*e.mult)
                continue
            if kind=="location_tracking":
                mix=.6+.15*float(e.params[0])
                for f in fs:yield F.Flow(f.src,f.dst,f.service,f.rate*e.mult*(1-mix))
                dst=self.topo.hlr[min(len(self.topo.hlr)-1,int(float(e.params[1])*len(self.topo.hlr)))]
                yield F.Flow(n,dst,Service.LOCATION,target*mix/sum(F.SERVICE_TEMPLATE[Service.LOCATION].values()))
                continue
            # Reweight service-category totals, without protocol-level attack scripts.
            group=(Service.SMS_MT,Service.MOBILITY) if kind=="sms_interception" else (Service.VOICE_INT,)
            mix=(.65+.15*float(e.params[0])) if kind=="sms_interception" else (.35+.40*float(e.params[0]))
            high=sum(weight(f) for f in fs if f.service in group);low=total-high
            if high<=0 or low<=0:
                for f in fs:yield F.Flow(f.src,f.dst,f.service,f.rate*e.mult)
                continue
            for f in fs:
                scale=target*mix/high if f.service in group else target*(1-mix)/low
                yield F.Flow(f.src,f.dst,f.service,f.rate*scale)

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
    eps=F.make_episodes(sim,cfg,rng,n_compromise=0,n_normal=48,n_ext=0)
    occ=defaultdict(list)
    def place(nodes,kind,length,**kwargs):
        for _ in range(5000):
            n=int(rng.choice(nodes));a=int(rng.integers(WARMUP+5,cfg.n_intervals-length-5));b=a+length-1
            if all(b+3<x or a>y+3 for x,y in occ[n]):
                occ[n].append((a,b));eps.append(Episode(kind=kind,nid=n,t0=a,t1=b,**kwargs));return
        raise RuntimeError("no nonoverlapping slot")
    sp=[n for n in sim.topo.slaves if sim.topo.nodes[n].ntype==F.NodeType.SP]
    for k in KINDS[:-1]:
        for _ in range(6):
            mult={"sms_interception":(2.,2.5),"location_tracking":(1.5,1.8),
                  "signalling_dos":(30.,40.),"irsf":(3.,4.)}[k]
            place(sp,k,int(rng.integers(30,61)),mult=float(rng.uniform(*mult)),params=rng.random(2))
    for role in F.NodeType:
        nodes=[n for n in sim.topo.slaves if sim.topo.nodes[n].ntype==role]
        for soph in F.Sophistication:
            for rates in ((.05,.15,.4),(1.,2.5)):
                place(nodes,"compromise",int(rng.integers(60,91)),
                      sophistication=soph,hidden_rate=float(rng.choice(rates)),
                      family=str(rng.choice(("F1_uniform","F4_temporal"))),
                      budget=float(rng.uniform(.7,1.)),params=rng.random(2))
    return eps

def features(d):
    d=d.sort_values(["nid","t"]).reset_index(drop=True)
    for name in TEMP:d[name]=np.nan
    local_temp=[x for x in TEMP if x!="integrity_fail_rate"]
    for name in local_temp:d["local_"+name]=np.nan
    for _,g in d.groupby("nid",sort=False):
        for base,name in zip(("total_messages","mtype_entropy","in_out_ratio","dest_top1_ratio","n_unique_dest"),
                            ("cv_total","cv_entropy","cv_io","cv_dominance","cv_dest")):
            rr=g[base].rolling(30,min_periods=5)
            d.loc[g.index,name]=(rr.std()/(rr.mean().abs()+1e-9)).to_numpy()
            lr=g["local_"+base].rolling(30,min_periods=5)
            d.loc[g.index,"local_"+name]=(lr.std()/(lr.mean().abs()+1e-9)).to_numpy()
        d.loc[g.index,"ar1_total"]=g.total_messages.rolling(30,min_periods=5).corr(g.total_messages.shift(1)).to_numpy()
        d.loc[g.index,"local_ar1_total"]=g.local_total_messages.rolling(30,min_periods=5).corr(g.local_total_messages.shift(1)).to_numpy()
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

def generate(seed,days=DAYS,arm="compact"):
    cfg=F.SimConfig(seed=seed,days=days);book=F.RngBook(seed)
    sim=IntegratedSim(cfg,book,arm);eps=schedule(sim,book["episode"])
    ids={id(e):i for i,e in enumerate(eps)}
    last={};losses=defaultdict(int);age=defaultdict(int);history=defaultdict(lambda:deque(maxlen=30))
    jitter=defaultdict(lambda:deque(maxlen=10))
    rows=[]
    for t in range(cfg.n_intervals):
        truth,claimed,typed,mat,boundary,audit,lstate=sim.step_true(t,eps)
        rng=book.child("measure",t);trng=extra_rng(seed,2,t);brng=extra_rng(seed,3,t)
        active={e.nid:e for e in eps if e.kind in KINDS and e.t0<=t<=e.t1}
        fresh={};tc={};mc={};sensors={};obs={};profiles={}
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
            if arm=="compact":a,b,z=encode_compact(rep,a,b)
            if node.role==F.NodeRole.MASTER:
                fresh[n]=rep;tc[n]=a;mc[n]=b
                if arm=="compact":profiles[n]=z
                continue
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
                if arm=="compact":profiles[n]=z
            r=last.get(n)
            if r is not None:assert (r.t==t) if valid else (r.t<t)
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
            # Local ablation receives the pre-transport claimed counters only.
            vals.update({"local_"+c:v for c,v in rep.feats.items()})
            lm=sum(rep.state.mtype[5:])
            vals.update({"local_obs_ratio_"+m:float(rep.state.mtype[F.MT_IDX[m]])/max(lm,1) for m in MAP})
            obs[n]=vals
        # Compare only current authenticated observations. Missing report != zero.
        inv=report_only(fresh,sim.topo)
        if arm=="dense":
            extra=O.telemetry(fresh,tc,mc,sensors,sim.topo)
            behavior=D.behavior_snapshot(fresh,tc,mc,sensors,sim.topo)
        else:
            extra=compact_telemetry(fresh,tc,mc,sensors,sim.topo)
            # Compute only base-state/boundary features here. The full typed
            # and matrix profiles arrive as scalars, never as omitted cells.
            behavior=D.behavior_snapshot(fresh,{n:{} for n in fresh},{n:{} for n in fresh},sensors,sim.topo)
            for n in fresh:behavior[n].update(profiles[n])
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
    if mode=="local":return ["local_"+x for x in list(F.OWN)+list(SUB)+[x for x in TEMP if x!="integrity_fail_rate"]]+["node_type_enc"]
    if mode=="base":return BASE
    if mode=="special":return SPECIAL
    if mode=="invariant":return BASE+[c for c in d.columns if any(c==b or c.startswith(b+"__") for b in list(NEW)+list(O.EXTRA))]
    if mode=="behavior":return list(F.OWN)+list(D.BEHAVIOR)+[c for c in d if c.startswith("hist_")]+list(CU)
    raise ValueError(mode)

def matrix(d,names):
    forbidden={"label","episode_id","attack","sophistication","nid","t","zone","ntype","uid","world_node"}
    assert not forbidden.intersection(names) and not any(c.startswith(("audit_","hidden_","true_","gt_")) for c in names)
    return d[names].to_numpy(float)

def target_y(d,target):
    return (d.attack==("compromise" if target=="comp" else target)).to_numpy(int) if target!="all" else d.label.to_numpy(int)

def fit(d,mode,target,kind):
    names=cols(d,mode);y=target_y(d,target)
    # Weight independent causal units rather than every highly correlated row.
    w=np.ones(len(d));p=y==1;n=~p
    if p.any():
        g=d.loc[p,"uid"];counts=g.value_counts()
        w[p]=1/g.map(counts).to_numpy()
    if n.any():
        g=d.loc[n,"uid"].where(d.loc[n,"label"].astype(bool),d.loc[n,"world_node"])
        counts=g.value_counts();w[n]=1/g.map(counts).to_numpy()
    for k in (0,1):
        m=y==k
        if m.any():w[m]*=len(y)/(2*w[m].sum())
    # Lower-capacity learners reduce memorization of simulator-world details.
    if kind=="rf":
        est=RandomForestClassifier(n_estimators=120,max_depth=10,min_samples_leaf=20,
            max_features="sqrt",bootstrap=True,max_samples=.70,n_jobs=2,random_state=42)
    elif kind=="gb":
        est=GradientBoostingClassifier(n_estimators=80,learning_rate=.04,max_depth=1,
            min_samples_leaf=30,subsample=.65,max_features="sqrt",random_state=42)
    else:
        est=HistGradientBoostingClassifier(max_iter=70,learning_rate=.05,max_leaf_nodes=7,
            min_samples_leaf=50,l2_regularization=2.0,early_stopping=False,random_state=42)
    model=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),est)
    model.fit(matrix(d,names),y,**{model.steps[-1][0]+"__sample_weight":w})
    return names,model
OTHER=KINDS[:-1]
COMMON_MODELS=("local","article_raw","integrated_raw","dual_tail","dual_tail_confirm","five_tail","five_role_tail")
MODELS=COMMON_MODELS+("five_calibrated",)
FIVE_CHANNELS=OTHER+("invariant",)
def load(root,arm,seed):
    return joblib.load(root/f"world-{arm}-{seed}"/"world.joblib")

def train(root,out,arm):
    data=[]
    for seed in TRAIN:
        d,e,cu=load(root,arm,seed)
        d["uid"]=str(seed)+":"+d.episode_id.astype(str)
        d["world_node"]=str(seed)+":"+d.nid.astype(str)
        phase=seed%TRAIN_STRIDE
        thin=(d.t.to_numpy()%TRAIN_STRIDE)==phase
        pos=np.flatnonzero((d.label.to_numpy()==1)&thin)
        neg=np.flatnonzero((d.label.to_numpy()==0)&thin)
        neg=np.random.default_rng(seed).choice(neg,min(TRAIN_NORMAL_PER_WORLD,len(neg)),replace=False)
        ix=np.sort(np.r_[pos,neg]);data.append(d.iloc[ix])
    d=pd.concat(data,ignore_index=True)
    spec={"local":("local","all","rf"),"general":("base","all","rf"),
          "article_rf":("special","comp","rf"),"article_gb":("special","comp","gb"),
          "invariant":("invariant","comp","hgb"),"behavior":("behavior","comp","hgb")}
    spec.update({k:("base",k,"hgb") for k in OTHER})
    models={}
    for n,sp in spec.items():
        models[n]=fit(d,*sp);print("FIT",arm,n,flush=True)

    # Train-vs-independent-CAL diagnostics. TEST is never loaded here.
    cal=[]
    for seed in CAL:
        q,_,_=load(root,arm,seed)
        q["uid"]=str(seed)+":"+q.episode_id.astype(str)
        q["world_node"]=str(seed)+":"+q.nid.astype(str)
        cal.append(q)
    cal=pd.concat(cal,ignore_index=True)
    diag=[]
    for name,(mode,target,kind) in spec.items():
        names,model=models[name]
        for split,frame in (("train",d),("cal",cal)):
            y=target_y(frame,target);score=model.predict_proba(matrix(frame,names))[:,1]
            diag.append(dict(model=name,split=split,n=len(frame),positive=int(y.sum()),
                auc_roc=roc_auc_score(y,score) if len(np.unique(y))==2 else np.nan,
                auc_pr=average_precision_score(y,score) if y.sum() else np.nan,
                f1_at_05=f1_score(y,score>=.5,zero_division=0)))
    dg=pd.DataFrame(diag)
    wide=dg.pivot(index="model",columns="split",values="auc_pr")
    gaps=(wide["train"]-wide["cal"]).sort_values(ascending=False)
    print("GENERALIZATION\n",dg.to_string(index=False),flush=True)
    print("AUC_PR_GAP\n",gaps.to_string(),flush=True)
    dg.to_csv(out/"generalization.csv",index=False)
    joblib.dump(models,out/"models.joblib")
    (out/"features.json").write_text(json.dumps({n:v[0] for n,v in models.items()},indent=2))
    (out/"anti_overfit.json").write_text(json.dumps(dict(train_stride=TRAIN_STRIDE,
        normal_per_world=TRAIN_NORMAL_PER_WORLD,
        rf=dict(max_depth=10,min_samples_leaf=20,max_features="sqrt",max_samples=.70),
        gb=dict(max_depth=1,min_samples_leaf=30,learning_rate=.04,subsample=.65),
        hgb=dict(max_leaf_nodes=7,min_samples_leaf=50,learning_rate=.05,l2_regularization=2.0)),indent=2))
    print("TRAIN",arm,len(d),d.attack.value_counts().to_dict(),flush=True)
def predict(d,models):
    s={n:m.predict_proba(matrix(d,c))[:,1] for n,(c,m) in models.items()}
    for n,v in s.items():assert np.isfinite(v).all(),n
    s["article_special"]=.5*(s["article_rf"]+s["article_gb"])
    return s

CHANNELS=("general","invariant","behavior")+OTHER

def fit_tails(frames,preds):
    # Only normal observations of the dedicated rank-reference worlds.
    # Per-world survival functions; worst-case evidence across worlds.
    ref={}
    for k in CHANNELS:
        ref[k]={}
        for role in ("all",)+tuple(x.value for x in F.NodeType):
            vals=[]
            for d,p in zip(frames,preds):
                mask=(d.label.to_numpy()==0)
                if role!="all":mask&=(d.ntype.to_numpy()==role)
                a=np.sort(p[k][mask])
                if len(a)>=1000:vals.append(a)
            if len(vals)==len(frames):ref[k][role]=vals
        assert "all" in ref[k]
    return ref

def tail(a,x):
    # Include score ties conservatively. No unsupported parametric extrapolation.
    return -np.log10((len(a)-np.searchsorted(a,x,side="left")+1)/(len(a)+1.))

def evidence(d,p,ref,roles=False):
    out={}
    for k in CHANNELS:
        v=np.empty(len(d))
        if not roles:v[:]=np.min([tail(a,p[k]) for a in ref[k]["all"]],axis=0)
        else:
            for role in np.unique(d.ntype):
                mask=d.ntype.to_numpy()==role
                v[mask]=np.min([tail(a,p[k][mask]) for a in ref[k].get(role,ref[k]["all"])],axis=0)
        out[k]=v
    return out

def channels(d,p,ref,mode):
    if mode=="local":return {"local":p["local"]},False
    if mode=="article_raw":return {"general":p["general"],"article_special":p["article_special"]},False
    if mode=="integrated_raw":return {k:p[k] for k in ("general","invariant","behavior")},False
    ev=evidence(d,p,ref,roles=mode=="five_role_tail")
    if mode.startswith("dual_"):
        return {k:ev[k] for k in ("general","invariant","behavior")},mode=="dual_tail_confirm"
    return {k:ev[k] for k in OTHER+("invariant","behavior")},True

def streak_score(d,s,length=3):
    """Minimum score across a causal same-node run of *length* windows."""
    if length<1:raise ValueError("length must be positive")
    s=np.asarray(s,float);n=d.nid.to_numpy();t=d.t.to_numpy();N=len(s)
    if length==1:return s.copy()
    # Vectorized causal rolling minimum. A candidate is valid only when every
    # shifted element belongs to the same node and has the exact time offset.
    z=s.copy();valid=np.ones(N,dtype=bool);valid[:length-1]=False
    for k in range(1,length):
        shifted=np.full(N,np.inf);shifted[k:]=s[:-k]
        z=np.minimum(z,shifted)
        ok=np.zeros(N,dtype=bool)
        ok[k:]=(n[k:]==n[:-k])&(t[k:]==t[:-k]+k)
        valid&=ok
    z[~valid]=-np.inf
    return z

def combined(d,ch,confirm,negative_only=False,streak_len=3):
    neg=(d.label.to_numpy()==0) if negative_only else np.ones(len(d),bool)
    vals=[np.where(neg,v,-np.inf) for v in ch.values()]
    s=np.maximum.reduce(vals)
    # Specialized fusion must be confirmed within one channel. This prevents
    # alternating weak channels from forming a synthetic multi-window alarm.
    z=np.maximum.reduce([streak_score(d,v,streak_len) for v in vals]) if confirm else streak_score(d,s,streak_len)
    return s,z

def cut_policy(frames,preds,ref,mode,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    """Calibrate raw score and persistence separately on independent CAL worlds.

    The old rule used max(raw_threshold, notification_threshold), so the small
    notification budget forced the score threshold into the extreme tail.
    Here the score threshold alone enforces the raw-window budget. We then
    choose the shortest predeclared persistence length that satisfies the
    notification budget on every CAL world. No TEST labels or scores are used.
    """
    raw_candidates=[]
    prepared=[]
    for d,p in zip(frames,preds):
        ch,confirm=channels(d,p,ref,mode)
        s,_=combined(d,ch,confirm,negative_only=True,streak_len=3)
        raw_candidates.append(threshold(s[np.isfinite(s)],raw,elapsed(d)))
        prepared.append((d,ch,confirm))
    thr=float(max(raw_candidates))

    chosen=None
    diagnostics=[]
    for length in (3,5,7,9,12):
        rates=[]
        for d,ch,confirm in prepared:
            s,z=combined(d,ch,confirm,negative_only=True,streak_len=length)
            rate=float(len(notify(d,z,thr))/elapsed(d))
            rates.append(rate)
        diagnostics.append(dict(streak=length,worst_notifications_day=max(rates),rates=rates))
        if max(rates)<=notice:
            chosen=length;break

    # Rare finite-sample fallback: keep the longest persistence and increase
    # only as much as required by the notification budget.
    if chosen is None:
        chosen=12
        extra=[]
        for d,ch,confirm in prepared:
            _,z=combined(d,ch,confirm,negative_only=True,streak_len=chosen)
            b=bucket_peaks(d,z);b=b[np.isfinite(b)]
            if len(b):extra.append(threshold(b,notice,elapsed(d)))
        if extra:thr=max(thr,float(max(extra)))

    return dict(threshold=float(thr),streak=int(chosen),diagnostics=diagnostics)


def _logit(x):
    x=np.clip(np.asarray(x,float),1e-8,1-1e-8)
    return np.log(x)-np.log1p(-x)

def five_raw_channels(p,thresholds):
    # Four attack-specific experts plus the invariant compromise expert.
    # Each channel is converted to a log-odds margin around its own frozen
    # calibration threshold, so margin > 0 has the same decision meaning.
    return {k:_logit(p[k])-_logit(thresholds[k]) for k in FIVE_CHANNELS}

def calibrate_five_raw(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    """Per-class calibration with a shared margin cut using the full FPR budget.

    First obtain conservative class-specific anchors. Then calibrate one common
    log-odds margin threshold on the fused negative score so the union, rather
    than each channel separately, can use the full raw-window budget. Finally
    choose persistence using only CAL worlds to satisfy the notification budget.
    """
    share=float(raw/len(FIVE_CHANNELS));thresholds={}
    for k in FIVE_CHANNELS:
        thresholds[k]=float(max(
            threshold(p[k][d.label.to_numpy()==0],share,elapsed(d))
            for d,p in zip(frames,preds)
        ))

    prepared=[]
    raw_cuts=[]
    for d,p in zip(frames,preds):
        ch=five_raw_channels(p,thresholds)
        s,_=combined(d,ch,True,negative_only=True,streak_len=3)
        raw_cuts.append(threshold(s[np.isfinite(s)],raw,elapsed(d)))
        prepared.append((d,ch))
    margin_cut=float(max(raw_cuts))

    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        rates=[];raw_rates=[]
        for d,ch in prepared:
            s,z=combined(d,ch,True,negative_only=True,streak_len=length)
            raw_rates.append(float((s>margin_cut).sum()/elapsed(d)))
            rates.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(rates),
            worst_windows_day=max(raw_rates),rates=rates))
        if max(rates)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="five_raw",threshold=margin_cut,streak=int(chosen),
        raw_budget_share=share,channel_thresholds=thresholds,diagnostics=diagnostics)

def policy_channels(d,p,ref,name,policy):
    if policy.get("kind")=="five_raw":
        return five_raw_channels(p,policy["channel_thresholds"]),True,float(policy["threshold"])
    n="integrated_raw" if name=="integrated_half" else name
    ch,confirm=channels(d,p,ref,n)
    return ch,confirm,float(policy["threshold"])

def class_name(k):
    return "compromise" if k in ("invariant","behavior","article_special") else k

def evaluate(root,modelpath,out,arm):
    models=joblib.load(modelpath/"models.joblib")
    ranks=[load(root,arm,s)[0] for s in RANK];ref=fit_tails(ranks,[predict(d,models) for d in ranks])
    del ranks
    cal=[load(root,arm,s)[0] for s in CAL];ps=[predict(d,models) for d in cal]
    policies={n:cut_policy(cal,ps,ref,n) for n in COMMON_MODELS}
    policies["five_calibrated"]=calibrate_five_raw(cal,ps)
    policies["integrated_half"]=cut_policy(cal,ps,ref,"integrated_raw",RAW_BUDGET*.5,INCIDENT_BUDGET*.5)

    controls=[]
    for seed,d,p in zip(CAL,cal,ps):
        for name,policy in policies.items():
            ch,confirm,thr=policy_channels(d,p,ref,name,policy)
            s,z=combined(d,ch,confirm,negative_only=True,streak_len=policy["streak"])
            controls.append(dict(seed=seed,arm=arm,model=name,
                threshold=thr,streak=policy["streak"],
                false_windows_day=float((s>thr).sum()/elapsed(d)),
                false_notifications_day=float(len(notify(d,z,thr))/elapsed(d))))
    controls=pd.DataFrame(controls)
    controls.to_csv(out/"calibration_controls.csv",index=False)
    print("POLICIES",arm,json.dumps(policies,indent=2),flush=True)
    print("CALIBRATION\n",controls.to_string(index=False),flush=True)
    del cal,ps

    rows=[];episodes=[];overhead=[]
    for seed in TEST:
        d,eligible,cu=load(root,arm,seed);overhead.append(dict(seed=seed,**cu))
        p=predict(d,models)
        for name,policy in policies.items():
            streak_len=policy["streak"]
            ch,confirm,thr=policy_channels(d,p,ref,name,policy)
            s,z=combined(d,ch,confirm,streak_len=streak_len)
            m,det=metrics(d,s,z,thr,eligible);fire=s>thr
            m.update(seed=seed,arm=arm,model=name,streak=streak_len,
                window_f1=f1_score(d.label,fire,zero_division=0),
                budget_ok=(m["false_windows_day"]<=RAW_BUDGET and m["false_notifications_day"]<=INCIDENT_BUDGET))
            rows.append(m)
            names=list(ch);zz=np.column_stack([streak_score(d,ch[k],streak_len) for k in names])
            for i,e in eligible.items():
                mask=d.episode_id.to_numpy()==i;predicted="missed"
                if i in det:
                    ix=np.flatnonzero(mask&(d.t.to_numpy()==e.t0+det[i]//30))
                    if len(ix):predicted=class_name(names[int(np.argmax(zz[ix[0]]))])
                episodes.append(dict(seed=seed,arm=arm,model=name,episode=i,attack=e.kind,
                    role=d.loc[mask,"ntype"].iloc[0],sophistication=e.sophistication.value if e.kind=="compromise" else "",
                    rate=e.hidden_rate,detected=int(i in det),predicted=predicted,
                    typed=int(predicted==e.kind),delay=det.get(i,np.nan),window_recall=float(fire[mask].mean())))
        print("TEST",arm,seed,len(d),cu["responses_delivered"]/cu["polls"],cu["max_mean_link_fraction"],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(episodes).to_csv(out/"episodes.csv",index=False)
    (out/"cu.json").write_text(json.dumps(overhead,indent=2))
    (out/"thresholds.json").write_text(json.dumps(policies,indent=2))
    print("METRICS",pd.DataFrame(rows).to_string(index=False),flush=True)


def aggregate(root,out):
    m=pd.concat([pd.read_csv(root/f"eval-{a}"/"metrics.csv") for a in ARMS],ignore_index=True)
    e=pd.concat([pd.read_csv(root/f"eval-{a}"/"episodes.csv") for a in ARMS],ignore_index=True)
    cu=pd.DataFrame([x for a in ARMS for x in json.loads((root/f"eval-{a}"/"cu.json").read_text())])
    cu["delivery"]=cu.responses_delivered/cu.polls
    for a in ARMS:
        q=e[e.arm==a].drop_duplicates(["seed","episode"])
        assert len(q)==len(TEST)*48
    # Matched scenarios across transport arms (not independent extra worlds).
    a=e[(e.arm=="dense")&(e.model=="integrated_raw")].sort_values(["seed","episode"])
    b=e[(e.arm=="compact")&(e.model=="integrated_raw")].sort_values(["seed","episode"])
    pd.testing.assert_frame_equal(a[["seed","episode","attack","role","sophistication","rate"]].reset_index(drop=True),
                                  b[["seed","episode","attack","role","sophistication","rate"]].reset_index(drop=True))
    summary=m.groupby(["arm","model"]).agg(worlds=("seed","count"),detected=("detected","sum"),episodes=("episodes","sum"),
        recall=("recall","mean"),fpr=("fpr","mean"),raw_day=("false_windows_day","mean"),
        worst_raw=("false_windows_day","max"),notice_day=("false_notifications_day","mean"),
        worst_notice=("false_notifications_day","max"),budget_worlds=("budget_ok","sum"),
        window_f1=("window_f1","mean")).reset_index()
    by=e.groupby(["arm","model","attack"]).agg(episodes=("episode","count"),detected=("detected","sum"),
        recall=("detected","mean"),typed_recall=("typed","mean")).reset_index()
    macro=by.groupby(["arm","model"]).recall.agg(macro_recall="mean",worst_class="min").reset_index()
    summary=summary.merge(macro,on=["arm","model"])
    comp=e[e.attack=="compromise"].groupby(["arm","model","role","sophistication"]).detected.agg(["count","sum","mean"]).reset_index()
    for n,d in (("metrics",m),("episodes",e),("summary",summary),("by_attack",by),("compromise",comp),("cu",cu.drop(columns=["link_fraction"]))):
        d.to_csv(out/f"{n}.csv",index=False)
    (out/"protocol.json").write_text(json.dumps(dict(train=TRAIN,rank=RANK,cal=CAL,test=TEST,days=DAYS,
        source_revision="a2a246d37e3faf44675e00ee39b50f9b54b80824",raw_budget=RAW_BUDGET,notice_budget=INCIDENT_BUDGET,
        arms=ARMS,fragment_payload_bytes=256,fragment_overhead_bytes=12,pilot=True,
        caveat="paired synthetic screen, no test selection or confirmatory inference"),indent=2))
    print("SUMMARY\n",summary.to_string(index=False),"\nBY ATTACK\n",by.to_string(index=False),"\nCU\n",
        cu[["seed","arm","delivery","max_mean_link_fraction","response_bytes_min","response_bytes_max","wire_bytes"]].to_string(index=False),flush=True)

def selftest():
    assert len(set(TRAIN+RANK+CAL+TEST))==len(TRAIN+RANK+CAL+TEST)
    cfg=F.SimConfig(seed=TRAIN[0],days=DAYS)
    for arm in ARMS:
        sim=IntegratedSim(cfg,F.RngBook(cfg.seed),arm)
        e=schedule(sim,sim.book["episode"])
        assert sum(x.kind in KINDS for x in e)==48
        assert sum(x.kind=="compromise" for x in e)==24
        st=sim.step_true(0,[])
        for n,r in st[0].items():
            r.check_identities();st[1][n].check_identities();assert r==st[1][n]
        assert len(sim.polls)==len(sim.topo.slaves)
        _,p=sim.cu.exchange(1,{},set(sim.topo.linksets));assert not any(x[0] for x in p.values())
        assert all(payload_bytes(sim.topo,n,True)<payload_bytes(sim.topo,n,False) for n in sim.topo.slaves)
    a={(0,(1,2),F.MT_IDX["IAM"]):4,(0,(1,2),F.MT_IDX["REL"]):3}
    b={((1,2),(2,3),F.MT_IDX["IAM"]):4}
    assert group_counts(a)=={(0,(1,2),0):7}
    assert marginals(b)=={(0,(1,2),0):4,(1,(2,3),0):4}
    assert np.allclose(tail(np.array([.1,.2,.3]),np.array([.1,.2,.4])), -np.log10([1.,.75,.25]))
    # Ensure no cross-channel confirmation for alternating high values.
    d=pd.DataFrame(dict(nid=[1]*3,t=[1,2,3],label=[0]*3))
    ch={"a":np.array([1.,0.,1.]),"b":np.array([0.,1.,0.])}
    assert combined(d,ch,False)[1][-1]==1 and combined(d,ch,True)[1][-1]==0
    print("paired schedule, CU wire accounting, grouped marginals, conservative ties and causal channel confirmation passed",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("selftest","generate","train","evaluate","aggregate"))
    p.add_argument("--seed",type=int);p.add_argument("--arm",choices=ARMS,default="compact")
    p.add_argument("--output",default="results");p.add_argument("--root",default="results/worlds")
    p.add_argument("--models",default="results/trained")
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.mode=="selftest":selftest()
    elif a.mode=="generate":
        assert a.seed in TRAIN+RANK+CAL+TEST
        d,e,cu=generate(a.seed,arm=a.arm);joblib.dump((d,e,cu),out/"world.joblib",compress=2)
        print("WORLD",a.arm,a.seed,len(d),d.attack.value_counts().to_dict(),cu,flush=True)
    elif a.mode=="train":train(Path(a.root),out,a.arm)
    elif a.mode=="evaluate":evaluate(Path(a.root),Path(a.models),out,a.arm)
    else:aggregate(Path(a.root),out)
if __name__=="__main__":main()
