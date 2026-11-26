"""Integrated CU transport + role-consistent compromise + five-class benchmark.

Synthetic offered-message traffic, M/G/1 delay and M/M/1/K blocking approximations.
CU requests and segmented reports share offered bandwidth with transaction traffic.
Scheduled response bandwidth is reserved even when request fails (conservative).
No measured operator data, packet-level MTP implementation, or exact article
feature registry is available. Four conventional attacks are service-load
abstractions, not proof of actual SMS interception/tracking/fraud.
"""
from __future__ import annotations
import argparse,itertools,json,math
from collections import defaultdict
from pathlib import Path
from typing import Dict,List,Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from ss7 import fix as F
from ss7.fix import Episode,NodeState,Service,MSG_TYPES,MSG_BYTES,SERVICE_TEMPLATE,INTERVALS_PER_DAY
from ss7 import observability_audit_development as O
from ss7 import dual_channel_behavior_development as D
from ss7 import factorial_model_comparison as B
from ss7.report_only_invariant_study import report_only,NEW
from ss7.report_feature_specialist import directional,DIRECTION,PEER,add_history
from ss7.weak_signal_confirmation import fit_model,weak_predict
from ss7.cusum_cascade_development import joint_cut
from ss7.episode_ranking_development import union,collect
from ss7.incident_calibration_study import elapsed,triple,metrics,notify,RAW_BUDGET,INCIDENT_BUDGET,WARMUP

TRAIN=tuple(range(16001,16005))
VAL=tuple(range(16101,16105))
TEST=tuple(range(16201,16209))
OTHER=("sms_intercept","location_track","signaling_dos","irsf")
CLASSES=("compromise",)+OTHER
ATTACK_ROLE=dict(sms_intercept="SCP",location_track="SCP",signaling_dos="STP",irsf="SP")
RANGES=dict(sms_intercept=(2.,2.5),location_track=(1.5,1.8),signaling_dos=(30.,40.),irsf=(3.,4.))
CU=("cu_delivered","cu_rtt_delay","cu_combined_loss_prob","integrity_check",
    "cu_processing_delay","cu_response_time_jitter","cu_staleness","cu_consistency_error")
MAP_CODES=tuple(m for m in F.MSG_TYPES if m not in ("IAM","ACM","ANM","REL","RLC"))
CV=("total_messages","mtype_entropy","map_share","in_out_ratio","cu_rtt_delay")
AC=("total_messages","mtype_entropy","in_out_ratio")
ZONE=("total_messages","mtype_entropy","map_share","in_out_ratio","cu_rtt_delay","cu_staleness","cu_delivered","integrity_check")
ARTICLE=CU+tuple(F.OWN)+tuple("share_"+m for m in MAP_CODES)+("international_fraction",)+tuple("cv_"+m for m in CV)+tuple("ac_"+m for m in AC)+tuple("zone_"+m for m in ZONE)+("node_type_enc","integrity_fail_rate")
SPECIAL=CU+tuple("cv_"+m for m in CV)+tuple("ac_"+m for m in AC)+("integrity_fail_rate","zone_in_out_ratio","zone_cu_rtt_delay","zone_cu_staleness","in_out_ratio")
BASE=("article_rf","article_cascade","expanded_no_cu","expanded_cu")
CANDIDATES=("expanded_cu","expanded_special","dual_25","dual_50","dual_75")
PAYLOAD=256;FRAME_OVERHEAD=12
DAYS=3.
assert len(ARTICLE)==50 and len(SPECIAL)==21

def fragments(n):return max(1,math.ceil(n/PAYLOAD))
def wire_bytes(n):return n+FRAME_OVERHEAD*fragments(n)

def report_bytes(topo,n):
    # Fixed-layout padded upper bound, independent of traffic or attack label.
    # Header, per-code counters, peer destinations, scalar counters,
    # typed link directions and ordered transit pairs, 32-bit counters.
    d=len(topo.links_of(n));m=len(F.MSG_TYPES)
    return 50+4*(m+len(topo.nodes)+7+2*d+2*d*m+d*(d-1)*m)

class IntegratedSim(O.CorrectedSim):
    def _build_flows(self,rng):
        fl=super()._build_flows(rng)
        # Legitimate low-rate location service also present in background.
        # Avoid making the operation category itself an attack label.
        for e in self.topo.external:
            for h in self.topo.hlr:
                fl.append(F.Flow(e,h,Service.LOCATION,float(rng.gamma(1.5,.04))))
        return fl

    def attack_paths(self,kind,n):
        if not hasattr(self,"_attack_cache"):self._attack_cache={}
        key=(kind,n)
        if key in self._attack_cache:return self._attack_cache[key]
        out=defaultdict(list)
        for f in self.flows:
            if kind=="sms_intercept" and not (f.dst==n and f.service in (Service.SMS_MT,Service.MOBILITY)):continue
            if kind=="location_track" and not (f.dst==n and f.service==Service.LOCATION):continue
            if kind=="irsf" and not (f.src==n and f.service==Service.VOICE_INT):continue
            for path in self.topo.route(f.src,f.dst).paths:
                if n not in path:continue
                if kind=="signaling_dos" and n not in path[1:-1]:continue
                out[f.service].append((f,path))
        if not out:raise ValueError((kind,n,"no legal traffic support"))
        self._attack_cache[key]=dict(out)
        return self._attack_cache[key]

    def poll_plans(self,t,failed):
        if not hasattr(self,"_poll_cache"):
            masters=[n for n in self.topo.nodes if self.topo.nodes[n].role==F.NodeRole.MASTER]
            self._poll_cache={}
            for n in self.topo.slaves:
                z=self.topo.nodes[n].zone
                options=[m for m in masters if self.topo.nodes[m].zone==z] or masters
                paths=[p for m in options for p in self.topo.route(m,n).paths]
                paths=sorted(paths,key=lambda p:(len(p),p))
                if not paths:raise ValueError("unreachable monitored node")
                self._poll_cache[n]=(paths,report_bytes(self.topo,n))
        out={}
        for n,(paths,size) in self._poll_cache.items():
            valid=[p for p in paths if not any(tuple(sorted((u,v))) in failed for u,v in zip(p[:-1],p[1:]))]
            # If no path is available, no offered CU is transmitted.
            out[n]=(valid[0] if valid else (),size)
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
        arng=O.extra_rng(self.cfg.seed,1,t)
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
        # Additional attack classes inject traffic through legal existing routes.
        # All are traffic-profile abstractions, not packet-level attack implementations.
        xrng=O.extra_rng(cfg.seed,4,t)
        for ep in episodes:
            if ep.kind not in OTHER or not ep.t0<=t<=ep.t1:continue
            groups=self.attack_paths(ep.kind,ep.nid)
            for service,opts in groups.items():
                usable=[(f,p) for f,p in opts if not any(tuple(sorted((u,v))) in failed_links for u,v in zip(p[:-1],p[1:]))]
                if not usable:continue
                f,path=usable[int(xrng.integers(len(usable)))]
                # Aggregate service-specific extra offered transactions, not edited features.
                rate=sum(f0.rate/len(topo.route(f0.src,f0.dst).paths) for f0,p0 in opts)
                cnt=int(xrng.poisson(max(0.,rate*(ep.mult-1)*dt*diurnal)))
                if not cnt:continue
                pending.append((f,path,cnt,None))
                bits=8.*sum(F.MSG_BYTES[m]*c for m,c in F.SERVICE_TEMPLATE[service].items())
                for u,v in zip(path[:-1],path[1:]):
                    k=tuple(sorted((u,v)));offered[k]=offered.get(k,0.)+cnt*bits/dt
        plans=self.poll_plans(t,failed_links)
        # Reserve offered bandwidth for the scheduled request and response.
        # Response reservation even for failed requests is conservative.
        for n,(path,size) in plans.items():
            for u,v in zip(path[:-1],path[1:]):
                k=tuple(sorted((u,v)));offered[k]=offered.get(k,0.)+8*(wire_bytes(50)+wire_bytes(size))/dt
        lstate={k:F.link_state(offered.get(k,0.),ls,self.mean_msg_bits) for k,ls in topo.linksets.items()}
        for k in failed_links:lstate[k]=(lstate[k][0],lstate[k][1],1.)
        self.plans=plans;self.offered=offered;self.failed=failed_links
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
            st=O.empty(n,t);cl=O.empty(n,t);tt=defaultdict(int);tm=defaultdict(int)
            ct=defaultdict(int);cm=defaultdict(int);hidden=0;removed=0;observable=0
            e=active.get(n)
            for path,i,service,cnt,closed,owner in records[n]:
                O.add_record(st,tt,tm,path,i,service,cnt,closed)
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
                if remain:O.add_record(cl,ct,cm,path,i,service,remain,cc)
            st.check_identities();cl.check_identities()
            true[n]=st;claimed[n]=cl;typed[n]=ct;mat[n]=cm
            # Explicit additional sensor: true offered counters sampled on
            # the domestic side of an external link, independently of node report.
            # Only sensor measurements, never this latent dictionary, reach features.
            boundary[n]={k:v for k,v in tt.items() if topo.nodes[topo.peer(k[1],n)].external}
            audit[n]=(hidden,removed,observable,max(st.total()-hidden,0))
        return true,claimed,typed,mat,boundary,audit,lstate



def transmit(path,size,lstate,topo,rng):
    # Whole report delivered only when every fragment traverses every hop.
    # Queue blocking is per fragment; no retries at this application layer.
    if not path:return False,np.nan
    prob=1.;delay=0.
    for u,v in zip(path[:-1],path[1:]):
        k=tuple(sorted((u,v)));rho,q,loss=lstate[k]
        prob*=max(0.,1.-loss)**fragments(size)
        # Conservative serial transmission of fragments; queue delay once
        # per fragment; independent exponential approximation.
        delay+=3.+8000.*wire_bytes(size)/topo.linksets[k].capacity_bps
        delay+=float(rng.gamma(fragments(size),max(q,0.001)))
    return bool(rng.random()<prob),delay

def make_episodes(sim,cfg,rng,balanced,stress=False):
    if not balanced:
        return F.make_episodes(sim,cfg,rng,n_compromise=0,n_normal=192 if stress else 128,n_ext=32 if stress else 20)
    eps=B.balanced("corrected",sim,cfg,rng,n_compromise=0,n_normal=128,n_ext=20)
    occupied=defaultdict(list)
    for e in eps:
        if e.kind=="compromise":occupied[e.nid].append((e.t0,e.t1))
    for kind in OTHER:
        nodes=[n for n in sim.topo.slaves if sim.topo.nodes[n].ntype.value==ATTACK_ROLE[kind]]
        for _ in range(25):
            length=int(rng.integers(60,121))
            for attempt in range(5000):
                n=int(rng.choice(nodes));a=int(rng.integers(WARMUP+5,cfg.n_intervals-length-5));b=a+length-1
                if all(b+3<c or a>d+3 for c,d in occupied[n]):break
            else:raise ValueError("cannot schedule conventional attack")
            occupied[n].append((a,b))
            eps.append(F.Episode(kind=kind,nid=n,t0=a,t1=b,mult=float(rng.uniform(*RANGES[kind]))))
    return eps

def generate(seed,balanced=True,template=None,stress=False):
    cfg=F.SimConfig(seed=seed,days=DAYS);book=F.RngBook(seed)
    sim=IntegratedSim(cfg,book,topo=template.topo if template else None,template=template)
    eps=make_episodes(sim,cfg,book["episode"],balanced,stress);ids={id(e):i for i,e in enumerate(eps)}
    cached={};last_rtt={};rows=[];transport=[];loads=[]
    for t in range(cfg.n_intervals):
        truth,claimed,typed,mat,boundary,audit,lstate=sim.step_true(t,eps)
        load_by_link=defaultdict(float)
        for path,size in sim.plans.values():
            for u,v in zip(path[:-1],path[1:]):load_by_link[tuple(sorted((u,v)))]+=8*(wire_bytes(50)+wire_bytes(size))/cfg.interval_s
        loads.append((t,sum(load_by_link.values()),max((v/sim.topo.linksets[k].capacity_bps for k,v in load_by_link.items()),default=0.)))
        rng=book.child("measure",t);trng=O.extra_rng(seed,2,t);brng=O.extra_rng(seed,3,t);crng=O.extra_rng(seed,5,t)
        active={e.nid:e for e in eps if e.kind in CLASSES and e.t0<=t<=e.t1}
        fresh={};held={};tc={};mc={};sensors={};cu={}
        for n,st in claimed.items():
            node=sim.topo.nodes[n]
            if node.external:continue
            e=active.get(n);comp=e is not None and e.kind=="compromise"
            keys=sim.topo.links_of(n)
            rho=float(np.mean([lstate[k][0] for k in keys])) if keys else 0.
            cap=float(np.mean([sim.topo.linksets[k].capacity_bps for k in keys])) if keys else 64000.
            r=F.render_report(st,rel_sigma=sim.rel_sigma,rho_path=rho,mean_msg_bits=sim.mean_msg_bits,
                lsx_cap_bps=cap,custody=cfg.key_custody,compromised=comp,t_compromise=e.t0 if comp else None,
                rotation=sim.rotation,background=sim.background,stale=False,rng=rng)
            ty=O.noisy(typed[n],trng,sim.rel_sigma);mx=O.noisy(mat[n],trng,sim.rel_sigma)
            if node.ntype==F.NodeType.IGW:sensors[n]=O.noisy(boundary[n],brng,sim.rel_sigma)
            if node.role==F.NodeRole.MASTER:
                fresh[n]=r;tc[n]=ty;mc[n]=mx;continue
            path,size=sim.plans[n]
            req,dr=transmit(path,50,lstate,sim.topo,crng)
            proc=float(max(.1,crng.normal(2.,.3))) if req else np.nan
            resp,ds=transmit(tuple(reversed(path)),size,lstate,sim.topo,crng) if req else (False,np.nan)
            rtt=dr+proc+ds if req and resp else np.nan
            delivered=bool(req and resp and rtt<=cfg.interval_s*1000.)
            if delivered:
                cached[n]=(r,ty,mx,t);fresh[n]=r;tc[n]=ty;mc[n]=mx
            if n in cached:
                pr,pt,pm,ts=cached[n];held[n]=(pr,pt,pm)
                age=t-ts
            else:age=t+1
            prev=last_rtt.get(n,np.nan);jitter=abs(rtt-prev) if delivered and np.isfinite(prev) else np.nan
            if delivered:last_rtt[n]=rtt
            cu[n]=dict(cu_delivered=float(delivered),cu_rtt_delay=rtt if delivered else np.nan,
                integrity_check=float(r.integrity_ok) if delivered else np.nan,
                cu_processing_delay=proc if delivered else np.nan,cu_response_time_jitter=jitter,
                cu_staleness=float(age),cu_consistency_error=float(abs(r.state.total()-sum(r.state.link_in.values())-sum(r.state.link_out.values()))) if delivered else np.nan)
            transport.append((t,n,int(delivered),int(req),size,wire_bytes(size),rtt if delivered else np.nan,
                len(path)-1 if path else -1,age))
        # Never compare counters from different timestamps. Lost reports
        # remain available for local carry-forward only, not current residuals.
        inv=report_only(fresh,sim.topo);di=directional(fresh,sim.topo);ex=O.telemetry(fresh,tc,mc,sensors,sim.topo)
        behavior=D.behavior_snapshot({n:x[0] for n,x in held.items()},
            {n:x[1] for n,x in held.items()},{n:x[2] for n,x in held.items()},sensors,sim.topo)
        for n in sim.topo.slaves:
            e=active.get(n);node=sim.topo.nodes[n];q={c:np.nan for c in F.OWN}
            q.update({c:np.nan for c in NEW+list(DIRECTION)+list(PEER)+list(O.EXTRA)+list(D.BEHAVIOR)})
            q.update({c:np.nan for c in ["share_"+m for m in MAP_CODES]+["international_fraction"]})
            if n in held:
                r,ty,mx=held[n];q.update(r.feats);st=r.state
                q.update({"share_"+m:float(st.mtype[F.MT_IDX[m]])/max(st.total(),1) for m in MAP_CODES})
                # External adjacency share, not exact international call fraction.
                q["international_fraction"]=sum(v for (_,k,j),v in ty.items() if sim.topo.nodes[sim.topo.peer(k,n)].external)/max(sum(ty.values()),1)
                q.update(behavior[n])
            q.update(inv.get(n,{}));q.update(di.get(n,{}));q.update(ex.get(n,{}));q.update(cu[n])
            rows.append(dict(nid=n,t=t,ntype=node.ntype.value,zone=node.zone,**q,
                node_type_enc=float(list(F.NodeType).index(node.ntype)),label=int(e is not None),
                episode_id=ids[id(e)] if e else -1,attack=e.kind if e else "normal"))
    d=pd.DataFrame(rows).sort_values(["nid","t"]).reset_index(drop=True)
    d=article_features(d)
    extras=NEW+list(DIRECTION)+list(PEER)+list(O.EXTRA)
    d=F.temporal_block(d,extras,cfg.feat_window)
    d=F.zone_deviation(d,extras,{n:v.zone for n,v in sim.topo.nodes.items()})
    d=add_history(d);d=D.history(d)
    d=d[d.t>=WARMUP].reset_index(drop=True)
    tr=pd.DataFrame(transport,columns=["t","nid","delivered","request","payload","wire","rtt","hops","age"])
    tr=tr[tr.t>=WARMUP]
    stats=tr.groupby("nid").agg(delivery=("delivered","mean"),request_delivery=("request","mean"),
        payload=("payload","mean"),wire=("wire","mean"),rtt=("rtt","median"),age=("age","mean")).reset_index()
    load=pd.DataFrame(loads,columns=["t","reserved_cu_bps","peak_link_cu_share"]);load=load[load.t>=WARMUP]
    stats["network_mean_reserved_cu_bps"]=load.reserved_cu_bps.mean()
    stats["network_peak_link_cu_share"]=load.peak_link_cu_share.max()
    return d,sim,eps,stats

def article_features(d):
    # Explicit reconstructed registry: article does not provide complete
    # exact list/order or original training parameters.
    g=d.groupby("nid",sort=False)
    def past(c,fn):
        r=g[c].rolling(30,min_periods=5,closed="left")
        return getattr(r,fn)().reset_index(level=0,drop=True).reindex(d.index)
    d["cu_combined_loss_prob"]=1.-past("cu_delivered","mean")
    d["integrity_fail_rate"]=1.-past("integrity_check","mean")
    for c in CV:d["cv_"+c]=past(c,"std")/(past(c,"mean").abs()+1e-6)
    for c in AC:
        x=d[c];lag=g[c].shift(1);prod=x*lag
        def roll(s):return s.groupby(d.nid,sort=False).rolling(30,min_periods=5,closed="left").mean().reset_index(level=0,drop=True).reindex(d.index)
        ma=roll(x);mb=roll(lag);va=(roll(x*x)-ma*ma).clip(lower=0);vb=(roll(lag*lag)-mb*mb).clip(lower=0)
        d["ac_"+c]=((roll(prod)-ma*mb)/np.sqrt(va*vb).replace(0,np.nan)).clip(-1,1)
    for c in ZONE:
        vals=d.groupby(["zone","t"],sort=False)[c].transform("median")
        d["zone_"+c]=d[c]-vals
    assert set(ARTICLE)<=set(d.columns)
    return d

def cols(d,name):
    extra=NEW+list(DIRECTION)+list(PEER)+list(O.EXTRA)
    noncu=list(F.OWN)+list(D.BEHAVIOR)+[c for c in d.columns if c.startswith("hist_") or any(c==b or c.startswith(b+"__") for b in extra)]
    if name=="expanded_no_cu":return list(dict.fromkeys(noncu))
    if name in ("expanded_cu","expanded_special"):return list(dict.fromkeys(list(ARTICLE)+noncu))
    return list(ARTICLE if name=="article_rf" else SPECIAL)

def fit_article(d,name,special=False):
    names=cols(d,name);x=d[names].replace([np.inf,-np.inf],np.nan);med=x.median().fillna(0.)
    y=(d.attack=="compromise").to_numpy(int) if special else d.label.to_numpy()
    est=RandomForestClassifier(n_estimators=150,max_depth=20,min_samples_leaf=3,n_jobs=2,random_state=16000,class_weight="balanced_subsample")
    est.fit(x.fillna(med),y)
    if not special:return (names,med,est)
    gb=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=3.,random_state=16000,class_weight="balanced")
    gb.fit(x.fillna(med),y)
    return (names,med,est,gb)

def fit_expanded(d,name):
    names=cols(d,name);y=(d.attack=="compromise").to_numpy(int) if name=="expanded_special" else d.label.to_numpy()
    p=y==1;ep=d.uid;w=np.ones(len(d));counts=ep[p].value_counts();w[p]=1./ep[p].map(counts).to_numpy()
    for k in (0,1):w[y==k]*=len(y)/(2*w[y==k].sum())
    return fit_model(d[names].to_numpy(float),y,w,names)

def train(out):
    parts=[];cu=[]
    for seed in TRAIN:
        d,sim,eps,tr=generate(seed);tr["seed"]=seed;cu.append(tr)
        d["uid"]=str(seed)+":"+d.episode_id.astype(str)
        pos=np.flatnonzero(d.label.to_numpy()==1);neg=np.flatnonzero(d.label.to_numpy()==0)
        ix=np.sort(np.r_[pos,np.random.default_rng(seed).choice(neg,min(50000,len(neg)),replace=False)])
        parts.append(d.iloc[ix][list(dict.fromkeys(cols(d,"expanded_cu")+["uid","label","attack"]))].copy())
        print("train",seed,len(pos),len(ix),"CU delivery",tr.delivery.mean(),flush=True)
    d=pd.concat(parts,ignore_index=True);models={"article_rf":fit_article(d,"article_rf"),
        "article_special":fit_article(d,"article_special",True)}
    for n in ("expanded_no_cu","expanded_cu","expanded_special"):models[n]=fit_expanded(d,n)
    joblib.dump(models,out/"models.joblib",compress=3);pd.concat(cu).to_csv(out/"train_cu.csv",index=False)

def predict(d,m,n):
    q=m[n]
    if n.startswith("article"):
        names,med,*est=q;x=d[names].replace([np.inf,-np.inf],np.nan).fillna(med)
        return np.mean([e.predict_proba(x)[:,1] for e in est],axis=0)
    return weak_predict(q,d[q["names"]].to_numpy(float))

def evaluate(seed,models,out,selection=None):
    assert seed in VAL+TEST
    val=seed in VAL
    if val!=(selection is None):raise ValueError("test requires frozen selection")
    d,sim,eps,tr=generate(seed);eligible={i:e for i,e in enumerate(eps) if e.kind in CLASSES}
    assert len(eligible)==200
    for i,e in eligible.items():
        q=d[(d.episode_id==i)&(d.nid==e.nid)]
        assert len(q)==e.t1-e.t0+1 and q.t.nunique()==len(q)
    refs=(generate(seed+10000,False,sim)[0],generate(seed+20000,False,sim,True)[0],
          generate(seed+40000,True,sim)[0])
    assert not refs[0].label.any() and not refs[1].label.any() and refs[2].label.any()
    frames=(*refs,d);ss={n:[predict(x,models,n) for x in frames] for n in models}
    # Article architecture OR under the same operational calibration policy.
    # Preserve original fixed .5 threshold as a descriptive secondary result.
    ss["article_cascade"]=[np.maximum(a,b) for a,b in zip(ss["article_rf"],ss["article_special"])]
    names=tuple(dict.fromkeys(BASE+(CANDIDATES if val else (selection["selected"],))))
    for n in names:
        if not n.startswith("dual_"):continue
        a,b=ss["expanded_cu"],ss["expanded_special"];share=int(n.split("_")[1])/100
        x,y,z=union(refs,a[:3],b[:3],share)
        ss[n]=[np.maximum(p-x,q-y)-z for p,q in zip(a,b)]
    names+=("article_cascade_fixed05",)
    ss["article_cascade_fixed05"]=ss["article_cascade"]
    rows=[];events=[];window=[];controls=[]
    y=d.label.to_numpy();t=d.t.to_numpy();nid=d.nid.to_numpy()
    for n in names:
        s=ss[n];tau=.5 if n.endswith("fixed05") else 0. if n.startswith("dual_") else joint_cut(refs,s[:3],RAW_BUDGET*.5,INCIDENT_BUDGET*.5)
        u=s[3];z=triple(d,u);m,det=metrics(d,u,z,tau,eligible);rows.append(dict(seed=seed,model=n,**m))
        for i,e in eligible.items():
            q=d[(d.episode_id==i)&(d.nid==e.nid)]
            events.append(dict(seed=seed,model=n,episode=i,detected=int(i in det),attack=e.kind,
                ntype=sim.topo.nodes[e.nid].ntype.value,sophistication=e.sophistication.value if e.kind=="compromise" else "na",
                hidden_rate=e.hidden_rate,mask_family=B.scenario_name("corrected",e) if e.kind=="compromise" else "na",
                delay_seconds=det.get(i),cu_delivery=q.cu_delivered.mean()))
        # Window-level recall reported separately from operational episode recall.
        for cls in CLASSES:
            mask=d.attack.to_numpy()==cls
            window.append(dict(seed=seed,model=n,attack=cls,positive_windows=int(mask.sum()),detected_windows=int((mask&(u>tau)).sum())))
        alerts=notify(d,z,tau);false_alerts=np.array([j for j in alerts if (y[j-2:j+1]==0).all()],dtype=int)
        for i,e in enumerate(eps):
            if e.kind in CLASSES:continue
            scope=(t>=max(e.t0,WARMUP))&(t<=e.t1)
            if e.kind!="ext_attack" and e.nid is not None:scope&=nid==e.nid
            neg=scope&(y==0)
            if neg.any():controls.append(dict(seed=seed,model=n,episode=i,event=e.kind,
                negative_windows=int(neg.sum()),false_windows=int((neg&(u>tau)).sum()),false_notifications=int(scope[false_alerts].sum())))
        print(seed,n,m["recall"],m["false_windows_day"],m["false_notifications_day"],flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)
    pd.DataFrame(window).to_csv(out/"window_metrics.csv",index=False);pd.DataFrame(controls).to_csv(out/"controls.csv",index=False)
    tr["seed"]=seed;tr.to_csv(out/"cu_transport.csv",index=False)
    print("CU delivery",tr.delivery.mean(),"median RTT",tr.rtt.median(),flush=True)

def attack_table(e):
    return e.groupby(["model","attack"]).agg(episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean"),
        delay=("delay_seconds","median"),cu_delivery=("cu_delivery","mean")).reset_index()

def select(root,out):
    m,e,s=collect(root,VAL,tuple(dict.fromkeys(BASE+CANDIDATES))+("article_cascade_fixed05",))
    a=attack_table(e);worst=a.groupby("model").recall.min().rename("worst_class_recall")
    q=s.merge(worst,on="model")
    rank=q[q.model.isin(CANDIDATES)].sort_values(["both_budgets","worst_class_recall","recall","worst_raw","model"],ascending=[False,False,False,True,True])
    chosen=str(rank.iloc[0].model);(out/"selection.json").write_text(json.dumps(dict(selected=chosen,rule="budget compliance, worst class, recall, worst raw"),indent=2))
    q.to_csv(out/"validation_summary.csv",index=False);a.to_csv(out/"validation_attacks.csv",index=False);print(q.to_string(index=False));print(a.to_string(index=False));print("SELECTED",chosen,flush=True)

def aggregate(root,out,selection):
    chosen=selection["selected"];m,e,s=collect(root,TEST,tuple(dict.fromkeys(BASE+(chosen,)))+("article_cascade_fixed05",))
    tr=pd.concat([pd.read_csv(x) for x in sorted(root.rglob("cu_transport.csv"))],ignore_index=True)
    a=attack_table(e);comp=e[e.attack=="compromise"].groupby(["model","sophistication","mask_family","hidden_rate","ntype"]).agg(
        episodes=("episode","size"),detected=("detected","sum"),recall=("detected","mean")).reset_index()
    def read(name):
        p=sorted(root.rglob(name+".csv"))
        if len(p)!=len(TEST):raise ValueError("incomplete "+name)
        return pd.concat([pd.read_csv(x) for x in p],ignore_index=True)
    win=read("window_metrics").groupby(["model","attack"])[["positive_windows","detected_windows"]].sum().reset_index()
    win["recall"]=win.detected_windows/win.positive_windows
    ctrl=read("controls").groupby(["model","event"])[["negative_windows","false_windows","false_notifications"]].sum().reset_index()
    ctrl["fpr"]=ctrl.false_windows/ctrl.negative_windows
    tests=[];rng=np.random.default_rng(16200)
    for cls in ("all",)+CLASSES:
        p=(e if cls=="all" else e[e.attack==cls]).groupby(["seed","model"]).detected.mean().unstack()
        for b in ("article_cascade","expanded_no_cu"):
            if chosen==b:continue
            v=(p[chosen]-p[b]).to_numpy();boot=rng.choice(v,size=(20000,len(v)),replace=True).mean(1)
            null=np.array([np.mean(v*np.array(q)) for q in itertools.product((-1,1),repeat=len(v))])
            tests.append(dict(endpoint=cls,a=chosen,b=b,delta=v.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),
                p_raw=np.mean(np.abs(null)>=abs(v.mean())-1e-12)))
    prev=0.
    for i,x in enumerate(sorted(tests,key=lambda z:z["p_raw"])):
        prev=max(prev,min(1.,(len(tests)-i)*x["p_raw"]));x["p_holm"]=prev
    for n,q in (("summary",s),("attacks",a),("compromise_factorial",comp),("paired",pd.DataFrame(tests)),("all_metrics",m),("episodes",e),("cu_transport",tr),("window_recall",win),("controls",ctrl)):
        q.to_csv(out/(n+".csv"),index=False)
    print("VALIDATION-SELECTED",chosen);print(s.to_string(index=False));print(a.to_string(index=False))
    print("CU TRANSPORT",tr[["delivery","request_delivery","payload","wire","rtt","age","network_mean_reserved_cu_bps","network_peak_link_cu_share"]].describe().to_string())
    print(pd.DataFrame(tests).to_string(index=False))
    print("Synthetic traffic-profile scenarios; article detector reconstructed, not exact original model.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["train","validate","select","test","aggregate"])
    p.add_argument("--seed",type=int);p.add_argument("--models",type=Path);p.add_argument("--input",type=Path)
    p.add_argument("--selection",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    selection=json.loads(a.selection.read_text()) if a.selection else None
    if a.mode=="train":train(a.output)
    elif a.mode=="select":select(a.input,a.output)
    elif a.mode=="aggregate":aggregate(a.input,a.output,selection)
    else:evaluate(a.seed,joblib.load(a.models),a.output,selection)
