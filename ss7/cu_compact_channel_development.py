"""Paired dense/compact CU transport and channel-wise calibration screen.
Synthetic interval-aggregate queues; no real network traffic or protocol I/O.
Dense and compact share topology, episode schedule and underlying traffic seeds.
Compact transmits grouped directional counts, relay marginals and local profiles.
"""
from __future__ import annotations
import argparse, json, math, os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
ROLE_ABUSE_V2=os.getenv("SS7_ROLE_ABUSE_V2","0")=="1"
FRESH_TEST_V3=os.getenv("SS7_FRESH_TEST_V3","0")=="1"
COMMISSIONING_V4=os.getenv("SS7_COMMISSIONING_V4","0")=="1"
TARGETED_ABUSE_V5=os.getenv("SS7_TARGETED_ABUSE_V5","0")=="1"
SUBSCRIBER_KIE_V6=os.getenv("SS7_SUBSCRIBER_KIE_V6","0")=="1"
SUBSCRIBER_KIE_V7=os.getenv("SS7_SUBSCRIBER_KIE_V7","0")=="1"
SUBSCRIBER_KIE_V8=os.getenv("SS7_SUBSCRIBER_KIE_V8","0")=="1"
SUBSCRIBER_KIE_V9=os.getenv("SS7_SUBSCRIBER_KIE_V9","0")=="1"
SUBSCRIBER_KIE_V10=os.getenv("SS7_SUBSCRIBER_KIE_V10","0")=="1"
SUBSCRIBER_KIE_V11=os.getenv("SS7_SUBSCRIBER_KIE_V11","0")=="1"
SUBSCRIBER_KIE_V12=os.getenv("SS7_SUBSCRIBER_KIE_V12","0")=="1"
SUBSCRIBER_KIE_V13=os.getenv("SS7_SUBSCRIBER_KIE_V13","0")=="1"
SUBSCRIBER_KIE_V14=os.getenv("SS7_SUBSCRIBER_KIE_V14","0")=="1"
SUBSCRIBER_KIE_V15=os.getenv("SS7_SUBSCRIBER_KIE_V15","0")=="1"
SUBSCRIBER_KIE_V16=os.getenv("SS7_SUBSCRIBER_KIE_V16","0")=="1"
SUBSCRIBER_KIE_V17=os.getenv("SS7_SUBSCRIBER_KIE_V17","0")=="1"
SUBSCRIBER_KIE_V18=os.getenv("SS7_SUBSCRIBER_KIE_V18","0")=="1"
SUBSCRIBER_KIE_V19=os.getenv("SS7_SUBSCRIBER_KIE_V19","0")=="1"
SUBSCRIBER_KIE_V22=os.getenv("SS7_SUBSCRIBER_KIE_V22","0")=="1"
CAPABILITY_KIE_V23=os.getenv("SS7_CAPABILITY_KIE_V23","0")=="1"
SUBSCRIBER_SKETCH=SUBSCRIBER_KIE_V6 or SUBSCRIBER_KIE_V7 or SUBSCRIBER_KIE_V8 or SUBSCRIBER_KIE_V9 or SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19 or SUBSCRIBER_KIE_V22
if CAPABILITY_KIE_V23:
    # v23 adds an independently verified authorization/capability invariant
    # for high-risk MAP operations. Only aggregate counters enter KIE.
    TRAIN=(41001,41002,41003)
    RANK=(41101,41102,41103)
    CAL=(41201,41202)
    TEST=(41301,41302)
elif SUBSCRIBER_KIE_V22:
    # v22 replaces the dense 32-bin wire vector with a compact heavy-hitter
    # summary. Local keyed hash resolution is 128 bins; only total plus top-4
    # pseudonymous IDs/counts per peer x exact MAP service leave the node.
    TRAIN=(40001,40002,40003)
    RANK=(40101,40102,40103)
    CAL=(40201,40202)
    TEST=(40301,40302)
elif SUBSCRIBER_KIE_V19:
    # v19 keeps the same 32-bin uint16 peer x exact-service KIE wire sketch.
    # It exposes a compact multiview set of causal identity summaries for a
    # low-capacity, temporally thinned linear detector. Fresh TRAIN/RANK only;
    # CAL/TEST remain untouched until the detector is frozen.
    TRAIN=(39001,39002,39003)
    RANK=(39101,39102,39103)
    CAL=(39201,39202)
    TEST=(39301,39302)
elif SUBSCRIBER_KIE_V18:
    # v18 keeps the v15-v17 wire sketch but discards absolute amplitude after
    # residual formation. It measures persistence of the SAME identity among
    # cross-sectional top residuals. Fresh RANK only; CAL/TEST untouched.
    TRAIN=(38001,38002,38003)
    RANK=(38101,38102,38103)
    CAL=(38201,38202)
    TEST=(38301,38302)
elif SUBSCRIBER_KIE_V17:
    # v17 uses the same v15/v16 32-bin uint16 KIE on the wire, but calibrates
    # each exact peer x service x keyed-bin against its own commissioning tail
    # and requires persistent tail evidence before max reduction.
    # Fresh RANK-only development; CAL/TEST remain untouched.
    TRAIN=(37001,37002,37003)
    RANK=(37101,37102,37103)
    CAL=(37201,37202)
    TEST=(37301,37302)
elif SUBSCRIBER_KIE_V16:
    # v16 preserves peer x exact-service x keyed-bin identity through both
    # raw-count aggregation and short temporal confirmation. Same v15 wire.
    # Fresh RANK-only development; CAL/TEST remain untouched.
    TRAIN=(36001,36002,36003)
    RANK=(36101,36102,36103)
    CAL=(36201,36202)
    TEST=(36301,36302)
elif SUBSCRIBER_KIE_V15:
    # v15 combines the corrected service-stable benign hash population of v14
    # with the fixed 15-minute raw-count aggregation of v11.
    # Fresh RANK-only development; CAL/TEST remain untouched.
    TRAIN=(35001,35002,35003)
    RANK=(35101,35102,35103)
    CAL=(35201,35202)
    TEST=(35301,35302)
elif SUBSCRIBER_KIE_V14:
    # v14 corrects the benign keyed-hash population model while keeping the
    # v13 wire sketch and identity-preserving uncapped CUSUM unchanged.
    # Fresh RANK-only development; CAL/TEST remain untouched.
    TRAIN=(34001,34002,34003)
    RANK=(34101,34102,34103)
    CAL=(34201,34202)
    TEST=(34301,34302)
elif SUBSCRIBER_KIE_V13:
    # v13 is the v12 identity-preserving CUSUM with the artificial state cap
    # removed. Fresh RANK-only development; CAL/TEST remain untouched.
    TRAIN=(33001,33002,33003)
    RANK=(33101,33102,33103)
    CAL=(33201,33202)
    TEST=(33301,33302)
elif SUBSCRIBER_KIE_V12:
    # v12 preserves exact peer x MAP-service x keyed-bin identity until after
    # a fixed per-bin CUSUM. Fresh RANK-only development; CAL/TEST untouched.
    TRAIN=(32001,32002,32003)
    RANK=(32101,32102,32103)
    CAL=(32201,32202)
    TEST=(32301,32302)
elif SUBSCRIBER_KIE_V11:
    # v11 keeps the v10 32-bin uint16 wire sketch, but accumulates raw
    # peer x service x bin counts for 15 minutes before residual formation.
    # RANK-only development; CAL/TEST remain untouched.
    TRAIN=(31001,31002,31003)
    RANK=(31101,31102,31103)
    CAL=(31201,31202)
    TEST=(31301,31302)
elif SUBSCRIBER_KIE_V10:
    # v10 doubles hash resolution while keeping v9 wire bytes constant by
    # encoding per-window counters as saturating uint16 instead of uint32.
    TRAIN=(30001,30002,30003)
    RANK=(30101,30102,30103)
    CAL=(30201,30202)
    TEST=(30301,30302)
elif SUBSCRIBER_KIE_V9:
    # v9 conditions the keyed subscriber sketch on adjacent peer and exact
    # MAP service. Fresh RANK-only development; CAL/TEST remain untouched.
    TRAIN=(28001,28002,28003)
    RANK=(28101,28102,28103)
    CAL=(28201,28202)
    TEST=(28301,28302)
elif SUBSCRIBER_KIE_V8:
    # v8 conditions the privacy-preserving subscriber sketch on the adjacent
    # SS7 peer. RANK-only development; TEST is frozen and untouched.
    TRAIN=(27001,27002,27003)
    RANK=(27101,27102,27103)
    CAL=(27201,27202)
    TEST=(27301,27302)
elif SUBSCRIBER_KIE_V7:
    # v7 keeps the exact v6 wire sketch but replaces raw concentration scores
    # with node-local commissioning residuals. TEST is frozen and untouched.
    TRAIN=(26001,26002,26003)
    RANK=(26101,26102,26103)
    CAL=(26201,26202)
    TEST=(26301,26302)
elif SUBSCRIBER_KIE_V6:
    # v6 adds privacy-preserving subscriber hash sketches to compact KIE.
    # TEST remains frozen and must not be generated during RANK development.
    TRAIN=(25001,25002,25003)
    RANK=(25101,25102,25103)
    CAL=(25201,25202)
    TEST=(25301,25302)
elif TARGETED_ABUSE_V5:
    # Frozen before generation or inspection of the v5 TEST worlds.
    TRAIN=(24001,24002,24003)
    RANK=(24101,24102,24103)
    CAL=(24201,24202)
    TEST=(24301,24302)
elif COMMISSIONING_V4:
    # Frozen before generation or inspection of the v4 TEST worlds.
    TRAIN=(23001,23002,23003)
    RANK=(23101,23102,23103)
    CAL=(23201,23202)
    TEST=(23301,23302)
elif ROLE_ABUSE_V2:
    # Development/calibration worlds remain frozen. Each detector revision gets
    # a new untouched holdout instead of reusing a previously inspected TEST.
    TRAIN=(22001,22002,22003)
    RANK=(22101,22102)
    CAL=(22201,22202)
    TEST=(22401,22402) if FRESH_TEST_V3 else (22301,22302)
else:
    TRAIN=(21001,21002,21003)
    RANK=(21101,21102)
    CAL=(21201,21202)
    TEST=(21301,21302)
DAYS=1.
# Optional deployment commissioning period. Default preserves historical worlds;
# development workflows may reserve a longer attack-free baseline explicitly.
DEPLOY_BASELINE=int(os.getenv("SS7_DEPLOY_BASELINE",str(WARMUP+5)))
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
KIE_GROUP_SKETCH_BINS=16
KIE_GROUP_SKETCH=tuple(f"kie_group_route_sketch_{i}" for i in range(KIE_GROUP_SKETCH_BINS))
KIE_GROUP_DERIVED=("kie_group_route_js","kie_group_route_maxshift")
DRIFT_SOURCE=("total_messages","map_share","isup_share","n_unique_dest","dest_entropy","dest_top1_ratio","in_out_ratio","dlg_closure_ratio","transit_share")+SUB
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
SUB_HASH_BINS=128 if SUBSCRIBER_KIE_V22 else (32 if (SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19) else 16)
SUB_PROFILE_NAMES=tuple(
    f"kie_sub_g{g}_{k}" for g in range(len(GROUPS))
    for k in ("top1","entropy","occupied")
)
SUB_BIN_NAMES=tuple(
    f"kie_sub_g{g}_b{b}" for g in range(len(GROUPS))
    for b in range(SUB_HASH_BINS)
)
SUB_RESID_NAMES=tuple(
    f"kie_sub_g{g}_{k}" for g in range(len(GROUPS))
    for k in ("max_resid","rms_resid","chi2")
)
SUB_PACKED_NAME="__kie_sub_bins_packed"
SUB_PEER_PACKED_NAME="__kie_peer_sub_bins_packed"
SUB_PEER_RESID_NAMES=("kie_peer_sub_max_resid","kie_peer_sub_rms_resid","kie_peer_sub_chi2")
KIE_SERVICES=tuple(s for s in F.Service if s in F.TCAP_SERVICES)
KIE_SERVICE_INDEX={s:i for i,s in enumerate(KIE_SERVICES)}
SUB_PEER_SERVICE_PACKED_NAME="__kie_peer_service_sub_bins_packed"
SUB_PEER_SERVICE_RESID_NAMES=("kie_peer_service_sub_max_resid","kie_peer_service_sub_rms_resid","kie_peer_service_sub_chi2")
SUB_PEER_SERVICE_ROLL30_NAME="kie_peer_service_roll30_max_resid"
SUB_PEER_SERVICE_IDMEAN5_NAME="kie_peer_service_samebin_mean5"
SUB_PEER_SERVICE_IDMIN3_NAME="kie_peer_service_samebin_min3"
SUB_PEER_SERVICE_TAILAREA5_NAME="kie_peer_service_tail3of5_area"
SUB_PEER_SERVICE_TAILCOUNT5_NAME="kie_peer_service_tail3of5_count"
SUB_PEER_SERVICE_TOP1PERSIST10_NAME="kie_peer_service_top1_persist10"
SUB_PEER_SERVICE_TOP5PERSIST10_NAME="kie_peer_service_top5_persist10"
SUB_MULTIVIEW_NAMES=(
    "kie_mv_max","kie_mv_top5mean","kie_mv_top10mean","kie_mv_margin12",
    "kie_mv_samebin_mean5","kie_mv_samebin_min3",
    "kie_mv_top1_persist10","kie_mv_top5_persist10"
)
SUB_PEER_SERVICE_BIN_CUSUM_NAME="kie_peer_service_bin_cusum"
SUB_PEER_SERVICE_BIN_CUSUM_UNCAPPED_NAME="kie_peer_service_bin_cusum_uncapped"
HH_TOPK=4
CAP_BENIGN_FAIL=2e-5
CAP_KIE_NAMES=("kie_cap_checked","kie_cap_missing","kie_cap_missing_rate")
CAP_DERIVED_NAMES=("kie_cap_missing_5m","kie_cap_missing_rate_5m")
SUB_HH_PACKED_NAME="__kie_hh_packed"
SUB_HH_NAMES=(
    "kie_hh_top1_share","kie_hh_top4_share","kie_hh_margin12",
    "kie_hh_sameid_mean5","kie_hh_sameid_min3","kie_hh_persist10"
)
SUB_ROLL_WINDOW=30
SUB_ROLL_MIN=15
SUB_BIN_CUSUM_K=0.5
SUB_BIN_CUSUM_LEAK=0.98
SUB_BIN_CUSUM_CAP=50.0
SUB_BIN_CUSUM_CAP_UNCAPPED=1_000_000.0
SERVICE_GROUP={}
for _service in F.Service:
    _gs={GROUP[F.MT_IDX[m]] for m in F.SERVICE_TEMPLATE[_service]}
    assert len(_gs)==1,(_service,_gs)
    SERVICE_GROUP[_service]=next(iter(_gs))
def service_group(service):
    return SERVICE_GROUP[service]
def wire_bytes(size):
    return int(size+12*math.ceil(size/256))
def payload_bytes(topo,n,compact):
    d=len(topo.links_of(n));m=len(F.MSG_TYPES);N=len(topo.nodes)
    relay=topo.nodes[n].ntype in (F.NodeType.STP,F.NodeType.IGW)
    if not compact:return 50+8*(m+2*d+N+8+2*d*m+d*d*m)
    # Fixed-layout envelope. Base state, four-group directional counts,
    # four-group matrix marginals for relays and two eight-float64 profiles.
    # Subscriber KIE carries keyed uint32 counters only; raw subscriber IDs
    # never leave the node. v8 conditions the same sketch on the adjacent peer.
    if SUBSCRIBER_KIE_V22:
        # total:uint16 + top4*(hash_id:uint8 + count:uint16) = 14 B/cell.
        sub_bytes=14*d*len(KIE_SERVICES)
    elif SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19:
        # 32 bins x uint16 == 16 bins x uint32: same nominal v9 wire cost.
        sub_bytes=2*d*len(KIE_SERVICES)*SUB_HASH_BINS
    elif SUBSCRIBER_KIE_V9:
        sub_bytes=4*d*len(KIE_SERVICES)*SUB_HASH_BINS
    elif SUBSCRIBER_KIE_V8:
        sub_bytes=4*d*len(GROUPS)*SUB_HASH_BINS
    else:
        sub_bytes=(4*len(GROUPS)*SUB_HASH_BINS) if SUBSCRIBER_SKETCH else 0
    cap_bytes=8 if CAPABILITY_KIE_V23 else 0
    return 50+4*(m+2*d+N+8+2*d*len(GROUPS)+(2*d*len(GROUPS) if relay else 0))+16*8+sub_bytes+cap_bytes

def noisy_subscriber_sketch(a,rng,sigma):
    a=np.asarray(a,float)
    z=np.rint(a*(1.0+rng.normal(0.0,sigma,size=a.shape)))
    return np.maximum(z,0.0).astype(np.int64)

def subscriber_profile(a):
    a=np.asarray(a,float);out={}
    if SUBSCRIBER_KIE_V22:
        assert a.ndim==3 and a.shape[1:]==(len(KIE_SERVICES),SUB_HASH_BINS)
        peers=a.shape[0]
        packed=np.zeros((peers,len(KIE_SERVICES),1+2*HH_TOPK),dtype=np.float32)
        for pi in range(peers):
            for si in range(len(KIE_SERVICES)):
                x=np.maximum(a[pi,si],0.0)
                total=float(x.sum());packed[pi,si,0]=min(total,65535.0)
                if total<=0:continue
                k=min(HH_TOPK,len(x))
                ids=np.argpartition(x,-k)[-k:]
                ids=ids[np.argsort(x[ids])[::-1]]
                for j,b in enumerate(ids):
                    packed[pi,si,1+2*j]=float(b)
                    packed[pi,si,2+2*j]=min(float(x[b]),65535.0)
        out[SUB_HH_PACKED_NAME]=packed.reshape(-1)
        return out
    if SUBSCRIBER_KIE_V9 or SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19 or SUBSCRIBER_KIE_V22:
        assert a.ndim==3 and a.shape[1:]==(len(KIE_SERVICES),SUB_HASH_BINS)
        out[SUB_PEER_SERVICE_PACKED_NAME]=a.astype(np.float32,copy=False).reshape(-1).copy()
        return out
    if SUBSCRIBER_KIE_V8:
        assert a.ndim==3 and a.shape[1:]==(len(GROUPS),SUB_HASH_BINS)
        out[SUB_PEER_PACKED_NAME]=a.astype(np.float32,copy=False).reshape(-1).copy()
        return out
    assert a.shape==(len(GROUPS),SUB_HASH_BINS)
    if SUBSCRIBER_KIE_V7:
        # Internal representation only: the wire format is still exactly the
        # same 4 x 16 uint32 KIE sketch. Packing avoids millions of Python dict
        # insertions during simulation; residuals are expanded once afterwards.
        out[SUB_PACKED_NAME]=a.astype(np.float32,copy=False).reshape(-1).copy()
        return out
    for g in range(len(GROUPS)):
        x=a[g];tot=float(x.sum())
        # v6 development keeps scalar bins for rolling-audit reproducibility.
        for b,v in enumerate(x):
            out[f"kie_sub_g{g}_b{b}"]=float(v)
        if tot<=0:
            top1=entropy=occupied=0.0
        else:
            p=x/tot
            top1=float(p.max())
            nz=p[p>0]
            entropy=float(-(nz*np.log2(nz)).sum()/np.log2(SUB_HASH_BINS))
            occupied=float(np.count_nonzero(x)/SUB_HASH_BINS)
        out[f"kie_sub_g{g}_top1"]=top1
        out[f"kie_sub_g{g}_entropy"]=entropy
        out[f"kie_sub_g{g}_occupied"]=occupied
    return out

def subscriber_residual_features(d):
    """Derive zero-extra-wire subscriber residuals from frozen commissioning data."""
    if not SUBSCRIBER_KIE_V7:
        for c in SUB_RESID_NAMES:d[c]=np.nan
        return d

    nrow=len(d)
    result={c:np.full(nrow,np.nan,dtype=float) for c in SUB_RESID_NAMES}
    if SUB_PACKED_NAME not in d.columns:
        for c,v in result.items():d[c]=v
        return d

    # Process each node as dense NumPy arrays. This replaces repeated pandas
    # column slicing/loc writes in the hot path and removes the packed sketch
    # before the rest of feature engineering.
    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy()
        tt=gdf.t.to_numpy()
        obj=gdf[SUB_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size==len(GROUPS)*SUB_HASH_BINS for v in obj],dtype=bool)
        if not valid.any():continue
        X=np.full((len(gdf),len(GROUPS),SUB_HASH_BINS),np.nan,dtype=float)
        X[valid]=np.stack(obj[valid]).reshape((-1,len(GROUPS),SUB_HASH_BINS))
        warm=valid&(tt<DEPLOY_BASELINE)
        if warm.sum()<20:continue

        for grp in range(len(GROUPS)):
            A=X[:,grp,:]
            H=np.maximum(A[warm],0.0)
            agg=H.sum(axis=0);alpha=1.0
            p=(agg+alpha)/(agg.sum()+alpha*SUB_HASH_BINS)
            def pearson(B):
                B=np.maximum(np.asarray(B,float),0.0)
                N=B.sum(axis=1,keepdims=True)
                E=N*p[None,:]
                var=E*(1.0-p[None,:])+(0.10*E)**2+1.0
                return (B-E)/np.sqrt(var)
            Zw=pearson(H)
            center=np.median(Zw,axis=0)
            q25,q75=np.quantile(Zw,[.25,.75],axis=0)
            scale=np.maximum((q75-q25)/1.349,1.0)
            Z=np.zeros_like(A,dtype=float)
            Z[valid]=(pearson(A[valid])-center[None,:])/scale[None,:]
            P=np.maximum(Z,0.0)
            result[f"kie_sub_g{grp}_max_resid"][ix]=np.where(valid,np.arcsinh(np.max(P,axis=1)),np.nan)
            result[f"kie_sub_g{grp}_rms_resid"][ix]=np.where(valid,np.arcsinh(np.sqrt(np.mean(P*P,axis=1))),np.nan)
            result[f"kie_sub_g{grp}_chi2"][ix]=np.where(valid,np.arcsinh(np.sqrt(np.sum(P*P,axis=1))),np.nan)

    for c,v in result.items():d[c]=v
    return d.drop(columns=[SUB_PACKED_NAME])

def peer_subscriber_residual_features(d):
    """Commissioning-normalized residuals for peer-conditioned subscriber KIE.

    Peer slots are stable within a node across the run. Only keyed hash-bin
    counts are processed; no subscriber identifier or attack label is used.
    """
    for c in SUB_PEER_RESID_NAMES:d[c]=np.nan
    if not SUBSCRIBER_KIE_V8:return d
    if SUB_PEER_PACKED_NAME not in d.columns:return d

    nrow=len(d)
    result={c:np.full(nrow,np.nan,dtype=float) for c in SUB_PEER_RESID_NAMES}
    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();tt=gdf.t.to_numpy()
        obj=gdf[SUB_PEER_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and
                        v.size%(len(GROUPS)*SUB_HASH_BINS)==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size)
        peers=width//(len(GROUPS)*SUB_HASH_BINS)
        X=np.full((len(gdf),peers,len(GROUPS),SUB_HASH_BINS),np.nan,dtype=float)
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if ok.any():
            X[ok]=np.stack(obj[ok]).reshape((-1,peers,len(GROUPS),SUB_HASH_BINS))
        warm=ok&(tt<DEPLOY_BASELINE)
        if warm.sum()<20:continue

        best=np.zeros((len(gdf),3),float)
        any_cell=np.zeros(len(gdf),bool)
        for pi in range(peers):
            for grp in range(len(GROUPS)):
                A=X[:,pi,grp,:]
                H=np.maximum(A[warm],0.0)
                agg=H.sum(axis=0)
                if agg.sum()<=0:continue
                alpha=1.0;p=(agg+alpha)/(agg.sum()+alpha*SUB_HASH_BINS)
                def pearson(B):
                    B=np.maximum(np.asarray(B,float),0.0)
                    N=B.sum(axis=1,keepdims=True);E=N*p[None,:]
                    var=E*(1.0-p[None,:])+(0.10*E)**2+1.0
                    return (B-E)/np.sqrt(var)
                Zw=pearson(H);center=np.median(Zw,axis=0)
                q25,q75=np.quantile(Zw,[.25,.75],axis=0)
                scale=np.maximum((q75-q25)/1.349,1.0)
                cell=np.isfinite(A).all(axis=1)
                Z=np.zeros_like(A,dtype=float)
                if cell.any():Z[cell]=(pearson(A[cell])-center[None,:])/scale[None,:]
                P=np.maximum(Z,0.0)
                vals=np.column_stack((
                    np.arcsinh(np.max(P,axis=1)),
                    np.arcsinh(np.sqrt(np.mean(P*P,axis=1))),
                    np.arcsinh(np.sqrt(np.sum(P*P,axis=1)))))
                best=np.maximum(best,vals);any_cell|=cell
        best[~any_cell]=np.nan
        for k,c in enumerate(SUB_PEER_RESID_NAMES):result[c][ix]=best[:,k]

    for c,v in result.items():d[c]=v
    return d.drop(columns=[SUB_PEER_PACKED_NAME])


def peer_service_subscriber_bin_cusum_features(d):
    """Fixed per-bin sequential detector for peer x exact-service KIE v12.

    Each keyed subscriber bin keeps its identity through commissioning
    normalization and CUSUM accumulation. The max across bins/cells is taken
    only after the sequential state is updated, preventing unrelated benign
    extrema from accumulating as one synthetic attack. No label, CAL or TEST
    information enters the feature.
    """
    if not (SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14):return d
    feature_name=(SUB_PEER_SERVICE_BIN_CUSUM_UNCAPPED_NAME
                  if (SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14)
                  else SUB_PEER_SERVICE_BIN_CUSUM_NAME)
    d[feature_name]=np.nan
    if SUB_PEER_SERVICE_PACKED_NAME not in d.columns:return d

    state_cap=(SUB_BIN_CUSUM_CAP_UNCAPPED
               if (SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14)
               else SUB_BIN_CUSUM_CAP)
    out=np.full(len(d),np.nan,dtype=float)
    cell_width=len(KIE_SERVICES)*SUB_HASH_BINS
    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();tt=gdf.t.to_numpy()
        obj=gdf[SUB_PEER_SERVICE_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and
                        v.size%cell_width==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size)
        peers=width//cell_width
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if not ok.any():continue

        X=np.zeros((len(gdf),peers,len(KIE_SERVICES),SUB_HASH_BINS),dtype=float)
        X[ok]=np.stack(obj[ok]).reshape((-1,peers,len(KIE_SERVICES),SUB_HASH_BINS))
        warm=ok&(tt<DEPLOY_BASELINE)
        if warm.sum()<40:continue

        best=np.full(len(gdf),np.nan,dtype=float)
        have=np.zeros(len(gdf),dtype=bool)
        for pi in range(peers):
            for si in range(len(KIE_SERVICES)):
                A=np.maximum(X[:,pi,si,:],0.0)
                H=A[warm]
                agg=H.sum(axis=0)
                if agg.sum()<=0:continue
                alpha=1.0
                p=(agg+alpha)/(agg.sum()+alpha*SUB_HASH_BINS)

                def pearson(B):
                    B=np.maximum(np.asarray(B,float),0.0)
                    N=B.sum(axis=1,keepdims=True)
                    E=N*p[None,:]
                    var=E*(1.0-p[None,:])+1.0
                    return (B-E)/np.sqrt(var)

                Zw=pearson(H)
                center=np.median(Zw,axis=0)
                q25,q75=np.quantile(Zw,[.25,.75],axis=0)
                scale=np.maximum((q75-q25)/1.349,1.0)
                Z=(pearson(A)-center[None,:])/scale[None,:]
                totals=A.sum(axis=1)

                state=np.zeros(SUB_HASH_BINS,dtype=float)
                cell_score=np.full(len(gdf),np.nan,dtype=float)
                for j in range(len(gdf)):
                    # Commissioning initializes the baseline only; sequential
                    # evidence starts cleanly at deployment.
                    if tt[j]<DEPLOY_BASELINE:
                        state.fill(0.0)
                        continue
                    if not ok[j] or totals[j]<=0 or not np.isfinite(Z[j]).all():
                        state.fill(0.0)
                        continue
                    state=np.clip(SUB_BIN_CUSUM_LEAK*state+Z[j]-SUB_BIN_CUSUM_K,
                                  0.0,state_cap)
                    cell_score[j]=np.arcsinh(float(state.max()))
                finite=np.isfinite(cell_score)
                if finite.any():
                    cur=np.where(np.isfinite(best),best,-np.inf)
                    cur[finite]=np.maximum(cur[finite],cell_score[finite])
                    best[finite]=cur[finite];have|=finite
        best[~have]=np.nan
        out[ix]=best

    d[feature_name]=out
    return d.drop(columns=[SUB_PEER_SERVICE_PACKED_NAME])


def heavy_hitter_subscriber_features(d):
    """v22 causal features from compact top-4 keyed subscriber identities.

    Identity is preserved as (peer slot, exact MAP service, keyed hash id).
    No raw subscriber identifier or ground-truth attack field is used.
    """
    for c in SUB_HH_NAMES:d[c]=np.nan
    if not SUBSCRIBER_KIE_V22:return d
    if SUB_HH_PACKED_NAME not in d.columns:return d

    out={c:np.full(len(d),np.nan,dtype=float) for c in SUB_HH_NAMES}
    width_cell=1+2*HH_TOPK
    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();n=len(gdf)
        obj=gdf[SUB_HH_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and
                        v.size%(len(KIE_SERVICES)*width_cell)==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size)
        cells=width//width_cell
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if not ok.any():continue

        X=np.zeros((n,cells,width_cell),dtype=float)
        X[ok]=np.stack(obj[ok]).reshape((-1,cells,width_cell))
        total=X[:,:,0]
        ids=np.rint(X[:,:,1::2]).astype(np.int16)
        counts=np.maximum(X[:,:,2::2],0.0)
        share=np.divide(counts,total[:,:,None],out=np.zeros_like(counts),
                        where=total[:,:,None]>0)
        cell_valid=ok[:,None]&(total>0)

        top1=np.where(cell_valid,share[:,:,0],np.nan)
        top4=np.where(cell_valid,share.sum(axis=2),np.nan)
        margin=np.where(cell_valid,share[:,:,0]-share[:,:,1],np.nan)
        with np.errstate(all="ignore"):
            out["kie_hh_top1_share"][ix]=np.nanmax(top1,axis=1)
            out["kie_hh_top4_share"][ix]=np.nanmax(top4,axis=1)
            out["kie_hh_margin12"][ix]=np.nanmax(margin,axis=1)

        # Same top-K pseudonymous identity across causal windows. Vectorized
        # over time/cell/identity so 128-bin local resolution remains practical.
        mean5=np.full(n,np.nan);min3=np.full(n,np.nan);persist=np.full(n,np.nan)

        if n>=3:
            cur_ids=ids[2:];cur_share=share[2:];cur_valid=(counts[2:]>0)&ok[2:,None,None]
            vals=[cur_share];valid3=cur_valid.copy()
            for lag in (1,2):
                prv_ids=ids[2-lag:n-lag];prv_share=share[2-lag:n-lag]
                prv_count=counts[2-lag:n-lag]
                match=(cur_ids[:,:,:,None]==prv_ids[:,:,None,:])&(prv_count[:,:,None,:]>0)
                got=np.max(np.where(match,prv_share[:,:,None,:],0.0),axis=3)
                valid3&=(got>0)&ok[2-lag:n-lag,None,None]
                vals.append(got)
            sc=np.where(valid3,np.minimum.reduce(vals),-np.inf)
            best=np.max(sc,axis=(1,2));good=np.isfinite(best)
            tmp=min3[2:];tmp[good]=best[good];min3[2:]=tmp

        if n>=5:
            cur_ids=ids[4:];acc=share[4:].copy()
            valid5=(counts[4:]>0)&ok[4:,None,None]
            for lag in (1,2,3,4):
                prv_ids=ids[4-lag:n-lag];prv_share=share[4-lag:n-lag]
                prv_count=counts[4-lag:n-lag]
                match=(cur_ids[:,:,:,None]==prv_ids[:,:,None,:])&(prv_count[:,:,None,:]>0)
                got=np.max(np.where(match,prv_share[:,:,None,:],0.0),axis=3)
                valid5&=(got>0)&ok[4-lag:n-lag,None,None]
                acc+=got
            sc=np.where(valid5,acc/5.0,-np.inf)
            best=np.max(sc,axis=(1,2));good=np.isfinite(best)
            tmp=mean5[4:];tmp[good]=best[good];mean5[4:]=tmp

        if n>=10:
            cur_ids=ids[9:];cur_valid=(counts[9:]>0)&ok[9:,None,None]
            hits=cur_valid.astype(np.int16)
            den=np.zeros(n-9,dtype=np.int16)
            for lag in range(10):
                den+=ok[9-lag:n-lag].astype(np.int16)
                if lag==0:continue
                prv_ids=ids[9-lag:n-lag];prv_count=counts[9-lag:n-lag]
                match=(cur_ids[:,:,:,None]==prv_ids[:,:,None,:])&(prv_count[:,:,None,:]>0)
                present=np.any(match,axis=3)&ok[9-lag:n-lag,None,None]
                hits+=present.astype(np.int16)
            score=np.divide(hits,den[:,None,None],out=np.zeros_like(hits,dtype=float),
                            where=den[:,None,None]>0)
            validp=cur_valid&(den[:,None,None]>=7)
            sc=np.where(validp,score,-np.inf)
            best=np.max(sc,axis=(1,2));good=np.isfinite(best)
            tmp=persist[9:];tmp[good]=best[good];persist[9:]=tmp

        out["kie_hh_sameid_mean5"][ix]=mean5
        out["kie_hh_sameid_min3"][ix]=min3
        out["kie_hh_persist10"][ix]=persist

    for c,v in out.items():d[c]=v
    return d.drop(columns=[SUB_HH_PACKED_NAME])


def peer_service_subscriber_multiview_features(d):
    """v19 compact multiview identity summaries for low-capacity ML.

    All summaries are causal and derived from the exact same KIE wire sketch as
    v15-v18. The key distinction is that no single hand-crafted scalar is
    expected to carry the decision. Amplitude, cross-sectional concentration,
    same-bin persistence and identity-rank persistence are retained together.
    """
    for c in SUB_MULTIVIEW_NAMES:d[c]=np.nan
    if not SUBSCRIBER_KIE_V19:return d
    if SUB_PEER_SERVICE_PACKED_NAME not in d.columns:return d

    out={c:np.full(len(d),np.nan,dtype=float) for c in SUB_MULTIVIEW_NAMES}
    cell_width=len(KIE_SERVICES)*SUB_HASH_BINS
    rank_window=10;rank_min=7;topk=5

    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();tt=gdf.t.to_numpy();n=len(gdf)
        obj=gdf[SUB_PEER_SERVICE_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and
                        v.size%cell_width==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size)
        cells=width//SUB_HASH_BINS
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if not ok.any():continue

        X=np.zeros((n,cells,SUB_HASH_BINS),dtype=float)
        X[ok]=np.stack(obj[ok]).reshape((-1,cells,SUB_HASH_BINS))

        # v15-style causal 30-interval raw-count accumulation.
        flat=X.reshape((n,-1))
        cs=np.vstack([np.zeros((1,flat.shape[1])),np.cumsum(flat,axis=0)])
        cc=np.r_[0,np.cumsum(ok.astype(np.int64))]
        jj=np.arange(n,dtype=np.int64);aa=np.maximum(0,jj-SUB_ROLL_WINDOW+1)
        R=(cs[jj+1]-cs[aa]).reshape((n,cells,SUB_HASH_BINS))
        count=cc[jj+1]-cc[aa]
        rvalid=ok&(count>=SUB_ROLL_MIN)
        warm_raw=ok&(tt<DEPLOY_BASELINE)
        warm_roll=rvalid&(tt<DEPLOY_BASELINE)&(tt>=SUB_ROLL_WINDOW-1)
        if warm_raw.sum()<40 or warm_roll.sum()<20:continue

        agg=np.maximum(X[warm_raw],0.0).sum(axis=0)
        cell_ref=agg.sum(axis=1)>0
        if not cell_ref.any():continue
        p=(agg+1.0)/(agg.sum(axis=1,keepdims=True)+SUB_HASH_BINS)

        A=np.maximum(R,0.0)
        N=A.sum(axis=2,keepdims=True);E=N*p[None,:,:]
        var=E*(1.0-p[None,:,:])+1.0
        Z=(A-E)/np.sqrt(var)
        Zw=Z[warm_roll]
        center=np.median(Zw,axis=0)
        q25,q75=np.quantile(Zw,[.25,.75],axis=0)
        scale=np.maximum((q75-q25)/1.349,1.0)
        Zn=(Z-center[None,:,:])/scale[None,:,:]

        cell_valid=(rvalid[:,None]&(N[:,:,0]>0)&cell_ref[None,:]&
                    np.isfinite(Zn).all(axis=2))
        ident_valid=np.repeat(cell_valid,SUB_HASH_BINS,axis=1)
        S=Zn.reshape((n,-1))
        P=np.arcsinh(np.maximum(S,0.0))
        P=np.where(ident_valid,P,0.0)
        Smask=np.where(ident_valid,S,-np.inf)
        m=Smask.shape[1];nvalid=np.sum(ident_valid,axis=1)
        decision=nvalid>=10

        # Cross-sectional amplitude/concentration views.
        if decision.any():
            rows=np.flatnonzero(decision)
            vals=np.partition(Smask[decision],-10,axis=1)[:,-10:]
            vals.sort(axis=1)
            out["kie_mv_max"][ix[rows]]=vals[:,-1]
            out["kie_mv_top5mean"][ix[rows]]=np.mean(vals[:,-5:],axis=1)
            out["kie_mv_top10mean"][ix[rows]]=np.mean(vals,axis=1)
            out["kie_mv_margin12"][ix[rows]]=vals[:,-1]-vals[:,-2]

        # Same exact identity over five/three consecutive valid epochs.
        pcs=np.vstack([np.zeros((1,m)),np.cumsum(P,axis=0)])
        vc=np.vstack([np.zeros((1,m),dtype=np.int16),
                      np.cumsum(ident_valid.astype(np.int16),axis=0,dtype=np.int16)])
        if n>=5:
            v5=(vc[5:]-vc[:-5])==5
            mean5=(pcs[5:]-pcs[:-5])/5.0
            mean5=np.where(v5,mean5,-np.inf)
            sc=np.max(mean5,axis=1);good=np.isfinite(sc)
            if good.any():out["kie_mv_samebin_mean5"][ix[np.flatnonzero(good)+4]]=sc[good]
        if n>=3:
            v3=ident_valid[2:]&ident_valid[1:-1]&ident_valid[:-2]
            min3=np.minimum(np.minimum(P[2:],P[1:-1]),P[:-2])
            min3=np.where(v3,min3,-np.inf)
            sc=np.max(min3,axis=1);good=np.isfinite(sc)
            if good.any():out["kie_mv_samebin_min3"][ix[np.flatnonzero(good)+2]]=sc[good]

        # Scale-free identity-rank persistence over ten epochs.
        row1=nvalid>=1;row5=nvalid>=topk
        H1=np.zeros((n,m),dtype=np.uint8)
        if row1.any():
            rows=np.flatnonzero(row1);cols=np.argmax(Smask[row1],axis=1)
            H1[rows,cols]=1
        H5=np.zeros((n,m),dtype=np.uint8)
        if row5.any():
            rows=np.flatnonzero(row5)
            top=np.argpartition(Smask[row5],-topk,axis=1)[:,-topk:]
            H5[np.repeat(rows,topk),top.reshape(-1)]=1
        c1=np.vstack([np.zeros((1,m),dtype=np.int16),
                      np.cumsum(H1,axis=0,dtype=np.int16)])
        c5=np.vstack([np.zeros((1,m),dtype=np.int16),
                      np.cumsum(H5,axis=0,dtype=np.int16)])
        rv=(row1&row5).astype(np.int16);rc=np.r_[0,np.cumsum(rv,dtype=np.int16)]
        if n>=rank_window:
            den=rc[rank_window:]-rc[:-rank_window];good=den>=rank_min
            if good.any():
                cnt1=c1[rank_window:]-c1[:-rank_window]
                cnt5=c5[rank_window:]-c5[:-rank_window]
                sc1=np.max(cnt1,axis=1)/np.maximum(den,1)
                sc5=np.max(cnt5,axis=1)/np.maximum(den,1)
                dst=np.flatnonzero(good)+(rank_window-1)
                out["kie_mv_top1_persist10"][ix[dst]]=sc1[good]
                out["kie_mv_top5_persist10"][ix[dst]]=sc5[good]

    for c,v in out.items():d[c]=v
    return d.drop(columns=[SUB_PEER_SERVICE_PACKED_NAME])


def peer_service_subscriber_rank_persistence_features(d):
    """v18: scale-free persistence of exact peer x service x keyed-bin identity.

    After the same commissioning-standardized 30-interval residual as v15-v17,
    amplitude is discarded. At each valid decision epoch, mark the top-1 and
    top-5 exact identities. Over the last ten 30-second epochs, measure the
    largest fraction belonging to the SAME identity. This is invariant to
    world-specific residual scale and prevents unrelated benign maxima from
    handing off evidence to one another.
    """
    for c in (SUB_PEER_SERVICE_TOP1PERSIST10_NAME,SUB_PEER_SERVICE_TOP5PERSIST10_NAME):
        d[c]=np.nan
    if not SUBSCRIBER_KIE_V18:return d
    if SUB_PEER_SERVICE_PACKED_NAME not in d.columns:return d

    out1=np.full(len(d),np.nan,dtype=float)
    out5=np.full(len(d),np.nan,dtype=float)
    cell_width=len(KIE_SERVICES)*SUB_HASH_BINS
    rank_window=10;min_valid=7;topk=5

    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();tt=gdf.t.to_numpy();n=len(gdf)
        obj=gdf[SUB_PEER_SERVICE_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and
                        v.size%cell_width==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size)
        cells=width//SUB_HASH_BINS
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if not ok.any():continue

        X=np.zeros((n,cells,SUB_HASH_BINS),dtype=float)
        X[ok]=np.stack(obj[ok]).reshape((-1,cells,SUB_HASH_BINS))
        flat=X.reshape((n,-1))
        cs=np.vstack([np.zeros((1,flat.shape[1])),np.cumsum(flat,axis=0)])
        cc=np.r_[0,np.cumsum(ok.astype(np.int64))]
        jj=np.arange(n,dtype=np.int64);aa=np.maximum(0,jj-SUB_ROLL_WINDOW+1)
        R=(cs[jj+1]-cs[aa]).reshape((n,cells,SUB_HASH_BINS))
        count=cc[jj+1]-cc[aa]
        rvalid=ok&(count>=SUB_ROLL_MIN)
        warm_raw=ok&(tt<DEPLOY_BASELINE)
        warm_roll=rvalid&(tt<DEPLOY_BASELINE)&(tt>=SUB_ROLL_WINDOW-1)
        if warm_raw.sum()<40 or warm_roll.sum()<20:continue

        agg=np.maximum(X[warm_raw],0.0).sum(axis=0)
        cell_ref=agg.sum(axis=1)>0
        if not cell_ref.any():continue
        p=(agg+1.0)/(agg.sum(axis=1,keepdims=True)+SUB_HASH_BINS)
        A=np.maximum(R,0.0)
        N=A.sum(axis=2,keepdims=True);E=N*p[None,:,:]
        var=E*(1.0-p[None,:,:])+1.0
        Z=(A-E)/np.sqrt(var)
        Zw=Z[warm_roll]
        center=np.median(Zw,axis=0)
        q25,q75=np.quantile(Zw,[.25,.75],axis=0)
        scale=np.maximum((q75-q25)/1.349,1.0)
        Zn=(Z-center[None,:,:])/scale[None,:,:]

        cell_valid=(rvalid[:,None]&(N[:,:,0]>0)&cell_ref[None,:]&
                    np.isfinite(Zn).all(axis=2))
        S=Zn.reshape((n,-1))
        identity_valid=np.repeat(cell_valid,SUB_HASH_BINS,axis=1)
        S=np.where(identity_valid,S,-np.inf)
        m=S.shape[1]
        valid_count=np.sum(identity_valid,axis=1)
        row1=valid_count>=1;row5=valid_count>=topk

        H1=np.zeros((n,m),dtype=np.uint8)
        if row1.any():
            rows=np.flatnonzero(row1);cols=np.argmax(S[row1],axis=1)
            H1[rows,cols]=1
        H5=np.zeros((n,m),dtype=np.uint8)
        if row5.any():
            rows=np.flatnonzero(row5)
            top=np.argpartition(S[row5],-topk,axis=1)[:,-topk:]
            H5[np.repeat(rows,topk),top.reshape(-1)]=1

        # Use int16 cumulative counts: n<=2880 intervals/day, safely <32767.
        c1=np.vstack([np.zeros((1,m),dtype=np.int16),
                      np.cumsum(H1,axis=0,dtype=np.int16)])
        c5=np.vstack([np.zeros((1,m),dtype=np.int16),
                      np.cumsum(H5,axis=0,dtype=np.int16)])
        rv=(row1&row5).astype(np.int16)
        rc=np.r_[0,np.cumsum(rv,dtype=np.int16)]
        if n<rank_window:continue
        den=rc[rank_window:]-rc[:-rank_window]
        good=den>=min_valid
        if not good.any():continue
        cnt1=c1[rank_window:]-c1[:-rank_window]
        cnt5=c5[rank_window:]-c5[:-rank_window]
        score1=np.max(cnt1,axis=1)/np.maximum(den,1)
        score5=np.max(cnt5,axis=1)/np.maximum(den,1)
        dst=np.flatnonzero(good)+(rank_window-1)
        out1[ix[dst]]=score1[good]
        out5[ix[dst]]=score5[good]

    d[SUB_PEER_SERVICE_TOP1PERSIST10_NAME]=out1
    d[SUB_PEER_SERVICE_TOP5PERSIST10_NAME]=out5
    return d.drop(columns=[SUB_PEER_SERVICE_PACKED_NAME])


def peer_service_subscriber_tail_features(d):
    """v17: per-identity commissioning-tail persistence before max reduction.

    For each exact peer x MAP-service x keyed subscriber bin, form the same
    30-interval Pearson residual used by v15/v16. Its 99th-percentile threshold
    is estimated only from the node's clean commissioning interval. Post-
    deployment evidence is the positive excess above that identity-specific
    tail, clipped so one extreme benign window cannot dominate. A candidate
    must exceed its own tail in at least 3 of the last 5 windows. Only then is
    evidence maximized across identities. Wire format is unchanged from v15.
    """
    for c in (SUB_PEER_SERVICE_TAILAREA5_NAME,SUB_PEER_SERVICE_TAILCOUNT5_NAME):
        d[c]=np.nan
    if not SUBSCRIBER_KIE_V17:return d
    if SUB_PEER_SERVICE_PACKED_NAME not in d.columns:return d

    out_area=np.full(len(d),np.nan,dtype=float)
    out_count=np.full(len(d),np.nan,dtype=float)
    cell_width=len(KIE_SERVICES)*SUB_HASH_BINS
    tail_q=.99
    evidence_cap=4.0

    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();tt=gdf.t.to_numpy();n=len(gdf)
        obj=gdf[SUB_PEER_SERVICE_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and
                        v.size%cell_width==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size)
        cells=width//SUB_HASH_BINS
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if not ok.any():continue

        X=np.zeros((n,cells,SUB_HASH_BINS),dtype=float)
        X[ok]=np.stack(obj[ok]).reshape((-1,cells,SUB_HASH_BINS))

        # Same causal 30-interval raw-count accumulation as v15/v16.
        flat=X.reshape((n,-1))
        cs=np.vstack([np.zeros((1,flat.shape[1])),np.cumsum(flat,axis=0)])
        cc=np.r_[0,np.cumsum(ok.astype(np.int64))]
        jj=np.arange(n,dtype=np.int64);aa=np.maximum(0,jj-SUB_ROLL_WINDOW+1)
        R=(cs[jj+1]-cs[aa]).reshape((n,cells,SUB_HASH_BINS))
        count=cc[jj+1]-cc[aa]
        rvalid=ok&(count>=SUB_ROLL_MIN)
        warm_raw=ok&(tt<DEPLOY_BASELINE)
        warm_roll=rvalid&(tt<DEPLOY_BASELINE)&(tt>=SUB_ROLL_WINDOW-1)
        if warm_raw.sum()<40 or warm_roll.sum()<60:continue

        agg=np.maximum(X[warm_raw],0.0).sum(axis=0)
        cell_ref=agg.sum(axis=1)>0
        if not cell_ref.any():continue
        p=(agg+1.0)/(agg.sum(axis=1,keepdims=True)+SUB_HASH_BINS)

        A=np.maximum(R,0.0)
        N=A.sum(axis=2,keepdims=True);E=N*p[None,:,:]
        var=E*(1.0-p[None,:,:])+1.0
        Z=(A-E)/np.sqrt(var)

        Zw=Z[warm_roll]
        center=np.median(Zw,axis=0)
        q25,q75=np.quantile(Zw,[.25,.75],axis=0)
        scale=np.maximum((q75-q25)/1.349,1.0)
        Zn=(Z-center[None,:,:])/scale[None,:,:]
        cell_valid=(rvalid[:,None]&(N[:,:,0]>0)&cell_ref[None,:]&
                    np.isfinite(Zn).all(axis=2))

        # Identity-specific tail calibration. A bin with a naturally heavy
        # benign tail gets its own higher threshold instead of dominating a
        # global max score.
        tail=np.quantile(Zn[warm_roll],tail_q,axis=0)
        excess=np.clip(Zn-tail[None,:,:],0.0,evidence_cap)
        excess[~cell_valid]=0.0
        hit=(excess>0.0).astype(np.int16)

        ef=excess.reshape((n,-1));hf=hit.reshape((n,-1))
        ecs=np.vstack([np.zeros((1,ef.shape[1])),np.cumsum(ef,axis=0)])
        hcs=np.vstack([np.zeros((1,hf.shape[1]),dtype=np.int32),
                       np.cumsum(hf,axis=0,dtype=np.int32)])
        if n<5:continue
        area5=ecs[5:]-ecs[:-5]
        count5=hcs[5:]-hcs[:-5]
        persistent=count5>=3

        # Area is averaged over the fixed five-window decision window; count
        # is retained as a second coarse feature for diagnostics.
        score_area=np.where(persistent,area5/5.0,-np.inf)
        score_count=np.where(persistent,count5.astype(float),-np.inf)
        ma=np.max(score_area,axis=1);mc=np.max(score_count,axis=1)
        # A valid decision epoch with no persistent identity evidence is a real
        # zero, not missing data. Keeping it as NaN would condition evaluation
        # on positive evidence and bias both calibration and AUC.
        decision_valid=rvalid[4:]
        ma=np.where(np.isfinite(ma),ma,0.0)
        mc=np.where(np.isfinite(mc),mc,0.0)
        if decision_valid.any():
            dst=np.flatnonzero(decision_valid)+4
            out_area[ix[dst]]=ma[decision_valid]
            out_count[ix[dst]]=mc[decision_valid]

    d[SUB_PEER_SERVICE_TAILAREA5_NAME]=out_area
    d[SUB_PEER_SERVICE_TAILCOUNT5_NAME]=out_count
    return d.drop(columns=[SUB_PEER_SERVICE_PACKED_NAME])


def peer_service_subscriber_identity_features(d):
    """v16: temporal confirmation inside one peer x service x keyed bin.

    Vectorized equivalent of the original v16 implementation. The wire sketch
    and statistics are unchanged: 30-interval raw-count Pearson residual per
    exact peer x service x keyed bin, followed by same-bin mean-5/min-3, and
    only then a maximum across identities.
    """
    for c in (SUB_PEER_SERVICE_IDMEAN5_NAME,SUB_PEER_SERVICE_IDMIN3_NAME):
        d[c]=np.nan
    if not SUBSCRIBER_KIE_V16:return d
    if SUB_PEER_SERVICE_PACKED_NAME not in d.columns:return d

    out_mean=np.full(len(d),np.nan,dtype=float)
    out_min=np.full(len(d),np.nan,dtype=float)
    cell_width=len(KIE_SERVICES)*SUB_HASH_BINS

    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();tt=gdf.t.to_numpy();n=len(gdf)
        obj=gdf[SUB_PEER_SERVICE_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and
                        v.size%cell_width==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size)
        peers=width//cell_width
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if not ok.any():continue

        # n x cell x bin, where cell preserves peer+exact-service identity.
        cells=peers*len(KIE_SERVICES)
        X=np.zeros((n,cells,SUB_HASH_BINS),dtype=float)
        X[ok]=np.stack(obj[ok]).reshape((-1,cells,SUB_HASH_BINS))

        # Causal 30-interval rolling sums, vectorized over all cells/bins.
        flat=X.reshape((n,-1))
        cs=np.vstack([np.zeros((1,flat.shape[1])),np.cumsum(flat,axis=0)])
        cc=np.r_[0,np.cumsum(ok.astype(np.int64))]
        jj=np.arange(n,dtype=np.int64);aa=np.maximum(0,jj-SUB_ROLL_WINDOW+1)
        R=(cs[jj+1]-cs[aa]).reshape((n,cells,SUB_HASH_BINS))
        count=cc[jj+1]-cc[aa]
        rvalid=ok&(count>=SUB_ROLL_MIN)
        warm_raw=ok&(tt<DEPLOY_BASELINE)
        warm_roll=rvalid&(tt<DEPLOY_BASELINE)&(tt>=SUB_ROLL_WINDOW-1)
        if warm_raw.sum()<40 or warm_roll.sum()<20:continue

        agg=np.maximum(X[warm_raw],0.0).sum(axis=0)              # cell x bin
        cell_ref=agg.sum(axis=1)>0
        if not cell_ref.any():continue
        p=(agg+1.0)/(agg.sum(axis=1,keepdims=True)+SUB_HASH_BINS)

        A=np.maximum(R,0.0)
        N=A.sum(axis=2,keepdims=True)
        E=N*p[None,:,:]
        var=E*(1.0-p[None,:,:])+1.0
        Z=(A-E)/np.sqrt(var)

        Zw=Z[warm_roll]                                          # warm x cell x bin
        center=np.median(Zw,axis=0)
        q25,q75=np.quantile(Zw,[.25,.75],axis=0)
        scale=np.maximum((q75-q25)/1.349,1.0)
        Zn=(Z-center[None,:,:])/scale[None,:,:]

        cell_valid=(rvalid[:,None]&(N[:,:,0]>0)&cell_ref[None,:]&
                    np.isfinite(Zn).all(axis=2))
        P=np.arcsinh(np.maximum(Zn,0.0))
        P[~cell_valid]=0.0

        # Five consecutive observations in exactly the same cell. The keyed
        # bin dimension is retained until after the temporal mean.
        pcs=np.concatenate([np.zeros((1,cells,SUB_HASH_BINS)),
                            np.cumsum(P,axis=0)],axis=0)
        vc=np.concatenate([np.zeros((1,cells),dtype=np.int64),
                           np.cumsum(cell_valid.astype(np.int64),axis=0)],axis=0)
        if n>=5:
            valid5=(vc[5:]-vc[:-5])==5                         # (n-4) x cell
            mean5=(pcs[5:]-pcs[:-5])/5.0                       # (n-4) x cell x bin
            mean5=np.where(valid5[:,:,None],mean5,-np.inf)
            score5=np.max(mean5,axis=(1,2))
            good5=np.isfinite(score5)
            if good5.any():
                dst=np.flatnonzero(good5)+4
                out_mean[ix[dst]]=score5[good5]

        # Three consecutive observations in exactly the same cell/bin.
        if n>=3:
            valid3=cell_valid[2:]&cell_valid[1:-1]&cell_valid[:-2]
            min3=np.minimum(np.minimum(P[2:],P[1:-1]),P[:-2])
            min3=np.where(valid3[:,:,None],min3,-np.inf)
            score3=np.max(min3,axis=(1,2))
            good3=np.isfinite(score3)
            if good3.any():
                dst=np.flatnonzero(good3)+2
                out_min[ix[dst]]=score3[good3]

    d[SUB_PEER_SERVICE_IDMEAN5_NAME]=out_mean
    d[SUB_PEER_SERVICE_IDMIN3_NAME]=out_min
    return d.drop(columns=[SUB_PEER_SERVICE_PACKED_NAME])


def peer_service_subscriber_roll30_features(d):
    """15-minute raw-count aggregation before peer-service subscriber residuals.

    v11 uses exactly the v10 32-bin uint16 KIE on the wire. The coordinator
    retains the last 30 fresh sketches, sums each peer x service x bin cell,
    and only then forms a commissioning-normalized Pearson residual. Missing
    reports do not count as zero traffic; at least 15 fresh reports and a
    fresh current report are required for an alertable score.
    """
    d[SUB_PEER_SERVICE_ROLL30_NAME]=np.nan
    if not (SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V15):return d
    if SUB_PEER_SERVICE_PACKED_NAME not in d.columns:return d

    out=np.full(len(d),np.nan,dtype=float)
    cell_width=len(KIE_SERVICES)*SUB_HASH_BINS

    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();tt=gdf.t.to_numpy()
        obj=gdf[SUB_PEER_SERVICE_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and
                        v.size%cell_width==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size)
        peers=width//cell_width
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if not ok.any():continue

        X=np.zeros((len(gdf),peers,len(KIE_SERVICES),SUB_HASH_BINS),dtype=float)
        X[ok]=np.stack(obj[ok]).reshape((-1,peers,len(KIE_SERVICES),SUB_HASH_BINS))

        # Causal rolling sums and fresh-report counts. No future samples.
        flat=X.reshape((len(gdf),-1))
        cs=np.vstack([np.zeros((1,flat.shape[1])),np.cumsum(flat,axis=0)])
        cc=np.r_[0,np.cumsum(ok.astype(np.int64))]
        R=np.zeros_like(flat)
        count=np.zeros(len(gdf),dtype=np.int64)
        for j in range(len(gdf)):
            a=max(0,j-SUB_ROLL_WINDOW+1)
            R[j]=cs[j+1]-cs[a]
            count[j]=cc[j+1]-cc[a]
        R=R.reshape(X.shape)
        rvalid=ok&(count>=SUB_ROLL_MIN)

        # Commissioning reference uses only windows ending before the frozen
        # deployment baseline. At least one full nominal window is required.
        warm_raw=ok&(tt<DEPLOY_BASELINE)
        warm_roll=rvalid&(tt<DEPLOY_BASELINE)&(tt>=SUB_ROLL_WINDOW-1)
        if warm_raw.sum()<40 or warm_roll.sum()<20:continue

        best=np.zeros(len(gdf),dtype=float)
        any_cell=np.zeros(len(gdf),dtype=bool)

        for pi in range(peers):
            for si in range(len(KIE_SERVICES)):
                raw=X[:,pi,si,:]
                agg=np.maximum(raw[warm_raw],0.0).sum(axis=0)
                if agg.sum()<=0:continue
                alpha=1.0
                p=(agg+alpha)/(agg.sum()+alpha*SUB_HASH_BINS)

                A=np.maximum(R[:,pi,si,:],0.0)
                N=A.sum(axis=1,keepdims=True)
                E=N*p[None,:]
                # Multinomial counting variance. Long-window benign
                # over-dispersion is learned empirically below, instead of a
                # 10%*E term that would erase the SNR gain from aggregation.
                var=E*(1.0-p[None,:])+1.0
                Z=(A-E)/np.sqrt(var)

                Zw=Z[warm_roll]
                center=np.median(Zw,axis=0)
                q25,q75=np.quantile(Zw,[.25,.75],axis=0)
                scale=np.maximum((q75-q25)/1.349,1.0)
                Zn=(Z-center[None,:])/scale[None,:]
                cell=rvalid&(N[:,0]>0)&np.isfinite(Zn).all(axis=1)
                score=np.arcsinh(np.maximum(Zn,0.0).max(axis=1))
                best=np.maximum(best,np.where(cell,score,0.0))
                any_cell|=cell

        best[~any_cell]=np.nan
        out[ix]=best

    d[SUB_PEER_SERVICE_ROLL30_NAME]=out
    return d.drop(columns=[SUB_PEER_SERVICE_PACKED_NAME])


def peer_service_subscriber_residual_features(d):
    """Residuals for peer x exact-MAP-service keyed subscriber sketches (v9)."""
    for c in SUB_PEER_SERVICE_RESID_NAMES:d[c]=np.nan
    if not (SUBSCRIBER_KIE_V9 or SUBSCRIBER_KIE_V10):return d
    if SUB_PEER_SERVICE_PACKED_NAME not in d.columns:return d

    nrow=len(d)
    result={c:np.full(nrow,np.nan,dtype=float) for c in SUB_PEER_SERVICE_RESID_NAMES}
    cell_width=len(KIE_SERVICES)*SUB_HASH_BINS
    for _,gdf in d.groupby("nid",sort=False):
        ix=gdf.index.to_numpy();tt=gdf.t.to_numpy()
        obj=gdf[SUB_PEER_SERVICE_PACKED_NAME].to_numpy()
        valid=np.array([isinstance(v,np.ndarray) and v.size>0 and v.size%cell_width==0 for v in obj],dtype=bool)
        if not valid.any():continue
        width=int(obj[np.flatnonzero(valid)[0]].size);peers=width//cell_width
        X=np.full((len(gdf),peers,len(KIE_SERVICES),SUB_HASH_BINS),np.nan,dtype=float)
        ok=valid.copy()
        for j in np.flatnonzero(valid):
            if obj[j].size!=width:ok[j]=False
        if ok.any():
            X[ok]=np.stack(obj[ok]).reshape((-1,peers,len(KIE_SERVICES),SUB_HASH_BINS))
        warm=ok&(tt<DEPLOY_BASELINE)
        if warm.sum()<20:continue

        best=np.zeros((len(gdf),3),float);any_cell=np.zeros(len(gdf),bool)
        for pi in range(peers):
            for si in range(len(KIE_SERVICES)):
                A=X[:,pi,si,:];H=np.maximum(A[warm],0.0);agg=H.sum(axis=0)
                # No commissioning traffic for this peer/service => no stable
                # reference cell; do not turn first use into an automatic alarm.
                if agg.sum()<=0:continue
                alpha=1.0;p=(agg+alpha)/(agg.sum()+alpha*SUB_HASH_BINS)
                def pearson(B):
                    B=np.maximum(np.asarray(B,float),0.0)
                    N=B.sum(axis=1,keepdims=True);E=N*p[None,:]
                    var=E*(1.0-p[None,:])+(0.10*E)**2+1.0
                    return (B-E)/np.sqrt(var)
                Zw=pearson(H);center=np.median(Zw,axis=0)
                q25,q75=np.quantile(Zw,[.25,.75],axis=0)
                scale=np.maximum((q75-q25)/1.349,1.0)
                cell=np.isfinite(A).all(axis=1)
                Z=np.zeros_like(A,dtype=float)
                if cell.any():Z[cell]=(pearson(A[cell])-center[None,:])/scale[None,:]
                P=np.maximum(Z,0.0)
                vals=np.column_stack((
                    np.arcsinh(np.max(P,axis=1)),
                    np.arcsinh(np.sqrt(np.mean(P*P,axis=1))),
                    np.arcsinh(np.sqrt(np.sum(P*P,axis=1)))))
                best=np.maximum(best,vals);any_cell|=cell
        best[~any_cell]=np.nan
        for k,c in enumerate(SUB_PEER_SERVICE_RESID_NAMES):result[c][ix]=best[:,k]

    for c,v in result.items():d[c]=v
    return d.drop(columns=[SUB_PEER_SERVICE_PACKED_NAME])


def group_counts(a):
    out=defaultdict(int)
    for (direction,k,j),v in a.items():out[(direction,k,GROUP[j])]+=int(v)
    return dict(out)

def marginals(b):
    out=defaultdict(int)
    for (ki,ko,j),v in b.items():
        out[(0,ki,GROUP[j])]+=int(v);out[(1,ko,GROUP[j])]+=int(v)
    return dict(out)

def _group_sketch_bin(parts):
    h=2166136261
    for x in parts:
        h=((h^int(x))*16777619)&0xffffffff
    return int(h%KIE_GROUP_SKETCH_BINS)

def grouped_route_sketch(n,typed,mat,topo):
    """Coordinator-side sketch from compact KIE counters already on the wire."""
    bins=np.zeros(KIE_GROUP_SKETCH_BINS,dtype=float)
    links=sorted(topo.links_of(n))
    rank={k:i for i,k in enumerate(links)}
    for (direction,k,g),v in typed.items():
        x=float(max(v,0))
        if x:bins[_group_sketch_bin((1,direction,rank.get(k,0),g))]+=x
    for (direction,k,g),v in mat.items():
        x=float(max(v,0))
        if x:bins[_group_sketch_bin((2,direction,rank.get(k,0),g))]+=x
    return bins

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
        sketch=grouped_route_sketch(n,tc,mc,topo)
        out[n].update({name:float(v) for name,v in zip(KIE_GROUP_SKETCH,sketch)})
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
    def compromise_candidates(self,n,e):
        """Role-consistent malicious SS7 procedures on already legal routes.

        The legacy corrected arm sampled any normal transaction touching the
        compromised node, often making a NAIVE compromise indistinguishable
        from benign load variation. In v2 the attacker still uses only routes
        and service templates already present in the simulated SS7 network,
        but abuses MAP procedures appropriate for the node role. No label,
        future state or ground-truth feature is exposed to the detector.
        """
        opts=self.candidates(n)
        if not (ROLE_ABUSE_V2 or TARGETED_ABUSE_V5 or SUBSCRIBER_SKETCH) or not opts:
            return opts
        typ=self.topo.nodes[n].ntype
        menus={
            F.NodeType.SP:(Service.LOCATION,Service.SMS_MT,Service.AUTH,Service.MOBILITY),
            F.NodeType.SCP:(Service.LOCATION,Service.ROAM_SRI,Service.SMS_MT,Service.AUTH,Service.MOBILITY),
            F.NodeType.IGW:(Service.ROAM_SRI,Service.SMS_MT,Service.LOCATION),
            F.NodeType.STP:(Service.LOCATION,Service.ROAM_SRI,Service.SMS_MT,Service.MOBILITY,Service.AUTH),
        }
        menu=menus[typ]
        j=min(len(menu)-1,int(float(e.params[0])*len(menu)))
        primary=menu[j]
        chosen=[x for x in opts if x[0].service==primary]
        if chosen:return chosen
        chosen=[x for x in opts if x[0].service in menu]
        return chosen or opts

    def _subscriber_probs(self,f):
        if not hasattr(self,"_subscriber_prob_cache"):self._subscriber_prob_cache={}
        service_idx=list(F.Service).index(f.service)
        if SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19 or SUBSCRIBER_KIE_V22:
            # A keyed subscriber hash should not acquire a radically different
            # bucket distribution merely because the same MAP service uses a
            # different src/dst flow. Keep modest service-level population
            # non-uniformity (~8.7% relative bucket SD for K=32, alpha=128)
            # while eliminating the former alpha=4 (~49%) flow-specific drift.
            key=("stable_service_hash",service_idx)
            if key not in self._subscriber_prob_cache:
                srng=np.random.default_rng(np.random.SeedSequence(
                    int(self.cfg.seed),spawn_key=(3400,service_idx)))
                self._subscriber_prob_cache[key]=srng.dirichlet(
                    np.full(SUB_HASH_BINS,128.0))
            return self._subscriber_prob_cache[key]
        key=(int(f.src),int(f.dst),service_idx)
        if key not in self._subscriber_prob_cache:
            srng=np.random.default_rng(np.random.SeedSequence(
                int(self.cfg.seed),spawn_key=(3100,key[0],key[1],key[2])))
            # Legacy development worlds retain their original per-flow model
            # for reproducibility.
            self._subscriber_prob_cache[key]=srng.dirichlet(
                np.full(SUB_HASH_BINS,4.0))
        return self._subscriber_prob_cache[key]

    def normal_subscriber_bins(self,f,cnt,t):
        if not SUBSCRIBER_SKETCH:return None
        if (SUBSCRIBER_KIE_V9 or SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19 or SUBSCRIBER_KIE_V22) and f.service not in KIE_SERVICE_INDEX:return None
        key=(int(f.src),int(f.dst),list(F.Service).index(f.service))
        srng=np.random.default_rng(np.random.SeedSequence(
            int(self.cfg.seed),spawn_key=(3101,int(t),key[0],key[1],key[2])))
        return srng.multinomial(int(cnt),self._subscriber_probs(f)).astype(np.int64)

    def attack_subscriber_bins(self,e,cnt):
        if not SUBSCRIBER_SKETCH:return None
        srng=np.random.default_rng(np.random.SeedSequence(
            int(self.cfg.seed),spawn_key=(3102,int(e.nid),int(e.t0))))
        b=int(srng.integers(SUB_HASH_BINS))
        out=np.zeros(SUB_HASH_BINS,dtype=np.int64);out[b]=int(cnt)
        return out

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
                sub=self.normal_subscriber_bins(f,int(cnt),t)
                pending.append((f, path, int(cnt), None, sub))
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
            opts=self.compromise_candidates(n,e)
            if not opts:continue
            if TARGETED_ABUSE_V5 or SUBSCRIBER_SKETCH:
                # A compromise episode represents one targeted abuse campaign.
                # Keep its legal flow/path fixed for the full episode instead of
                # re-randomising the destination every 30 s. e.params[1] is
                # sampled once when the episode is created and is not observable
                # by the detector.
                j=min(len(opts)-1,int(float(e.params[1])*len(opts)))
                f,path=opts[j]
            else:
                f,path=opts[int(arng.integers(len(opts)))]
            if any(tuple(sorted((u,v))) in failed_links for u,v in zip(path[:-1],path[1:])):continue
            sub=self.attack_subscriber_bins(e,h)
            pending.append((f,path,h,n,sub))
            bits=8.*sum(F.MSG_BYTES[m]*c for m,c in F.SERVICE_TEMPLATE[f.service].items())
            for u,v in zip(path[:-1],path[1:]):
                k=tuple(sorted((u,v)));offered[k]=offered.get(k,0.)+h*bits/dt
        lstate,self.polls=self.cu.exchange(t,offered,failed_links)
        records=defaultdict(list)
        cap_checked=defaultdict(int);cap_missing=defaultdict(int)
        caprng=extra_rng(self.cfg.seed,44,t)
        for f,path,cnt,owner,sub in pending:
            hops=[tuple(sorted((u,v))) for u,v in zip(path[:-1],path[1:])]
            p=float(np.prod([1.-lstate[k][2] for k in hops]))**2
            p=1.-(1.-p)**(1+self.max_retry)
            crng=rng if owner is None else arng
            closed=int(crng.binomial(cnt,np.clip(p,0.,1.)))
            if CAPABILITY_KIE_V23 and f.service in KIE_SERVICE_INDEX and closed>0:
                # The verifier is independent from the signalling node report.
                # Normal operations almost always carry a valid authorization
                # context; hidden injected operations do not possess one.
                actor=int(owner) if owner is not None else int(f.src)
                if actor in topo.nodes and not topo.nodes[actor].external:
                    cap_checked[actor]+=closed
                    if owner is not None:
                        cap_missing[actor]+=closed
                    else:
                        cap_missing[actor]+=int(caprng.binomial(closed,CAP_BENIGN_FAIL))
            for i,n in enumerate(path):records[n].append((path,i,f.service,cnt,closed,owner,sub))
        true={};claimed={};typed={};mat={};boundary={};audit={};subclaimed={}
        for n in topo.nodes:
            st=empty(n,t);cl=empty(n,t);tt=defaultdict(int);tm=defaultdict(int)
            ct=defaultdict(int);cm=defaultdict(int);hidden=0;removed=0;observable=0
            if SUBSCRIBER_KIE_V9 or SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19 or SUBSCRIBER_KIE_V22:
                peer_ids=[topo.peer(k,n) for k in topo.links_of(n)]
                peer_pos={p:i for i,p in enumerate(peer_ids)}
                sub_claim=np.zeros((len(peer_ids),len(KIE_SERVICES),SUB_HASH_BINS),dtype=np.int64)
            elif SUBSCRIBER_KIE_V8:
                peer_ids=[topo.peer(k,n) for k in topo.links_of(n)]
                peer_pos={p:i for i,p in enumerate(peer_ids)}
                sub_claim=np.zeros((len(peer_ids),len(GROUPS),SUB_HASH_BINS),dtype=np.int64)
            else:
                peer_pos={}
                sub_claim=np.zeros((len(GROUPS),SUB_HASH_BINS),dtype=np.int64)
            e=active.get(n)
            for path,i,service,cnt,closed,owner,sub in records[n]:
                add_record(st,tt,tm,path,i,service,cnt,closed)
                remain=cnt;cc=closed
                sg=service_group(service)
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
                if SUBSCRIBER_SKETCH and sub is not None:
                    if owner==n and remain!=cnt:
                        cv=np.zeros_like(sub)
                        if sub.sum()>0:
                            # Malicious v6 batches occupy one keyed bin; keep
                            # the claimed sketch coherent with the claimed txn count.
                            if np.count_nonzero(sub)==1:
                                cv[int(np.argmax(sub))]=int(remain)
                            else:
                                psub=sub/sub.sum()
                                cv=np.floor(psub*remain).astype(np.int64)
                                if cv.sum()<remain:cv[int(np.argmax(psub))]+=int(remain-cv.sum())
                    else:
                        cv=sub
                    if SUBSCRIBER_KIE_V9 or SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19 or SUBSCRIBER_KIE_V22:
                        si=KIE_SERVICE_INDEX.get(service)
                        if si is not None:
                            adj=[]
                            if i>0:adj.append(path[i-1])
                            if i<len(path)-1:adj.append(path[i+1])
                            for peer in adj:
                                if peer in peer_pos:sub_claim[peer_pos[peer],si]+=cv
                    elif SUBSCRIBER_KIE_V8:
                        adj=[]
                        if i>0:adj.append(path[i-1])
                        if i<len(path)-1:adj.append(path[i+1])
                        for peer in adj:
                            if peer in peer_pos:sub_claim[peer_pos[peer],sg]+=cv
                    else:
                        sub_claim[sg]+=cv
            st.check_identities();cl.check_identities()
            true[n]=st;claimed[n]=cl;typed[n]=ct;mat[n]=cm;subclaimed[n]=sub_claim
            # Explicit additional sensor: true offered counters sampled on
            # the domestic side of an external link, independently of node report.
            # Only sensor measurements, never this latent dictionary, reach features.
            boundary[n]={k:v for k,v in tt.items() if topo.nodes[topo.peer(k[1],n)].external}
            audit[n]=(hidden,removed,observable,max(st.total()-hidden,0))
        if CAPABILITY_KIE_V23:
            self.last_cap_kie={n:(int(cap_checked.get(n,0)),int(cap_missing.get(n,0)))
                               for n in topo.nodes if not topo.nodes[n].external}
        else:
            self.last_cap_kie={}
        return true,claimed,typed,mat,boundary,audit,lstate,subclaimed

def schedule(sim,rng):
    # Every test topology has all classes; role/sophistication balanced for compromise.
    cfg=sim.cfg
    eps=F.make_episodes(sim,cfg,rng,n_compromise=0,n_normal=48,n_ext=0)
    occ=defaultdict(list)
    def place(nodes,kind,length,**kwargs):
        for _ in range(5000):
            n=int(rng.choice(nodes));a=int(rng.integers(max(WARMUP+5,DEPLOY_BASELINE),cfg.n_intervals-length-5));b=a+length-1
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
    for c in CAP_DERIVED_NAMES:d[c]=np.nan
    if CAPABILITY_KIE_V23 and all(c in d.columns for c in CAP_KIE_NAMES):
        for _,g in d.groupby("nid",sort=False):
            miss=g["kie_cap_missing"].fillna(0.0).astype(float)
            checked=g["kie_cap_checked"].fillna(0.0).astype(float)
            sm=miss.rolling(10,min_periods=4).sum()
            sc=checked.rolling(10,min_periods=4).sum()
            d.loc[g.index,"kie_cap_missing_5m"]=sm.to_numpy()
            d.loc[g.index,"kie_cap_missing_rate_5m"]=(sm/np.maximum(sc,1.0)).to_numpy()
    d=heavy_hitter_subscriber_features(d)
    d=peer_service_subscriber_multiview_features(d)
    d=peer_service_subscriber_rank_persistence_features(d)
    d=peer_service_subscriber_tail_features(d)
    d=peer_service_subscriber_identity_features(d)
    d=peer_service_subscriber_bin_cusum_features(d)
    d=peer_service_subscriber_roll30_features(d)
    d=peer_service_subscriber_residual_features(d)
    d=peer_subscriber_residual_features(d)
    d=subscriber_residual_features(d)
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
        # Sustained-level drift: recent 5 min versus an earlier 30 min baseline.
        # Baseline ends 5 min before the recent window, so the current shift
        # cannot immediately contaminate its own reference distribution.
        for c in DRIFT_SOURCE:
            x=g[c].astype(float)
            recent=x.rolling(10,min_periods=4).mean()
            past=x.shift(10)
            base=past.rolling(60,min_periods=15).mean()
            sd=past.rolling(60,min_periods=15).std()
            delta=recent-base
            d.loc[g.index,"drift_"+c+"_rel"]=(delta/(base.abs()+1e-3)).clip(-20,20).to_numpy()
            d.loc[g.index,"drift_"+c+"_z"]=(delta/(sd+.05*base.abs()+1e-3)).clip(-20,20).to_numpy()
    # Grouped-route sketch is derived at the coordinator from compact KIE counters. Convert it
    # at the coordinator into causal composition drift against the node's own
    # earlier baseline. Missing current KIE => derived sketch evidence missing.
    for name in KIE_GROUP_DERIVED:d[name]=np.nan
    if all(c in d.columns for c in KIE_GROUP_SKETCH):
        for _,g in d.groupby("nid",sort=False):
            ix=g.index.to_numpy()
            frame=g[list(KIE_GROUP_SKETCH)].astype(float)
            recent=frame.rolling(10,min_periods=4).mean()
            baseline=frame.shift(10).rolling(60,min_periods=15).mean()
            P=recent.to_numpy(float);Q=baseline.to_numpy(float)
            ps=np.nansum(P,axis=1,keepdims=True);qs=np.nansum(Q,axis=1,keepdims=True)
            valid=(np.isfinite(P).all(axis=1)&np.isfinite(Q).all(axis=1)&
                   (ps[:,0]>0)&(qs[:,0]>0)&(g.cu_delivered.to_numpy(float)>0))
            P=np.divide(P,ps,out=np.zeros_like(P),where=ps>0)
            Q=np.divide(Q,qs,out=np.zeros_like(Q),where=qs>0)
            M=.5*(P+Q);eps=1e-12
            klp=np.sum(np.where(P>0,P*np.log((P+eps)/(M+eps)),0.0),axis=1)
            klq=np.sum(np.where(Q>0,Q*np.log((Q+eps)/(M+eps)),0.0),axis=1)
            js=.5*(klp+klq)
            shift=np.max(np.abs(P-Q),axis=1)
            js[~valid]=np.nan;shift[~valid]=np.nan
            d.loc[ix,"kie_group_route_js"]=js
            d.loc[ix,"kie_group_route_maxshift"]=shift

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
    row_feature_cols=list(NEW)+list(O.EXTRA)+list(D.BEHAVIOR)+list(KIE_GROUP_SKETCH)
    if CAPABILITY_KIE_V23:
        row_feature_cols+=list(CAP_KIE_NAMES)+list(CAP_DERIVED_NAMES)
    if SUBSCRIBER_KIE_V22:
        row_feature_cols+=list(SUB_HH_NAMES)
    elif SUBSCRIBER_KIE_V6:
        row_feature_cols+=list(SUB_PROFILE_NAMES)+list(SUB_BIN_NAMES)
    elif SUBSCRIBER_KIE_V7:
        row_feature_cols+=list(SUB_RESID_NAMES)
    elif SUBSCRIBER_KIE_V8:
        row_feature_cols+=list(SUB_PEER_RESID_NAMES)
    elif SUBSCRIBER_KIE_V9 or SUBSCRIBER_KIE_V10:
        row_feature_cols+=list(SUB_PEER_SERVICE_RESID_NAMES)
    elif SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V15:
        row_feature_cols+=[SUB_PEER_SERVICE_ROLL30_NAME]
    elif SUBSCRIBER_KIE_V16:
        row_feature_cols+=[SUB_PEER_SERVICE_IDMEAN5_NAME,SUB_PEER_SERVICE_IDMIN3_NAME]
    elif SUBSCRIBER_KIE_V17:
        row_feature_cols+=[SUB_PEER_SERVICE_TAILAREA5_NAME,SUB_PEER_SERVICE_TAILCOUNT5_NAME]
    elif SUBSCRIBER_KIE_V18:
        row_feature_cols+=[SUB_PEER_SERVICE_TOP1PERSIST10_NAME,SUB_PEER_SERVICE_TOP5PERSIST10_NAME]
    elif SUBSCRIBER_KIE_V19:
        row_feature_cols+=list(SUB_MULTIVIEW_NAMES)
    elif SUBSCRIBER_KIE_V12:
        row_feature_cols+=[SUB_PEER_SERVICE_BIN_CUSUM_NAME]
    elif SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14:
        row_feature_cols+=[SUB_PEER_SERVICE_BIN_CUSUM_UNCAPPED_NAME]
    for t in range(cfg.n_intervals):
        truth,claimed,typed,mat,boundary,audit,lstate,subclaimed=sim.step_true(t,eps)
        capkie=getattr(sim,"last_cap_kie",{})
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
            if arm=="compact":
                a,b,z=encode_compact(rep,a,b)
                if SUBSCRIBER_SKETCH:
                    srng=extra_rng(seed,33,t*1000+n)
                    subwire=noisy_subscriber_sketch(subclaimed[n],srng,sim.rel_sigma)
                    if SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19:
                        subwire=np.minimum(subwire,np.iinfo(np.uint16).max)
                    z.update(subscriber_profile(subwire))
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
            row={c:np.nan for c in row_feature_cols}
            if SUBSCRIBER_KIE_V22:row[SUB_HH_PACKED_NAME]=None
            if SUBSCRIBER_KIE_V7:row[SUB_PACKED_NAME]=None
            if SUBSCRIBER_KIE_V8:row[SUB_PEER_PACKED_NAME]=None
            if SUBSCRIBER_KIE_V9 or SUBSCRIBER_KIE_V10 or SUBSCRIBER_KIE_V11 or SUBSCRIBER_KIE_V12 or SUBSCRIBER_KIE_V13 or SUBSCRIBER_KIE_V14 or SUBSCRIBER_KIE_V15 or SUBSCRIBER_KIE_V16 or SUBSCRIBER_KIE_V17 or SUBSCRIBER_KIE_V18 or SUBSCRIBER_KIE_V19 or SUBSCRIBER_KIE_V22:row[SUB_PEER_SERVICE_PACKED_NAME]=None
            row.update(inv.get(n,{}));row.update(extra.get(n,{}));row.update(behavior.get(n,{}))
            if CAPABILITY_KIE_V23:
                checked,missing=capkie.get(n,(0,0))
                row["kie_cap_checked"]=float(checked)
                row["kie_cap_missing"]=float(missing)
                row["kie_cap_missing_rate"]=float(missing/max(checked,1))
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
    if mode=="naive_drift":return list(F.OWN)+list(SUB)+[c for c in d if c.startswith("drift_")]+["node_type_enc"]
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
        model=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),est)
    elif kind=="gb":
        est=GradientBoostingClassifier(n_estimators=80,learning_rate=.04,max_depth=1,
            min_samples_leaf=30,subsample=.65,max_features="sqrt",random_state=42)
        model=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),est)
    elif kind=="logit":
        # Strong L2 shrinkage and role-specific fitting deliberately trade
        # training fit for transfer across independently generated worlds.
        est=LogisticRegression(C=.05,max_iter=800,solver="lbfgs",random_state=42)
        model=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),
                            StandardScaler(),est)
    else:
        est=HistGradientBoostingClassifier(max_iter=70,learning_rate=.05,max_leaf_nodes=7,
            min_samples_leaf=50,l2_regularization=2.0,early_stopping=False,random_state=42)
        model=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),est)
    model.fit(matrix(d,names),y,**{model.steps[-1][0]+"__sample_weight":w})
    return names,model
def fit_compromise_soph(d,soph,mode,kind="hgb"):
    # Train each evasion-mode expert against normal traffic only. Other attack
    # modes are not negatives because firing on any attack is harmless for the
    # binary anomaly decision; this prevents artificial inter-attack borders.
    keep=(d.attack=="normal")|((d.attack=="compromise")&(d.sophistication==soph))
    q=d.loc[keep].copy()
    return fit(q,mode,"comp",kind)

def fit_compromise_role_soph(d,soph,role,mode="naive_drift",kind="logit"):
    """Fit one compromise expert inside a single observable node role."""
    keep=((d.attack=="normal")|((d.attack=="compromise")&(d.sophistication==soph))) & (d.ntype.astype(str)==role)
    q=d.loc[keep].copy()
    if q.empty or target_y(q,"comp").sum()==0:
        raise ValueError(f"no training support for {soph}/{role}")
    return fit(q,mode,"comp",kind)



OTHER=KINDS[:-1]
COMMON_MODELS=("local","article_raw","integrated_raw","dual_tail","dual_tail_confirm","five_tail","five_role_tail")
MODELS=COMMON_MODELS+("five_calibrated","five_augmented")
FIVE_CHANNELS=OTHER+("invariant",)
def load(root,arm,seed):
    return joblib.load(root/f"world-{arm}-{seed}"/"world.joblib")

def _training_frame(root,arm,seeds):
    data=[]
    for seed in seeds:
        d,e,cu=load(root,arm,seed)
        d["uid"]=str(seed)+":"+d.episode_id.astype(str)
        d["world_node"]=str(seed)+":"+d.nid.astype(str)
        phase=seed%TRAIN_STRIDE
        thin=(d.t.to_numpy()%TRAIN_STRIDE)==phase
        pos=np.flatnonzero((d.label.to_numpy()==1)&thin)
        neg=np.flatnonzero((d.label.to_numpy()==0)&thin)
        neg=np.random.default_rng(seed).choice(neg,min(TRAIN_NORMAL_PER_WORLD,len(neg)),replace=False)
        ix=np.sort(np.r_[pos,neg]);data.append(d.iloc[ix])
    return pd.concat(data,ignore_index=True)

def train(root,out,arm):
    d=_training_frame(root,arm,TRAIN)
    spec={"local":("local","all","rf"),"general":("base","all","rf"),
          "article_rf":("special","comp","rf"),"article_gb":("special","comp","gb"),
          "invariant":("invariant","comp","hgb"),"behavior":("behavior","comp","hgb")}
    spec.update({k:("base",k,"hgb") for k in OTHER})
    models={}
    for n,sp in spec.items():
        models[n]=fit(d,*sp);print("FIT",arm,n,flush=True)

    # v5 keeps RANK as a true method-selection holdout. Earlier development
    # arms are preserved for reproducibility, but no v5 learner may fit RANK.
    aug_seeds=TRAIN if (TARGETED_ABUSE_V5 or SUBSCRIBER_SKETCH) else TRAIN+RANK
    aug=_training_frame(root,arm,aug_seeds)
    aug_spec={k+"_aug":("base",k,"hgb") for k in OTHER}
    aug_spec["invariant_aug"]=("invariant","comp","hgb")
    aug_spec["local_aug"]=("local","all","rf")
    for n,sp in aug_spec.items():
        models[n]=fit(aug,*sp);print("FIT",arm,n,flush=True)

    # Evasion-mode experts. NAIVE exposes its true report, so its signal lives
    # in local/base temporal and zone behaviour. STATISTICAL/ADAPTIVE actively
    # falsify reports, so cross-node invariants are the appropriate evidence.
    soph_spec={
        # Keep the former global naive model only as a diagnostic / CUSUM
        # baseline. The primary cascade below uses role-specific linear experts.
        "comp_naive_aug":("naive","naive_drift","hgb"),
        "comp_statistical_aug":("statistical","invariant","hgb"),
        "comp_adaptive_aug":("adaptive","invariant","hgb"),
    }
    for n,(soph,mode,kind) in soph_spec.items():
        models[n]=fit_compromise_soph(aug,soph,mode,kind);print("FIT",arm,n,flush=True)
    for role,name in NAIVE_ROLE_MODEL_NAMES.items():
        models[name]=fit_compromise_role_soph(aug,"naive",role,"naive_drift","logit")
        print("FIT",arm,name,flush=True)

    cal=[]
    for seed in CAL:
        q,_,_=load(root,arm,seed)
        q["uid"]=str(seed)+":"+q.episode_id.astype(str)
        q["world_node"]=str(seed)+":"+q.nid.astype(str)
        cal.append(q)
    cal=pd.concat(cal,ignore_index=True)
    diag=[]
    all_spec=dict(spec);all_spec.update(aug_spec)
    for name,(mode,target,kind) in all_spec.items():
        names,model=models[name]
        train_frame=aug if name.endswith("_aug") else d
        for split,frame in (("train",train_frame),("cal",cal)):
            y=target_y(frame,target);score=model.predict_proba(matrix(frame,names))[:,1]
            diag.append(dict(model=name,split=split,n=len(frame),positive=int(y.sum()),
                auc_roc=roc_auc_score(y,score) if len(np.unique(y))==2 else np.nan,
                auc_pr=average_precision_score(y,score) if y.sum() else np.nan,
                f1_at_05=f1_score(y,score>=.5,zero_division=0)))
    for name,(soph,mode,kind) in soph_spec.items():
        names,model=models[name]
        for split,frame in (("train",aug),("cal",cal)):
            keep=(frame.attack=="normal")|((frame.attack=="compromise")&(frame.sophistication==soph))
            q=frame.loc[keep]
            y=((q.attack=="compromise")&(q.sophistication==soph)).to_numpy(int)
            score=model.predict_proba(matrix(q,names))[:,1]
            diag.append(dict(model=name,split=split,n=len(q),positive=int(y.sum()),
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
    (out/"anti_overfit.json").write_text(json.dumps(dict(train=TRAIN,augmented_train=aug_seeds,rank=RANK,
        train_stride=TRAIN_STRIDE,normal_per_world=TRAIN_NORMAL_PER_WORLD,
        rf=dict(max_depth=10,min_samples_leaf=20,max_features="sqrt",max_samples=.70),
        gb=dict(max_depth=1,min_samples_leaf=30,learning_rate=.04,subsample=.65),
        hgb=dict(max_leaf_nodes=7,min_samples_leaf=50,learning_rate=.05,l2_regularization=2.0),
        naive_role_logit=dict(C=.05,features="naive_drift",roles=sorted(NAIVE_ROLE_MODEL_NAMES))),indent=2))
    print("TRAIN",arm,len(d),d.attack.value_counts().to_dict(),flush=True)
    print("TRAIN_AUG",arm,len(aug),aug.attack.value_counts().to_dict(),flush=True)
def predict(d,models):
    s={n:m.predict_proba(matrix(d,c))[:,1] for n,(c,m) in models.items()}
    for n,v in s.items():assert np.isfinite(v).all(),n
    s["article_special"]=.5*(s["article_rf"]+s["article_gb"])
    # Runtime role is part of network inventory, not an attack label. Select
    # the corresponding frozen role expert without mixing incomparable scores.
    if all(name in s for name in NAIVE_ROLE_MODEL_NAMES.values()):
        role=d.ntype.astype(str).to_numpy()
        v=np.zeros(len(d),float)
        for r,name in NAIVE_ROLE_MODEL_NAMES.items():
            mask=role==r
            v[mask]=s[name][mask]
        s["comp_naive_role_aug"]=v
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

def _threshold_vector(d,value):
    if not isinstance(value,dict):return np.full(len(d),float(value))
    role=d.ntype.to_numpy()
    fallback=float(value.get("all",next(iter(value.values()))))
    return np.array([float(value.get(str(r),fallback)) for r in role],dtype=float)

def five_raw_channels(d,p,thresholds,suffix=""):
    # Four attack-specific experts plus the invariant compromise expert.
    # The compromise anchor may be role-specific because node type is directly
    # observable by the coordinator and has different benign score baselines.
    out={}
    for k in FIVE_CHANNELS:
        src=k+suffix if suffix else k
        out[k]=_logit(p[src])-_logit(_threshold_vector(d,thresholds[k]))
    return out

def calibrate_five_raw(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET,suffix=""):
    """Per-class calibration with a shared margin cut using the full FPR budget.

    First obtain conservative class-specific anchors. Then calibrate one common
    log-odds margin threshold on the fused negative score so the union, rather
    than each channel separately, can use the full raw-window budget. Finally
    choose persistence using only CAL worlds to satisfy the notification budget.
    """
    share=float(raw/len(FIVE_CHANNELS));thresholds={}
    for k in FIVE_CHANNELS:
        src=k+suffix if suffix else k
        if k=="invariant":
            by_role={}
            for role in sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames]))):
                vals=[]
                for d,p in zip(frames,preds):
                    mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
                    if mask.any():vals.append(threshold(p[src][mask],share,elapsed(d)))
                if vals:by_role[role]=float(max(vals))
            thresholds[k]=by_role
        else:
            thresholds[k]=float(max(
                threshold(p[src][d.label.to_numpy()==0],share,elapsed(d))
                for d,p in zip(frames,preds)
            ))

    prepared=[]
    raw_cuts=[]
    for d,p in zip(frames,preds):
        ch=five_raw_channels(d,p,thresholds,suffix)
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
    return dict(kind="five_raw",threshold=margin_cut,streak=int(chosen),model_suffix=suffix,
        raw_budget_share=share,channel_thresholds=thresholds,diagnostics=diagnostics)

def hybrid_raw_channels(d,p,thresholds):
    out={"fallback":_logit(p["local_aug"])-_logit(_threshold_vector(d,thresholds["fallback"]))}
    for k in FIVE_CHANNELS:
        out[k]=_logit(p[k+"_aug"])-_logit(_threshold_vector(d,thresholds[k]))
    return out

def calibrate_hybrid_raw(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    """Fallback local detector plus five augmented experts under one budget."""
    thresholds={}
    # Let the fallback use the full budget as an anchor; the subsequent fused
    # margin calibration enforces the actual union budget across all channels.
    thresholds["fallback"]=float(max(
        threshold(p["local_aug"][d.label.to_numpy()==0],raw,elapsed(d))
        for d,p in zip(frames,preds)))
    share=float(raw/len(FIVE_CHANNELS))
    for k in FIVE_CHANNELS:
        if k=="invariant":
            by_role={}
            for role in sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames]))):
                vals=[]
                for d,p in zip(frames,preds):
                    mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
                    if mask.any():vals.append(threshold(p[k+"_aug"][mask],share,elapsed(d)))
                if vals:by_role[role]=float(max(vals))
            thresholds[k]=by_role
        else:
            thresholds[k]=float(max(
                threshold(p[k+"_aug"][d.label.to_numpy()==0],share,elapsed(d))
                for d,p in zip(frames,preds)))

    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=hybrid_raw_channels(d,p,thresholds)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,ch))
    margin_cut=float(max(cuts))

    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,ch in prepared:
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="hybrid_raw",threshold=margin_cut,streak=int(chosen),
        channel_thresholds=thresholds,diagnostics=diagnostics)

COMP_SOPH_CHANNELS=("comp_naive_role","comp_statistical","comp_adaptive")
NAIVE_ROLE_MODEL_NAMES={r.value:f"comp_naive_{r.value.lower()}_lin" for r in F.NodeType}
NAIVE_CUSUM_K=24
NAIVE_CUSUM_Q=.50

def bounded_margin_cusum(d,margin,k,valid_mask=None):
    """Finite-memory positive CUSUM over causal same-node intervals."""
    x=np.asarray(margin,float).copy()
    if valid_mask is None:valid_mask=np.isfinite(x)
    else:valid_mask=np.asarray(valid_mask,bool)&np.isfinite(x)
    n=d.nid.to_numpy();t=d.t.to_numpy();N=len(x)
    inc=np.clip(np.where(valid_mask,x,0.),-3.,3.)
    boundary=np.r_[True,(n[1:]!=n[:-1])|(t[1:]!=t[:-1]+1)|(~valid_mask[1:])|(~valid_mask[:-1])]
    group=np.cumsum(boundary)
    best=np.where(valid_mask,np.maximum(inc,0.),-np.inf)
    accum=inc.copy()
    for lag in range(1,min(k,N)):
        same=(group[lag:]==group[:-lag])&valid_mask[lag:]&valid_mask[:-lag]
        accum[lag:]+=inc[:-lag]
        cand=np.where(same,accum[lag:],-np.inf)
        best[lag:]=np.maximum(best[lag:],cand)
    return best/max(k,1)

def fit_naive_cusum(frames,preds,raw=RAW_BUDGET):
    drifts={}
    roles=sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames])))
    for role in roles:
        qs=[]
        for d,p in zip(frames,preds):
            mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
            if mask.any():qs.append(float(np.quantile(_logit(p["comp_naive_aug"][mask]),NAIVE_CUSUM_Q)))
        if qs:drifts[role]=float(max(qs))
    scores=[];anchors=[]
    for d,p in zip(frames,preds):
        drift=_threshold_vector(d,drifts)
        margin=_logit(p["comp_naive_aug"])-drift
        neg=d.label.to_numpy()==0
        cs=bounded_margin_cusum(d,margin,NAIVE_CUSUM_K,valid_mask=neg)
        finite=cs[np.isfinite(cs)]
        anchors.append(threshold(finite,raw,elapsed(d)))
        scores.append(cs)
    anchor=float(max(anchors));scale=max(abs(anchor),.05)
    return dict(k=NAIVE_CUSUM_K,q=NAIVE_CUSUM_Q,drifts=drifts,anchor=anchor,scale=scale)

def naive_cusum_margin(d,p,rule,negative_only=False):
    drift=_threshold_vector(d,rule["drifts"])
    margin=_logit(p["comp_naive_aug"])-drift
    valid=(d.label.to_numpy()==0) if negative_only else None
    cs=bounded_margin_cusum(d,margin,rule["k"],valid_mask=valid)
    return (cs-rule["anchor"])/rule["scale"]


NAIVE_SELF_WIN=60
NAIVE_SELF_MIN=20

def _causal_robust_z(d,values,window=NAIVE_SELF_WIN,min_periods=NAIVE_SELF_MIN):
    """World-independent positive shift relative to each node's own past."""
    x=np.asarray(values,float);out=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        v=pd.Series(x[ix],index=ix,dtype=float)
        prev=v.shift(1)
        med=prev.rolling(window,min_periods=min_periods).median()
        q25=prev.rolling(window,min_periods=min_periods).quantile(.25)
        q75=prev.rolling(window,min_periods=min_periods).quantile(.75)
        scale=((q75-q25)/1.349).to_numpy(float)
        center=med.to_numpy(float)
        # Poisson-like floor for count-valued signals and a small absolute
        # floor for ratios prevent unstable z-scores when the IQR collapses.
        floor=np.where(np.nanmedian(np.abs(v.to_numpy(float)))>2.,
                       np.sqrt(np.maximum(np.abs(center),1.0)),.01)
        scale=np.maximum(np.nan_to_num(scale,nan=0.0),floor)
        out[ix]=(v.to_numpy(float)-center)/scale
    return np.nan_to_num(out,nan=0.0,posinf=0.0,neginf=0.0)

def naive_self_score(d):
    total=d["local_total_messages"].to_numpy(float)
    map_share=d["local_map_share"].to_numpy(float)
    sms_cols=[c for c in ("local_obs_ratio_SRI_SM","local_obs_ratio_MT_FSM","local_obs_ratio_MO_FSM") if c in d]
    if sms_cols:
        sms=np.nan_to_num(d[sms_cols].to_numpy(float),nan=0.0).sum(axis=1)
    else:
        sms=np.zeros(len(d),float)
    z_total=np.maximum(_causal_robust_z(d,total),0.0)
    z_map=np.maximum(_causal_robust_z(d,map_share),0.0)
    z_sms=np.maximum(_causal_robust_z(d,sms),0.0)
    # Naive compromise adds genuine SMS_MT traffic. Normal traffic multipliers
    # mostly change volume without a persistent MAP/SMS composition shift.
    return .15*z_total+.50*z_map+.35*z_sms

def fit_naive_selfnorm(frames,raw=RAW_BUDGET):
    roles=sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames])))
    cuts={};scales={}
    share=float(raw/8.0)
    for role in roles:
        vals=[];role_scores=[]
        for d in frames:
            score=naive_self_score(d)
            mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
            a=score[mask]
            if len(a):
                vals.append(threshold(a,share,elapsed(d)))
                role_scores.append(a)
        if vals:
            cuts[role]=float(max(vals))
            allv=np.concatenate(role_scores)
            q25,q75=np.quantile(allv,[.25,.75])
            scales[role]=float(max((q75-q25)/1.349,.25))
    return dict(window=NAIVE_SELF_WIN,min_periods=NAIVE_SELF_MIN,
                thresholds=cuts,scales=scales,raw_share=share)

def naive_selfnorm_margin(d,rule):
    score=naive_self_score(d)
    cut=_threshold_vector(d,rule["thresholds"])
    scale=_threshold_vector(d,rule["scales"])
    return (score-cut)/np.maximum(scale,.25)

def soph_selfnorm_channels(d,p,thresholds,self_rule):
    out=soph_hybrid_channels(d,p,thresholds)
    out["comp_naive_selfnorm"]=naive_selfnorm_margin(d,self_rule)
    return out

def calibrate_soph_selfnorm(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    base=calibrate_soph_hybrid(frames,preds,raw,notice)
    thresholds=base["channel_thresholds"]
    self_rule=fit_naive_selfnorm(frames,raw)
    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=soph_selfnorm_channels(d,p,thresholds,self_rule)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,p))
    margin_cut=float(max(cuts))
    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,p in prepared:
            ch=soph_selfnorm_channels(d,p,thresholds,self_rule)
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="soph_selfnorm",threshold=margin_cut,streak=int(chosen),
        channel_thresholds=thresholds,naive_selfnorm=self_rule,diagnostics=diagnostics)

def naive_load_shift_score(d,variant="total"):
    """Causal level-shift score for NAIVE compromise.

    Corrected compromise injects extra legitimate traffic chosen from the
    node's legal flow support. Therefore the only invariant observable for an
    unmasked (NAIVE) attacker is a sustained excess above the node's own past,
    optionally accompanied by a service/direction mix shift.
    """
    total=np.nan_to_num(d.get("drift_total_messages_z",pd.Series(np.zeros(len(d)))).to_numpy(float),nan=0.0)
    level=np.maximum(total,0.0)
    if variant=="total":
        return level
    cols=("drift_map_share_z","drift_isup_share_z","drift_n_unique_dest_z",
          "drift_dest_entropy_z","drift_dest_top1_ratio_z","drift_in_out_ratio_z",
          "drift_dlg_closure_ratio_z","drift_transit_share_z")
    mix=[np.abs(np.nan_to_num(d[c].to_numpy(float),nan=0.0)) for c in cols if c in d.columns]
    mix_score=np.maximum.reduce(mix) if mix else np.zeros(len(d),float)
    if variant=="total_mix":
        return .70*level+.30*mix_score
    raise ValueError(variant)

def fit_naive_load_shift(frames,variant,raw=RAW_BUDGET):
    roles=sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames])))
    cuts={};scales={}
    # The final fused margin is calibrated again below, so this channel can use
    # the full raw budget without a Bonferroni penalty.
    for role in roles:
        vals=[];allv=[]
        for d in frames:
            score=naive_load_shift_score(d,variant)
            mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
            a=score[mask]
            if len(a):
                vals.append(threshold(a,raw,elapsed(d)));allv.append(a)
        if vals:
            cuts[role]=float(max(vals))
            q=np.concatenate(allv);q25,q75=np.quantile(q,[.25,.75])
            scales[role]=float(max((q75-q25)/1.349,.25))
    return dict(variant=variant,thresholds=cuts,scales=scales)

def naive_load_shift_margin(d,rule):
    score=naive_load_shift_score(d,rule["variant"])
    cut=_threshold_vector(d,rule["thresholds"])
    scale=_threshold_vector(d,rule["scales"])
    return (score-cut)/np.maximum(scale,.25)

def soph_load_shift_channels(d,p,thresholds,rule):
    out=soph_hybrid_channels(d,p,thresholds)
    out["comp_naive_load_"+rule["variant"]]=naive_load_shift_margin(d,rule)
    return out

def calibrate_soph_load_shift(frames,preds,variant,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    base=calibrate_soph_hybrid(frames,preds,raw,notice)
    thresholds=base["channel_thresholds"]
    rule=fit_naive_load_shift(frames,variant,raw)
    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=soph_load_shift_channels(d,p,thresholds,rule)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,p))
    margin_cut=float(max(cuts))
    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,p in prepared:
            ch=soph_load_shift_channels(d,p,thresholds,rule)
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="soph_load_shift",threshold=margin_cut,streak=int(chosen),
        channel_thresholds=thresholds,naive_load_shift=rule,diagnostics=diagnostics)

ROLE_NAIVE_SPEC={
    "IGW":("obs_ratio_MT_FSM",60,"pos"),
    "SCP":("obs_ratio_ATI",30,"abs"),
    "SP":("obs_ratio_ATI",60,"pos"),
    "STP":("obs_ratio_ISD",60,"pos"),
}

def _node_roll_mean(d,x,win):
    x=np.asarray(x,float);out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(x[ix],index=ix).rolling(win,min_periods=max(10,win//3)).mean().fillna(0.0).to_numpy()
    return out

def naive_role_score(d):
    """Frozen RANK-selected causal score using delivered KIE features only."""
    roles=d.ntype.astype(str).to_numpy()
    out=np.zeros(len(d),float)
    for role,(feature,win,direction) in ROLE_NAIVE_SPEC.items():
        mask=roles==role
        if not mask.any() or feature not in d.columns:continue
        z=_causal_robust_z(d,d[feature].to_numpy(float))
        base=np.abs(z) if direction=="abs" else np.maximum(z,0.0)
        rolled=_node_roll_mean(d,base,win)
        out[mask]=rolled[mask]
    return out

def fit_naive_role(frames,raw=RAW_BUDGET):
    roles=sorted(ROLE_NAIVE_SPEC)
    cuts={};scales={}
    for role in roles:
        vals=[];allv=[]
        for d in frames:
            score=naive_role_score(d)
            mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
            a=score[mask]
            if len(a):
                vals.append(threshold(a,raw,elapsed(d)));allv.append(a)
        if vals:
            cuts[role]=float(max(vals))
            q=np.concatenate(allv);q25,q75=np.quantile(q,[.25,.75])
            scales[role]=float(max((q75-q25)/1.349,.05))
    return dict(spec=ROLE_NAIVE_SPEC,thresholds=cuts,scales=scales)

def naive_role_margin(d,rule):
    score=naive_role_score(d)
    cut=_threshold_vector(d,rule["thresholds"])
    scale=_threshold_vector(d,rule["scales"])
    return (score-cut)/np.maximum(scale,.05)

def soph_role_channels(d,p,thresholds,rule):
    out=soph_hybrid_channels(d,p,thresholds)
    out["comp_naive_role"]=naive_role_margin(d,rule)
    return out

def calibrate_soph_role(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    base=calibrate_soph_hybrid(frames,preds,raw,notice)
    thresholds=base["channel_thresholds"]
    rule=fit_naive_role(frames,raw)
    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=soph_role_channels(d,p,thresholds,rule)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,p))
    margin_cut=float(max(cuts))
    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,p in prepared:
            ch=soph_role_channels(d,p,thresholds,rule)
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="soph_role_naive",threshold=margin_cut,streak=int(chosen),
        channel_thresholds=thresholds,naive_role=rule,diagnostics=diagnostics)

def _count_threshold(x,allowed):
    """Strict threshold permitting at most *allowed* exceedances."""
    a=np.sort(np.asarray(x,float)[np.isfinite(x)])[::-1]
    if allowed<0:return float("inf")
    if not len(a):return float("inf")
    if allowed>=len(a):return float("-inf")
    return float(a[int(allowed)])

def _supplement_bucket_peaks(d,z,exclude):
    """Role-score peaks in node/buckets not already notified by base rule."""
    x=pd.DataFrame(dict(nid=d.nid.to_numpy(),bucket=d.t.to_numpy()//60,z=z))
    q=x.groupby(["nid","bucket"],sort=False).z.max().reset_index()
    if exclude:
        keep=np.array([(int(n),int(b)) not in exclude for n,b in zip(q.nid,q.bucket)],bool)
        q=q.loc[keep]
    return q.z.to_numpy(float)

def soph_role_supplement_channels(d,p,policy):
    base=soph_hybrid_channels(d,p,policy["channel_thresholds"])
    out={k:v-float(policy["base_threshold"]) for k,v in base.items()}
    out["comp_naive_role"]=naive_role_margin(d,policy["naive_role"])-float(policy["supplement_threshold"])
    return out

def calibrate_soph_role_supplement(frames,preds,base_raw,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    """Preserve the base detector and spend only its unused false-alarm budget.

    Feature selection is already frozen on RANK. CAL supplies thresholds only.
    The role branch is calibrated against negative windows/buckets not already
    consumed by the base branch, so it cannot raise the common base threshold.
    """
    base=calibrate_soph_hybrid(frames,preds,raw=float(base_raw),notice=notice)
    role_rule=fit_naive_role(frames,raw=raw)
    world_cuts=[];diagnostics=[]
    for d,p in zip(frames,preds):
        neg=d.label.to_numpy()==0
        base_ch=soph_hybrid_channels(d,p,base["channel_thresholds"])
        shifted={k:v-float(base["threshold"]) for k,v in base_ch.items()}
        bs,bz=combined(d,shifted,True,negative_only=True,streak_len=base["streak"])
        role=naive_role_margin(d,role_rule)
        rneg=np.where(neg,role,-np.inf)
        rz=streak_score(d,rneg,base["streak"])
        days=elapsed(d)

        raw_cap=int(np.floor(raw*days))
        base_raw_count=int((bs>0).sum())
        raw_left=max(0,raw_cap-base_raw_count)
        raw_extra=role[neg&(bs<=0)]
        raw_cut=_count_threshold(raw_extra,raw_left)

        bix=notify(d,bz,0.0)
        base_buckets={(int(d.nid.iloc[j]),int(d.t.iloc[j]//60)) for j in bix}
        notice_cap=int(np.floor(notice*days))
        notice_left=max(0,notice_cap-len(base_buckets))
        peaks=_supplement_bucket_peaks(d,rz,base_buckets)
        notice_cut=_count_threshold(peaks,notice_left)
        cut=max(raw_cut,notice_cut)
        world_cuts.append(cut)
        diagnostics.append(dict(base_raw_budget=float(base_raw),base_raw_count=base_raw_count,
            raw_cap=raw_cap,raw_left=raw_left,base_notifications=len(base_buckets),
            notice_cap=notice_cap,notice_left=notice_left,raw_cut=raw_cut,
            notice_cut=notice_cut,world_cut=cut))
    supp=float(max(world_cuts))
    return dict(kind="soph_role_supplement",threshold=0.0,streak=int(base["streak"]),
        base_raw_budget=float(base_raw),base_threshold=float(base["threshold"]),
        channel_thresholds=base["channel_thresholds"],naive_role=role_rule,
        supplement_threshold=supp,diagnostics=diagnostics)

def _profile_js_rows(d, recent=5, baseline=60, gap=5):
    """Causal Jensen-Shannon divergence of recent traffic mix from own history."""
    map_cols=[c for c in ("local_obs_ratio_"+m for m in MAP) if c in d.columns]
    out=np.zeros(len(d),float)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        map_share=np.nan_to_num(g["local_map_share"].to_numpy(float),nan=0.0)
        isup=np.nan_to_num(g["local_isup_share"].to_numpy(float),nan=0.0)
        parts=[]
        for c in map_cols:
            parts.append(map_share*np.nan_to_num(g[c].to_numpy(float),nan=0.0))
        parts.append(isup)
        used=isup.copy()
        if parts[:-1]:
            used+=np.sum(np.column_stack(parts[:-1]),axis=1)
        parts.append(np.clip(1.0-used,0.0,1.0))
        X=np.column_stack(parts)
        X=np.maximum(X,0.0)
        X/=np.maximum(X.sum(axis=1,keepdims=True),1e-12)

        frame=pd.DataFrame(X,index=ix)
        r=frame.rolling(recent,min_periods=max(2,recent//2)).mean()
        b=frame.shift(gap).rolling(baseline,min_periods=max(10,baseline//4)).mean()
        P=r.to_numpy(float).copy();Q=b.to_numpy(float).copy()
        P/=np.maximum(np.nansum(P,axis=1,keepdims=True),1e-12)
        Q/=np.maximum(np.nansum(Q,axis=1,keepdims=True),1e-12)
        M=.5*(P+Q)
        eps=1e-12
        kl1=np.nansum(np.where(P>0,P*np.log((P+eps)/(M+eps)),0.0),axis=1)
        kl2=np.nansum(np.where(Q>0,Q*np.log((Q+eps)/(M+eps)),0.0),axis=1)
        z=.5*(kl1+kl2)
        z[~np.isfinite(P).all(axis=1)|~np.isfinite(Q).all(axis=1)]=0.0
        out[ix]=z
    return out

def fit_naive_profile(frames,raw=RAW_BUDGET):
    roles=sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames])))
    cuts={}
    share=float(raw/8.0)
    for role in roles:
        vals=[]
        for d in frames:
            score=_profile_js_rows(d)
            mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
            if mask.any():vals.append(threshold(score[mask],share,elapsed(d)))
        if vals:cuts[role]=float(max(vals))
    return dict(thresholds=cuts,raw_share=share,recent=5,baseline=60,gap=5)

def naive_profile_margin(d,rule):
    score=_profile_js_rows(d,rule["recent"],rule["baseline"],rule["gap"])
    cut=_threshold_vector(d,rule["thresholds"])
    scale=np.maximum(cut,.005)
    return (score-cut)/scale

def soph_profile_channels(d,p,thresholds,profile_rule):
    out=soph_hybrid_channels(d,p,thresholds)
    out["comp_naive_profile"]=naive_profile_margin(d,profile_rule)
    return out

def calibrate_soph_profile(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    base=calibrate_soph_hybrid(frames,preds,raw,notice)
    thresholds=base["channel_thresholds"]
    profile_rule=fit_naive_profile(frames,raw)
    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=soph_profile_channels(d,p,thresholds,profile_rule)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,p))
    margin_cut=float(max(cuts))
    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,p in prepared:
            ch=soph_profile_channels(d,p,thresholds,profile_rule)
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="soph_profile",threshold=margin_cut,streak=int(chosen),
        channel_thresholds=thresholds,naive_profile=profile_rule,diagnostics=diagnostics)

NAIVE_ROUTING_WIN=60

def naive_routing_score(d,win=NAIVE_ROUTING_WIN):
    """Causal long-horizon drift in routing MAP procedures SRI/PRN."""
    cols=[c for c in ("local_obs_ratio_SRI","local_obs_ratio_PRN") if c in d.columns]
    if not cols:return np.zeros(len(d),float)
    z=[]
    for c in cols:
        z.append(np.maximum(_causal_robust_z(d,d[c].to_numpy(float)),0.0))
    base=np.mean(np.column_stack(z),axis=1)
    out=np.full(len(d),0.0)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy()
        out[ix]=pd.Series(base[ix],index=ix).rolling(win,min_periods=max(10,win//3)).mean().fillna(0.0).to_numpy()
    return out

def fit_naive_routing(frames,raw=RAW_BUDGET):
    roles=sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames])))
    cuts={};scales={}
    # Do not pre-split the budget here; the fused margin calibration below
    # enforces the final union budget across all channels.
    for role in roles:
        vals=[];allv=[]
        for d in frames:
            score=naive_routing_score(d)
            mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
            a=score[mask]
            if len(a):
                vals.append(threshold(a,raw,elapsed(d)));allv.append(a)
        if vals:
            cuts[role]=float(max(vals))
            q=np.concatenate(allv)
            q25,q75=np.quantile(q,[.25,.75])
            scales[role]=float(max((q75-q25)/1.349,.05))
    return dict(window=NAIVE_ROUTING_WIN,thresholds=cuts,scales=scales)

def naive_routing_margin(d,rule):
    score=naive_routing_score(d,rule["window"])
    cut=_threshold_vector(d,rule["thresholds"])
    scale=_threshold_vector(d,rule["scales"])
    return (score-cut)/np.maximum(scale,.05)

def soph_routing_channels(d,p,thresholds,routing_rule):
    out=soph_hybrid_channels(d,p,thresholds)
    out["comp_naive_routing"]=naive_routing_margin(d,routing_rule)
    return out

def calibrate_soph_routing(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    base=calibrate_soph_hybrid(frames,preds,raw,notice)
    thresholds=base["channel_thresholds"]
    routing_rule=fit_naive_routing(frames,raw)
    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=soph_routing_channels(d,p,thresholds,routing_rule)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,p))
    margin_cut=float(max(cuts))
    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,p in prepared:
            ch=soph_routing_channels(d,p,thresholds,routing_rule)
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="soph_routing",threshold=margin_cut,streak=int(chosen),
        channel_thresholds=thresholds,naive_routing=routing_rule,diagnostics=diagnostics)

def soph_hybrid_channels(d,p,thresholds):
    out={"fallback":_logit(p["local_aug"])-_logit(_threshold_vector(d,thresholds["fallback"]))}
    for k in OTHER:
        out[k]=_logit(p[k+"_aug"])-_logit(_threshold_vector(d,thresholds[k]))
    for k in COMP_SOPH_CHANNELS:
        out[k]=_logit(p[k+"_aug"])-_logit(_threshold_vector(d,thresholds[k]))
    return out

def calibrate_soph_hybrid(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    """General fallback + attack experts + three compromise-evasion experts."""
    thresholds={}
    thresholds["fallback"]=float(max(
        threshold(p["local_aug"][d.label.to_numpy()==0],raw,elapsed(d))
        for d,p in zip(frames,preds)))
    n_special=len(OTHER)+len(COMP_SOPH_CHANNELS)
    share=float(raw/n_special)
    for k in OTHER:
        thresholds[k]=float(max(
            threshold(p[k+"_aug"][d.label.to_numpy()==0],share,elapsed(d))
            for d,p in zip(frames,preds)))
    # Role-normalize each compromise expert; node role is observable at runtime.
    for k in COMP_SOPH_CHANNELS:
        by_role={}
        for role in sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames]))):
            vals=[]
            for d,p in zip(frames,preds):
                mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
                if mask.any():vals.append(threshold(p[k+"_aug"][mask],share,elapsed(d)))
            if vals:by_role[role]=float(max(vals))
        thresholds[k]=by_role

    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=soph_hybrid_channels(d,p,thresholds)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,ch))
    margin_cut=float(max(cuts))
    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,ch in prepared:
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="soph_hybrid",threshold=margin_cut,streak=int(chosen),
        channel_thresholds=thresholds,diagnostics=diagnostics)

CORE_COMP_CHANNELS=("comp_statistical","comp_adaptive")

def soph_core_channels(d,p,thresholds):
    """Stable cascade core: no NAIVE channel and therefore no naive-score shift."""
    out={"fallback":_logit(p["local_aug"])-_logit(_threshold_vector(d,thresholds["fallback"]))}
    for k in OTHER:
        out[k]=_logit(p[k+"_aug"])-_logit(_threshold_vector(d,thresholds[k]))
    for k in CORE_COMP_CHANNELS:
        out[k]=_logit(p[k+"_aug"])-_logit(_threshold_vector(d,thresholds[k]))
    return out

def calibrate_soph_core(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    """CAL-only fusion for common attacks plus statistical/adaptive compromise."""
    thresholds={}
    thresholds["fallback"]=float(max(
        threshold(p["local_aug"][d.label.to_numpy()==0],raw,elapsed(d))
        for d,p in zip(frames,preds)))
    n_special=len(OTHER)+len(CORE_COMP_CHANNELS)
    share=float(raw/n_special)
    for k in OTHER:
        thresholds[k]=float(max(
            threshold(p[k+"_aug"][d.label.to_numpy()==0],share,elapsed(d))
            for d,p in zip(frames,preds)))
    for k in CORE_COMP_CHANNELS:
        by_role={}
        for role in sorted(set(np.concatenate([d.ntype.astype(str).unique() for d in frames]))):
            vals=[]
            for d,p in zip(frames,preds):
                mask=(d.label.to_numpy()==0)&(d.ntype.astype(str).to_numpy()==role)
                if mask.any():vals.append(threshold(p[k+"_aug"][mask],share,elapsed(d)))
            if vals:by_role[role]=float(max(vals))
        thresholds[k]=by_role

    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=soph_core_channels(d,p,thresholds)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,ch))
    margin_cut=float(max(cuts))
    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,ch in prepared:
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="soph_core",threshold=margin_cut,streak=int(chosen),
        raw_budget=float(raw),channel_thresholds=thresholds,diagnostics=diagnostics)

def soph_cusum_channels(d,p,thresholds,cusum_rule,negative_only=False):
    out=soph_hybrid_channels(d,p,thresholds)
    out["comp_naive_cusum"]=naive_cusum_margin(d,p,cusum_rule,negative_only=negative_only)
    return out

def calibrate_soph_cusum(frames,preds,raw=RAW_BUDGET,notice=INCIDENT_BUDGET):
    base=calibrate_soph_hybrid(frames,preds,raw,notice)
    thresholds=base["channel_thresholds"]
    cusum_rule=fit_naive_cusum(frames,preds,raw)

    prepared=[];cuts=[]
    for d,p in zip(frames,preds):
        ch=soph_cusum_channels(d,p,thresholds,cusum_rule,negative_only=True)
        score,_=combined(d,ch,True,negative_only=True,streak_len=3)
        cuts.append(threshold(score[np.isfinite(score)],raw,elapsed(d)))
        prepared.append((d,p))
    margin_cut=float(max(cuts))
    diagnostics=[];chosen=None
    for length in (3,5,7,9,12):
        nr=[];wr=[]
        for d,p in prepared:
            ch=soph_cusum_channels(d,p,thresholds,cusum_rule,negative_only=True)
            score,z=combined(d,ch,True,negative_only=True,streak_len=length)
            wr.append(float((score>margin_cut).sum()/elapsed(d)))
            nr.append(float(len(notify(d,z,margin_cut))/elapsed(d)))
        diagnostics.append(dict(streak=length,worst_notifications_day=max(nr),
            worst_windows_day=max(wr),rates=nr))
        if max(nr)<=notice:
            chosen=length;break
    if chosen is None:chosen=12
    return dict(kind="soph_cusum",threshold=margin_cut,streak=int(chosen),
        channel_thresholds=thresholds,naive_cusum=cusum_rule,diagnostics=diagnostics)

def policy_channels(d,p,ref,name,policy):
    if policy.get("kind")=="five_raw":
        return five_raw_channels(d,p,policy["channel_thresholds"],policy.get("model_suffix","")),True,float(policy["threshold"])
    if policy.get("kind")=="hybrid_raw":
        return hybrid_raw_channels(d,p,policy["channel_thresholds"]),True,float(policy["threshold"])
    if policy.get("kind")=="soph_hybrid":
        return soph_hybrid_channels(d,p,policy["channel_thresholds"]),True,float(policy["threshold"])
    if policy.get("kind")=="soph_core":
        return soph_core_channels(d,p,policy["channel_thresholds"]),True,float(policy["threshold"])
    if policy.get("kind")=="soph_cusum":
        return soph_cusum_channels(d,p,policy["channel_thresholds"],policy["naive_cusum"]),True,float(policy["threshold"])
    if policy.get("kind")=="soph_selfnorm":
        return soph_selfnorm_channels(d,p,policy["channel_thresholds"],policy["naive_selfnorm"]),True,float(policy["threshold"])
    if policy.get("kind")=="soph_load_shift":
        return soph_load_shift_channels(d,p,policy["channel_thresholds"],policy["naive_load_shift"]),True,float(policy["threshold"])
    if policy.get("kind")=="soph_role_naive":
        return soph_role_channels(d,p,policy["channel_thresholds"],policy["naive_role"]),True,float(policy["threshold"])
    if policy.get("kind")=="soph_role_supplement":
        return soph_role_supplement_channels(d,p,policy),True,0.0
    if policy.get("kind")=="soph_profile":
        return soph_profile_channels(d,p,policy["channel_thresholds"],policy["naive_profile"]),True,float(policy["threshold"])
    if policy.get("kind")=="soph_routing":
        return soph_routing_channels(d,p,policy["channel_thresholds"],policy["naive_routing"]),True,float(policy["threshold"])
    n="integrated_raw" if name=="integrated_half" else name
    ch,confirm=channels(d,p,ref,n)
    return ch,confirm,float(policy["threshold"])

def class_name(k):
    return "compromise" if (k in ("invariant","behavior","article_special") or k.startswith("comp_")) else k

def evaluate(root,modelpath,out,arm):
    models=joblib.load(modelpath/"models.joblib")
    ranks=[load(root,arm,s)[0] for s in RANK];ref=fit_tails(ranks,[predict(d,models) for d in ranks])
    del ranks
    cal=[load(root,arm,s)[0] for s in CAL];ps=[predict(d,models) for d in cal]
    policies={n:cut_policy(cal,ps,ref,n) for n in COMMON_MODELS}
    policies["five_calibrated"]=calibrate_five_raw(cal,ps)
    if all((k+"_aug") in ps[0] for k in FIVE_CHANNELS):
        policies["five_augmented"]=calibrate_five_raw(cal,ps,suffix="_aug")
        if "local_aug" in ps[0]:
            policies["hybrid_augmented"]=calibrate_hybrid_raw(cal,ps)
            if all((k+"_aug") in ps[0] for k in COMP_SOPH_CHANNELS):
                policies["hybrid_soph_augmented"]=calibrate_soph_hybrid(cal,ps)
                for base_raw in (20.,18.,16.,14.,12.,10.,8.):
                    policies[f"hybrid_soph_core_{int(base_raw)}"]=calibrate_soph_core(cal,ps,base_raw)
                policies["hybrid_soph_cusum"]=calibrate_soph_cusum(cal,ps)
                policies["hybrid_soph_selfnorm"]=calibrate_soph_selfnorm(cal,ps)
                policies["hybrid_soph_load_total"]=calibrate_soph_load_shift(cal,ps,"total")
                policies["hybrid_soph_load_mix"]=calibrate_soph_load_shift(cal,ps,"total_mix")
                policies["hybrid_soph_role_naive"]=calibrate_soph_role(cal,ps)
                for base_raw in (20.,18.,16.):
                    policies[f"hybrid_soph_role_supp_{int(base_raw)}"]=calibrate_soph_role_supplement(cal,ps,base_raw)
                policies["hybrid_soph_profile"]=calibrate_soph_profile(cal,ps)
                policies["hybrid_soph_routing"]=calibrate_soph_routing(cal,ps)
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
        if a.seed is None or a.seed<0:raise ValueError("generate requires a non-negative --seed")
        d,e,cu=generate(a.seed,arm=a.arm);joblib.dump((d,e,cu),out/"world.joblib",compress=2)
        print("WORLD",a.arm,a.seed,len(d),d.attack.value_counts().to_dict(),cu,flush=True)
    elif a.mode=="train":train(Path(a.root),out,a.arm)
    elif a.mode=="evaluate":evaluate(Path(a.root),Path(a.models),out,a.arm)
    else:aggregate(Path(a.root),out)
if __name__=="__main__":main()
