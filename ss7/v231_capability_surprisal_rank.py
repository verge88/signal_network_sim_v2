"""Deployment-normalized capability KIE v23.1 diagnostic.

The security invariant is unchanged: high-risk MAP operations are checked by an
independent authorization verifier. This detector normalizes missing-capability
counts by the node's attack-free commissioning mismatch rate and current checked
volume. It uses no attack labels for calibration.

A conservative 99% upper confidence bound for the benign mismatch probability
is estimated per node with a Jeffreys Beta prior. A 5-minute rolling count is
flagged only when:
  * at least 5 capability checks are missing, and
  * the Poisson upper-tail probability under that conservative benign rate is
    <= 1e-6.
The significance level is fixed from the desired network-wide false-window
budget, not tuned on RANK labels.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, poisson

from ss7 import fix as F
from ss7 import cu_compact_channel_development as C

BASELINE_END=int(os.getenv("SS7_DEPLOY_BASELINE","180"))
WINDOW=10
MIN_MISSING=5
ALPHA=1e-6
CONFIDENCE=.99
RAW_DAY=20.0
NOTICE_DAY=2.0
REFRACTORY=60
EXTERNAL=("sms_interception","location_tracking","signalling_dos","irsf")

def monitor_days(d):
    return max((int(d.t.max())-BASELINE_END+1)/F.INTERVALS_PER_DAY,
               1/F.INTERVALS_PER_DAY)

def clean_normal_mask(d,history=WINDOW):
    out=np.zeros(len(d),dtype=bool)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();a=g.label.to_numpy(int)
        contaminated=pd.Series(a,index=ix).rolling(
            history,min_periods=1).max().to_numpy()>0
        out[ix]=(a==0)&(~contaminated)&(g.t.to_numpy()>=BASELINE_END)
    return out

def capability_surprisal(d):
    pval=np.ones(len(d),float)
    roll_missing=np.full(len(d),np.nan)
    roll_checked=np.full(len(d),np.nan)
    base_rate=np.full(len(d),np.nan)
    for _,g in d.groupby("nid",sort=False):
        ix=g.index.to_numpy();tt=g.t.to_numpy()
        checked=g.kie_cap_checked.fillna(0.0).to_numpy(float)
        missing=g.kie_cap_missing.fillna(0.0).to_numpy(float)
        warm=tt<BASELINE_END
        n=float(checked[warm].sum());k=float(missing[warm].sum())
        if n<=0:continue
        # Jeffreys prior, one-sided 99% upper bound. This deliberately
        # overestimates the benign error rate rather than fitting to attacks.
        pu=float(beta.ppf(CONFIDENCE,k+.5,max(n-k,0.0)+.5))
        pu=max(pu,1e-9)
        rm=pd.Series(missing,index=ix).rolling(WINDOW,min_periods=4).sum().to_numpy()
        rc=pd.Series(checked,index=ix).rolling(WINDOW,min_periods=4).sum().to_numpy()
        lam=np.maximum(rc*pu,1e-9)
        pv=poisson.sf(np.maximum(rm-1.0,0.0),lam)
        pval[ix]=np.where(np.isfinite(rm)&np.isfinite(rc),pv,1.0)
        roll_missing[ix]=rm;roll_checked[ix]=rc;base_rate[ix]=pu
    score=-np.log10(np.clip(pval,1e-300,1.0))
    fire=(roll_missing>=MIN_MISSING)&(pval<=ALPHA)&(
        d.t.to_numpy()>=BASELINE_END)
    return score,fire,pval,roll_missing,roll_checked,base_rate

def notifications(d,fire):
    out=[];t=d.t.to_numpy();nid=d.nid.to_numpy()
    for node in np.unique(nid):
        ix=np.flatnonzero(nid==node);prev=False;last=-10**9
        for j in ix:
            cur=bool(fire[j])
            if cur and not prev and int(t[j])-last>=REFRACTORY:
                out.append(int(j));last=int(t[j])
            prev=cur
    return out

def evaluate(d,eligible):
    score,fire,pval,rm,rc,base_rate=capability_surprisal(d)
    clean=clean_normal_mask(d)
    normal=(d.label.to_numpy()==0)&(d.t.to_numpy()>=BASELINE_END)
    notes=notifications(d,fire)

    eps=[]
    q=d[d.episode_id>=0]
    for eid,g in q.groupby("episode_id",sort=False):
        ix=g.index.to_numpy();e=eligible.get(int(eid));hit=np.flatnonzero(fire[ix])
        eps.append(dict(
            episode=int(eid),attack=str(g.attack.iloc[0]),
            sophistication=str(g.sophistication.iloc[0]),
            role=str(g.ntype.iloc[0]),
            hidden_rate=float(e.hidden_rate) if e is not None and
                        getattr(e,"kind","")=="compromise" else np.nan,
            hit=int(len(hit)>0),
            delay_seconds=float(hit[0]*30) if len(hit) else np.nan,
            peak_score=float(np.nanmax(score[ix])),
            peak_missing=float(np.nanmax(rm[ix]))))
    ep=pd.DataFrame(eps)
    comp=ep.attack=="compromise";naive=comp&(ep.sophistication=="naive")
    stat=comp&(ep.sophistication=="statistical")
    adap=comp&(ep.sophistication=="adaptive");ext=ep.attack.isin(EXTERNAL)
    days=monitor_days(d)
    m=dict(
        compromise_recall=float(ep.loc[comp,"hit"].mean()),
        naive_recall=float(ep.loc[naive,"hit"].mean()),
        statistical_recall=float(ep.loc[stat,"hit"].mean()),
        adaptive_recall=float(ep.loc[adap,"hit"].mean()),
        external_episode_activation=float(ep.loc[ext,"hit"].mean()),
        false_windows_day=float((fire&clean).sum()/days),
        false_notifications_day=float(sum(bool(clean[j]) for j in notes)/days),
        notifications_day=float(len(notes)/days),
        base_rate_q50=float(np.nanquantile(base_rate,.50)),
        base_rate_q90=float(np.nanquantile(base_rate,.90)),
        clean_peak_score=float(np.nanmax(score[clean])) if clean.any() else np.nan,
        clean_peak_missing=float(np.nanmax(rm[clean])) if clean.any() else np.nan)
    return ep,m

def main(root,out,seeds):
    rows=[];episodes=[]
    for seed in map(int,seeds):
        d,eligible,_=C.load(root,"compact",seed)
        ep,m=evaluate(d,eligible)
        m["seed"]=seed;rows.append(m);ep["seed"]=seed;episodes.append(ep)
    df=pd.DataFrame(rows);ed=pd.concat(episodes,ignore_index=True)
    summary=dict(
        mean_compromise_recall=float(df.compromise_recall.mean()),
        min_compromise_recall=float(df.compromise_recall.min()),
        mean_naive_recall=float(df.naive_recall.mean()),
        min_naive_recall=float(df.naive_recall.min()),
        mean_statistical_recall=float(df.statistical_recall.mean()),
        min_statistical_recall=float(df.statistical_recall.min()),
        mean_adaptive_recall=float(df.adaptive_recall.mean()),
        min_adaptive_recall=float(df.adaptive_recall.min()),
        worst_external_episode_activation=float(df.external_episode_activation.max()),
        worst_false_windows_day=float(df.false_windows_day.max()),
        worst_false_notifications_day=float(df.false_notifications_day.max()),
        alpha=ALPHA,min_missing=MIN_MISSING,confidence=CONFIDENCE)
    summary["budget_ok"]=bool(
        summary["worst_false_windows_day"]<=RAW_DAY and
        summary["worst_false_notifications_day"]<=NOTICE_DAY)
    summary["accepted"]=bool(
        summary["budget_ok"] and summary["min_naive_recall"]>=.90 and
        summary["min_statistical_recall"]>=.90 and
        summary["min_adaptive_recall"]>=.90 and
        summary["worst_external_episode_activation"]<=.05)

    byrole=ed[ed.attack=="compromise"].groupby("role").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_delay_seconds=("delay_seconds","mean")).reset_index()
    bysoph=ed[ed.attack=="compromise"].groupby("sophistication").agg(
        episodes=("episode","count"),recall=("hit","mean"),
        mean_delay_seconds=("delay_seconds","mean")).reset_index()
    byattack=ed.groupby("attack").agg(
        episodes=("episode","count"),activation=("hit","mean")).reset_index()

    df.to_csv(out/"worlds.csv",index=False);ed.to_csv(out/"episodes.csv",index=False)
    byrole.to_csv(out/"by_role.csv",index=False)
    bysoph.to_csv(out/"by_sophistication.csv",index=False)
    byattack.to_csv(out/"by_attack.csv",index=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print("WORLDS\n",df.to_string(index=False),flush=True)
    print("BY_ROLE\n",byrole.to_string(index=False),flush=True)
    print("BY_SOPH\n",bysoph.to_string(index=False),flush=True)
    print("BY_ATTACK\n",byattack.to_string(index=False),flush=True)
    print("SUMMARY",json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    main(a.root,a.output,a.seeds)
