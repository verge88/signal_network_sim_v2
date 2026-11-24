"""Paired observability audit, not a trained detector or an attack recipe.

Compare synthetic observations at identical node/time keys under matched
ordinary event schedules, with/without the modeled compromise intervention.
Report generation and peer measurements are those of the existing simulator.
The paired worlds share initial topology/flows and traffic seeds. At positive
intensity, changing the number of simulated transactions can change downstream
random draws: paired differences are diagnostics, not isolated causal effects.
Ground-truth state is used only for this audit, never as detector input.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from ss7 import fix as F
from ss7.report_only_invariant_study import NEW, report_only

SEEDS=(8501,8502,8503)
RATES=(0.,.05,.4,2.5)
DAYS=.35
N_COMPROMISE=8

def simulate(cfg,template,events,positives):
    sim=F.Ss7Sim(cfg,F.RngBook(cfg.seed),topo=template.topo,
                size_capacities=False,template=template)
    rows=[]
    for t in range(cfg.n_intervals):
        state,hidden,meta=sim.step_true(t,events)
        reps=sim.step_reports(t,state,hidden,meta,events)
        old=F.all_invariants(reps,sim.topo,meta["lstate"])
        new=report_only(reps,sim.topo)
        active={e.nid for e in positives if e.t0<=t<=e.t1}
        for n,r in reps.items():
            if sim.topo.nodes[n].role==F.NodeRole.MASTER:continue
            row={"nid":n,"t":t,"role":sim.topo.nodes[n].ntype.value,
                 "active":n in active,"hidden_observed_in_sim":int(n in hidden)}
            row.update({f"own/{k}":v for k,v in r.feats.items() if k in F.OWN})
            row.update({f"old/{k}":old[n][k] for k in F.INV})
            row.update({f"new/{k}":new[n][k] for k in NEW})
            truth=F.derived(state[n])
            row.update({f"truth/{k}":truth[k] for k in F.OWN})
            rows.append(row)
    df=pd.DataFrame(rows).set_index(["nid","t"]).sort_index()
    assert df.index.is_unique
    return df

def audit(seed,out):
    cfg=F.SimConfig(seed=seed,days=DAYS)
    book=F.RngBook(seed); template=F.Ss7Sim(cfg,book)
    episodes=F.make_episodes(template,cfg,book["episode"],
        n_compromise=N_COMPROMISE,n_normal=12,n_ext=4)
    pos=[e for e in episodes if e.kind=="compromise"]
    ordinary=[e for e in episodes if e.kind!="compromise"]
    base=simulate(cfg,template,ordinary,pos)
    cols=[c for c in base if "/" in c]
    # Normalize paired shifts by the benign-world, same-node temporal SD.
    # Constant/missing reference dimensions are explicitly marked unavailable.
    sd=base.groupby(level="nid")[cols].transform("std")
    sd=sd.where(sd>1e-12)
    records=[];checks=[]
    for h in RATES:
        varied=[replace(e,hidden_rate=h) for e in pos]
        d=simulate(cfg,template,ordinary+varied,varied)
        if not d.index.equals(base.index):raise AssertionError("Unpaired node/time keys")
        if not np.array_equal(d.active,base.active):raise AssertionError("Changed evaluation support")
        same=np.array_equal(d[cols].to_numpy(),base[cols].to_numpy(),equal_nan=True)
        if h==0 and not same:raise AssertionError("Zero-intensity paired feature equality failed")
        checks.append({"seed":seed,"h":h,"paired_windows":len(d),
                       "active_windows":int(d.active.sum()),
                       "sim_hidden_windows":int(d.hidden_observed_in_sim.sum()),
                       "exact_equality":bool(same)})
        dif=d[cols]-base[cols];norm=dif.abs()/sd
        for role in sorted(d.role.unique()):
            for active in (False,True):
                mask=(d.role==role)&(d.active==active)
                if not mask.any():continue
                for c in cols:
                    a=dif.loc[mask,c].dropna()
                    b=norm.loc[mask,c].dropna()
                    records.append(dict(seed=seed,h=h,role=role,active=active,
                        feature=c,n_pairs=len(a),n_normalizable=len(b),
                        mean_signed=float(a.mean()) if len(a) else np.nan,
                        median_abs=float(a.abs().median()) if len(a) else np.nan,
                        fraction_changed=float((a!=0).mean()) if len(a) else np.nan,
                        median_abs_sd=float(b.median()) if len(b) else np.nan))
        print("paired",seed,h,len(d),int(d.active.sum()),"exact",same,flush=True)
    pd.DataFrame(records).to_csv(out/"paired_shifts.csv",index=False)
    (out/"checks.json").write_text(json.dumps(checks,indent=2))
    (out/"audit_scope.json").write_text(json.dumps({
        "seed":seed,"days":DAYS,"rates":RATES,"generated_compromise":len(pos),
        "units":"same node/time pair, descriptive only",
        "normalization":"same-node control-world temporal standard deviation",
        "positive_intensity_rng_caveat":True,
        "nonvictim_windows_may_include_affected_peers":True,
        "ordinary_schedule_identical":True},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    audit(a.seed,a.output)
