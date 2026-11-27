"""Two auxiliary causal convolutional heads for five-class SS7 CU detection.

ROCKET-inspired compact random kernels (Dempster et al., 2020).
The simulator and original specialist heads are imported without modification.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ss7 import five_specialist_development as S

TRAIN = S.TRAIN
TEST = tuple(range(21101, 21109))
TARGETS = ("irsf", "location_track")
BASE = "special_context_equal"
VARIANTS = (BASE, "level_aux", "kernel_aux")
CHANNELS = tuple(dict.fromkeys((*S.CONTEXT, *S.TEMPORAL)))


def kernels():
    rng = np.random.default_rng(21000)
    bank = []
    for _ in range(96):
        n = int(rng.choice([7, 9, 11]))
        w = rng.normal(size=n)
        w -= w.mean()
        bank.append((rng.choice(len(CHANNELS), 2, replace=False),
                     w / np.linalg.norm(w), int(rng.choice([1, 2, 4])),
                     rng.uniform(-1, 1)))
    return bank


def temporal(d, values, bank):
    """Causal convolution and trailing max/positive-proportion pooling."""
    nid, t = d.nid.to_numpy(), d.t.to_numpy()
    cuts = np.flatnonzero((nid[1:] != nid[:-1]) | (np.diff(t) != 1)) + 1
    result = np.zeros((len(d), len(bank)*2), dtype=np.float32)
    for indices in np.split(np.arange(len(d)), cuts):
        if not len(indices):
            continue
        data = values[indices]
        for j, (channels, weights, dilation, bias) in enumerate(bank):
            signal = (data[:, channels[0]] + data[:, channels[1]]) / np.sqrt(2.)
            taps = np.zeros((len(weights)-1)*dilation + 1)
            taps[::dilation] = weights
            response = np.convolve(signal, taps, mode="full")[:len(indices)] + bias
            response[:len(taps)-1] = np.nan
            trace = pd.Series(response)
            positive = trace.gt(0).astype(float).where(trace.notna())
            result[indices, 2*j] = trace.rolling(16, min_periods=1).max().fillna(0.)
            result[indices, 2*j+1] = positive.rolling(16, min_periods=1).mean().fillna(0.)
    return result


def normalized(d, norm):
    x = d.loc[:, list(CHANNELS)].to_numpy(dtype=np.float32)
    return np.clip(np.nan_to_num((x-norm[0])/norm[1], nan=0., posinf=0., neginf=0.), -12, 12)


def train(out):
    frames = []
    for seed in TRAIN:
        d = S.generate(seed)[0]
        cols = ["nid", "t", "label", "episode_id", "attack", *CHANNELS]
        frames.append(d[list(dict.fromkeys(cols))].copy())
        del d
    rest = pd.concat([d.loc[d.label == 0, list(CHANNELS)].iloc[::20] for d in frames])
    med = rest.median().fillna(0.).to_numpy(dtype=np.float32)
    scale = rest.std().fillna(1.).clip(lower=.01).to_numpy(dtype=np.float32)
    norm = (med, scale)
    bank = kernels()
    parts, labels, ids = [], [], []
    for seed, d in zip(TRAIN, frames, strict=True):
        x = normalized(d, norm)
        pos = np.flatnonzero(d.label.to_numpy() == 1)
        neg = np.flatnonzero(d.label.to_numpy() == 0)
        rng = np.random.default_rng(seed)
        ix = np.r_[pos, rng.choice(neg, min(len(neg), 25000), replace=False)]
        parts.append(np.column_stack((x[ix], temporal(d,x,bank)[ix])))
        labels.extend(np.where(d.label == 1, d.attack, "normal")[ix])
        ids.extend(str(seed)+":"+d.episode_id.iloc[ix].astype(str))
    x = np.concatenate(parts)
    y, episodes = np.asarray(labels), pd.Series(ids)
    models = {}
    for variant in VARIANTS[1:]:
        width = len(CHANNELS) if variant == "level_aux" else x.shape[1]
        scaler = StandardScaler().fit(x[:, :width])
        features = scaler.transform(x[:, :width])
        heads = []
        for label in TARGETS:
            target = y == label
            w = np.ones(len(y))
            counts = episodes[target].value_counts()
            w[target] = 1 / episodes[target].map(counts).to_numpy()
            for mask in (target, ~target):
                w[mask] *= len(w)/(2*w[mask].sum())
            heads.append(LogisticRegression(C=.1, max_iter=500).fit(features,target,sample_weight=w))
        models[variant] = (width, scaler, heads)
    joblib.dump((norm,bank,models),out/"aux.joblib",compress=3)


def observe(d, original, aux):
    norm, bank, models = aux
    x = normalized(d, norm)
    features = np.column_stack((x, temporal(d,x,bank)))
    scores = {BASE: [S.score(d, original["context_"+label]) for label in S.CLASSES]}
    for name, (width, scaler, heads) in models.items():
        z = scaler.transform(features[:, :width])
        scores[name] = [m.predict_proba(z)[:,1] for m in heads]
    return d[["nid","t","label","episode_id"]].copy(), scores


def test(seed, original, aux, out):
    if seed not in TEST:
        raise ValueError("unexpected test seed")
    d, sim, eps, _ = S.generate(seed)
    eligible = {i:e for i,e in enumerate(eps) if e.kind in S.CLASSES}
    if len(eligible) != 200:
        raise ValueError("incomplete test design")
    for i,e in eligible.items():
        q = d[(d.episode_id==i)&(d.nid==e.nid)]
        if len(q)!=e.t1-e.t0+1 or q.t.nunique()!=len(q):
            raise ValueError("incomplete episode")
    test_data = observe(d,original,aux)
    del d
    refs = [observe(S.generate(seed+offset, balanced, sim, stress)[0],original,aux)
            for offset,balanced,stress in ((10000,False,False),(20000,False,True),(40000,True,False))]
    frames, outputs = zip(*refs, test_data)
    rows, events = [], []
    for name in VARIANTS:
        channels = list(zip(*(s[BASE] for s in outputs)))
        weights = np.full(5,.2)
        if name != BASE:
            channels.extend(zip(*(s[name] for s in outputs)))
            weights = np.r_[weights*.8, .1, .1]
        margins, _, _ = S.fuse(frames[:3],channels,weights)
        raw = margins[-1].max(axis=1)
        z, _ = S.confirm(frames[-1],margins[-1])
        metrics, hits = S.metrics(frames[-1],raw,z,0.,eligible)
        rows.append(dict(model=name,seed=seed,**metrics))
        events.extend(dict(model=name,seed=seed,episode=i,attack=e.kind,detected=int(i in hits),
                           delay_seconds=hits.get(i)) for i,e in eligible.items())
        print(seed,name,metrics,flush=True)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(events).to_csv(out/"episodes.csv",index=False)


def aggregate(root,out):
    m,e,s = S.collect(root,TEST,VARIANTS)
    attacks = e.groupby(["model","attack"]).detected.agg(["size","sum","mean"])
    s["target_met"] = (s.recall > .90) & (s.both_budgets == len(TEST))
    worlds = m.pivot(index="seed",columns="model",values="recall")
    samples = np.random.default_rng(21100).integers(0,len(TEST),(20000,len(TEST)))
    contrasts = []
    for baseline in (BASE,"level_aux"):
        delta = (worlds.kernel_aux-worlds[baseline]).to_numpy()
        lo,hi = np.quantile(delta[samples].mean(axis=1),[.025,.975])
        contrasts.append(dict(baseline=baseline,gain=delta.mean(),lo=lo,hi=hi))
    for name,data in (("summary",s),("episodes",e),("worlds",m),("paired",pd.DataFrame(contrasts))):
        data.to_csv(out/(name+".csv"),index=False)
    attacks.to_csv(out/"class_recall.csv")
    print(s.to_string(index=False),attacks.to_string(),contrasts,sep="\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage",choices=("train","test","aggregate"))
    p.add_argument("--seed",type=int)
    p.add_argument("--frozen",type=Path)
    p.add_argument("--models",type=Path)
    p.add_argument("--input",type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=="train":
        train(a.output)
    elif a.stage=="test":
        test(a.seed,joblib.load(a.frozen),joblib.load(a.models),a.output)
    else:
        aggregate(a.input,a.output)
