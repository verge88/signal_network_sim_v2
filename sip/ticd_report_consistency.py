"""Causal fusion of TICD levels, SIP method accounting and independent audit counts."""
import numpy as np
import pandas as pd

from ticd_sequential_physical import sequence_features
from ticd_correlated_block_evidence import block_features

METHODS = ("invite", "ack", "bye", "register", "options", "cancel", "subscribe", "notify", "other")
RATIOS = ["obs_ratio_" + name for name in METHODS]
KEYS = ["simulation_seed", "node_id", "interval"]


def report_votes(frame, normalize=False, ratio_resolution=.001):
    """Compare fields from a single delivered CU report with an exhaustive method taxonomy."""
    if not np.isfinite(ratio_resolution) or not 0 <= ratio_resolution <= 1:
        raise ValueError("ratio_resolution must be a finite fraction in [0, 1]")
    share = frame[RATIOS].to_numpy(dtype=float, copy=True)
    if normalize:
        share = np.maximum(share, 0)
        total = share.sum(axis=1, keepdims=True)
        share = share / np.where(total > 0, total, 1)
    total = share.sum(axis=1)
    largest = share.max(axis=1)
    remainder = np.maximum(.001, total - largest)
    dominance = frame.obs_method_dominance_ratio.to_numpy()
    # Cross multiplication avoids amplifying rounding at small denominators.
    disagreement = np.abs(dominance * remainder - largest)
    # The sum of nine independently rounded fractions has a known error bound.
    rounding = .5 * len(RATIOS) * ratio_resolution + 1e-12
    bad_total = np.abs(total - 1) > max(.012, rounding)
    bad_dominance = disagreement > .07 * remainder + max(.005, rounding) * (1 + np.abs(dominance))
    # The INVITE/ACK ratio and the two method shares have the same report window.
    # This identity survives renormalization of the entire histogram.
    invite_ack = frame.obs_invite_ack_ratio.to_numpy()
    ack = np.maximum(.001, share[:, METHODS.index("ack")])
    pair_error = np.abs(share[:, METHODS.index("invite")] - invite_ack * ack)
    single_rounding = .5 * ratio_resolution + 1e-6
    bad_pair = pair_error > .060 * ack + single_rounding * (1 + np.abs(invite_ack))
    valid = frame.primary_observer_valid & frame.cu_delivered.astype(bool)
    finite = np.isfinite(share).all(axis=1) & np.isfinite(dominance)
    return (bad_total | bad_dominance | (bad_pair & np.isfinite(invite_ack))) & valid.to_numpy() & finite


def extra_evidence(raw, normalize=False, ratio_resolution=.001):
    """Expect one row per node and interval. Reset memory after gaps or observer loss."""
    x = raw.sort_values(KEYS)
    if not x.index.is_unique or x.duplicated(KEYS).any():
        raise ValueError("duplicate observation key or index")
    groups = [x[k] for k in KEYS[:2]]
    gap = x.interval.groupby(groups, sort=False).diff().ne(1)
    primary_reset = gap | ~x.primary_observer_valid
    segment = primary_reset.groupby(groups, sort=False).cumsum()

    def rolling(values, keys, width):
        values = pd.Series(values, index=x.index)
        return values.groupby(keys, sort=False).transform(
            lambda s: s.rolling(width, min_periods=1).sum())

    votes = rolling(report_votes(x, normalize, ratio_resolution), groups + [segment], 6)
    report = x.primary_observer_valid & votes.ge(2)
    # This audit is a separate synthetic observer. Never infer exposure from a missing sample.
    n = x.audit_observed_count
    k = x.audit_mismatch_count
    if not (np.isfinite(n) & np.isfinite(k) & n.ge(k) & k.ge(0)).all():
        raise ValueError("invalid policy audit counts")
    # A missed poll adds zero information; the finite windows expire stale evidence.
    # An actual gap in the input timeline still starts a new audit segment.
    audit_keys = groups + [gap.groupby(groups, sort=False).cumsum()]
    log_lr = k * np.log(.045 / .013) + (n - k) * np.log(.955 / .987)
    short = rolling(log_lr, audit_keys, 16)
    long = rolling(log_lr, audit_keys, 32)
    audit = n.gt(0) & (short.gt(9) | long.gt(12))
    return pd.DataFrame({"report": report, "audit": audit}).reindex(raw.index)


class ReportConsistencyTICD:
    def __init__(self, reference, block_model, ratio_resolution=.001):
        self.ratio_resolution = ratio_resolution
        self.reference = reference
        self.block_model = block_model

    def baseline(self, raw, commissioning):
        x = self.reference.prepare(raw, commissioning)
        x = sequence_features(x, commissioning, self.reference)
        x = block_features(x, commissioning, self.reference)
        levels = pd.Series(self.block_model.predict_levels(x), index=x.index).reindex(raw.index)
        if levels.isna().any():
            raise ValueError("unaligned base predictions")
        return np.where(raw.primary_observer_valid, levels, 0).astype(np.int8)

    def predict_levels(self, raw, commissioning):
        base = self.baseline(raw, commissioning)
        confirm = extra_evidence(raw, ratio_resolution=self.ratio_resolution).any(axis=1)
        return np.where(confirm, 2, base).astype(np.int8)
