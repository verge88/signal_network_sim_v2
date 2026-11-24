"""Scope SIP TICD early warnings to explicitly eligible network-function roles.

This adapter does NOT change the frozen physical or temporal detector.
Confirmed high-confidence alerts still apply to all SIP node types.
The early-warning branch is specialised for covert proxy compromise and
optionally closely related SIP signalling functions. The set of eligible
roles is network inventory metadata, not learned compromised-node labels.

CAVEAT: a role-restricted warning cannot detect early compromises of excluded
network-function types. Nor is role metadata trustworthy if an adversary can
compromise the operator's inventory. All upstream loss/CU measurements and
attack profiles in the current experiment are synthetic.
"""
from __future__ import annotations
import numpy as np
from ticd_zonal_warnings import ZoneContextTwoTierTICD

KNOWN_ROLES=frozenset(("PROXY","REDIRECT","REGISTRAR","UA_GW"))

class RoleScopedWarningTICD:
    """Frozen physical alert for all roles, role-scoped causal warning-only."""
    def __init__(self,source_model:ZoneContextTwoTierTICD,eligible_roles=None):
        if not isinstance(source_model,ZoneContextTwoTierTICD) or not source_model.fitted:
            raise TypeError("Expected already frozen normal-trained zonal TICD")
        if eligible_roles is not None:
            roles=frozenset(eligible_roles)
            if not roles or not roles<=KNOWN_ROLES:
                raise ValueError("Eligible SIP roles must be a nonempty known inventory subset")
            self.eligible_roles=tuple(sorted(roles))
        else:
            self.eligible_roles=None
        self.source_model=source_model
        self.confirmed_threshold=None
        self.warning_threshold=None
        self.fitted=False

    def eligible_mask(self,df):
        if "node_type" not in df:
            raise ValueError("Missing operator-known SIP function type")
        values=df["node_type"].astype(str).to_numpy()
        unknown=set(values)-KNOWN_ROLES
        if unknown:
            raise ValueError("Unexpected SIP function type: "+str(sorted(unknown)))
        if self.eligible_roles is None:
            return np.ones(len(df),dtype=bool)
        return np.isin(values,self.eligible_roles)

    def score_channels(self,df):
        # All features needed by the zonal score must be precomputed from a
        # complete same-time master zone snapshot; no test batch medians here.
        scores=self.source_model.score_channels(df)
        instant=np.asarray(scores["instant"],dtype=float)
        warning=np.asarray(scores["warning"],dtype=float)
        if not np.isfinite(instant).all() or not np.isfinite(warning).all():
            raise ValueError("Non-finite role-specific evidence")
        return {"instant":instant,"warning":warning}

    def freeze_thresholds(self,confirmed_threshold,warning_threshold):
        for value in (confirmed_threshold,warning_threshold):
            if not np.isfinite(float(value)) or float(value)<0.:
                raise ValueError("Expected nonnegative finite threshold")
        self.confirmed_threshold=float(confirmed_threshold)
        self.warning_threshold=float(warning_threshold)
        self.fitted=True
        return self

    def predict_levels(self,df):
        if not self.fitted:
            raise RuntimeError("Thresholds must be calibrated on benign reference")
        x=self.score_channels(df)
        eligible=self.eligible_mask(df)
        confirmed=x["instant"]>self.confirmed_threshold
        warned=(x["warning"]>self.warning_threshold)&eligible&(~confirmed)
        labels=np.zeros(len(df),dtype=np.int8)
        labels[warned]=1
        labels[confirmed]=2
        return labels

    def predict_masks(self,df):
        levels=self.predict_levels(df)
        return {"normal":levels==0,"warning_only":levels==1,
                "confirmed_alert":levels==2,"any_signal":levels>=1}

    def manifest(self):
        return {
            "eligible_roles":list(self.eligible_roles) if self.eligible_roles else "all_inventory_roles",
            "confirmed_scope":"all roles",
            "warning_scope":"only eligible operator inventory roles",
            "source_warning_mode":self.source_model.mode,
            "source_warning_strength":self.source_model.strength,
            "confirmed_threshold":self.confirmed_threshold,
            "warning_threshold":self.warning_threshold,
            "no_confirmed_alert_suppression":True,
            "warning_is_not_confirmed_detection":True,
            "scores_do_not_use_attack_labels":True,
            "test_batch_normalization":False,
            "limited_detection_for_excluded_roles":True
        }
