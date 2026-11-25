"""Causal detection of replayed report profiles from fresh Diameter CU polls."""
from collections import Counter, deque

import numpy as np
import pandas as pd


MODEL_NAME = "Cascade + report recurrence"
PROFILE_COLUMNS = (
    tuple(f"obs_ratio_s6a_{code}" for code in ("air", "ulr", "clr", "idr", "dsr", "pur", "rsr", "nor"))
    + tuple(f"obs_ratio_gx_{code}" for code in ("ccr_i", "ccr_u", "ccr_t", "rar"))
)
REQUIRED_COLUMNS = {"node_id", "interval", "cu_delivered", *PROFILE_COLUMNS}


def validate_compromise_state(frame: pd.DataFrame) -> None:
    expected = frame["attack_type"].eq("node_compromise")
    state = frame["is_compromised"].eq(1)
    if not expected.equals(state):
        mismatches = int((expected != state).sum())
        raise ValueError(
            f"{mismatches} windows have inconsistent compromise state and labels. "
            "Regenerate the dataset with the corrected simulator."
        )


def report_recurrence(frame: pd.DataFrame) -> np.ndarray:
    """Score each row using only the current report and earlier rows of its node.

    A matching command profile must have appeared in a fresh poll within
    the previous 60 intervals. Aggregate matches over 40 intervals, requiring
    15 informative deliveries. Omitted intervals break the history.
    """
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing observable report fields: {sorted(missing)}")

    ordered = frame.reset_index(drop=True).sort_values(["node_id", "interval"])
    out = np.zeros(len(frame), dtype=float)
    for _, node in ordered.groupby("node_id", sort=False):
        times = node["interval"].to_numpy()
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Intervals must be finite and unique within each node")

        profiles = node[list(PROFILE_COLUMNS)].round(9)
        informative = (
            node["cu_delivered"].eq(1).to_numpy()
            & profiles.ge(0).all(axis=1).to_numpy()
            & profiles.le(1).all(axis=1).to_numpy()
            & profiles.gt(0).sum(axis=1).ge(3).to_numpy()
        )
        # Hash the complete joint profile rather than individual commands.
        keys = pd.util.hash_pandas_object(profiles, index=False).to_numpy()
        boundaries = np.flatnonzero(np.diff(times) != 1) + 1
        for segment in np.split(np.arange(len(node)), boundaries):
            history = deque()
            frequencies = Counter()
            hits = np.full(len(segment), np.nan)
            for offset, row in enumerate(segment):
                if not informative[row]:
                    continue
                while history and history[0][0] < offset - 60:
                    _, expired = history.popleft()
                    frequencies[expired] -= 1
                    if not frequencies[expired]:
                        del frequencies[expired]
                key = keys[row]
                hits[offset] = float(key in frequencies)
                history.append((offset, key))
                frequencies[key] += 1
            score = pd.Series(hits).rolling(40, min_periods=15).mean().fillna(0)
            out[node.index.to_numpy()[segment]] = score.to_numpy()
    return out
