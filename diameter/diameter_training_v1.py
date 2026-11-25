"""
Diameter Master-Slave Anomaly Detection Training Pipeline v1
=============================================================
Adapted from SS7 pipeline for Diameter protocol specifics.

Models:
  1. Isolation Forest (unsupervised)
  2. One-Class SVM (unsupervised)
  3. Autoencoder (unsupervised)
  4. Variational Autoencoder (unsupervised)
  5. LSTM-Autoencoder (unsupervised, sequential)
  6. Temporal-Attention-AE (unsupervised, sequential)
  7. Random Forest (supervised)
  8. Gradient Boosting (supervised)
  9. Cascade Compromise Detector (supervised, 2-level)
  10. Specialized Ensemble (unsupervised, per-attack-type)
  11. Meta-Ensemble (stacking)

Plus ablation studies for MS-features and temporal features.
"""

import argparse
from pathlib import Path

if __package__:
    from .diameter_report_recurrence import (
        MODEL_NAME as RECURRENCE_MODEL, REQUIRED_COLUMNS as RECURRENCE_COLUMNS,
        report_recurrence, validate_compromise_state,
    )
else:
    from diameter_report_recurrence import (
        MODEL_NAME as RECURRENCE_MODEL, REQUIRED_COLUMNS as RECURRENCE_COLUMNS,
        report_recurrence, validate_compromise_state,
    )

import numpy as np
import pandas as pd
import os
import json
import random
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from sklearn.base import clone
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.facecolor': 'white',
    'axes.facecolor': '#f2f2f2', 'axes.grid': True,
    'grid.alpha': 0.35, 'grid.linestyle': '--',
})

# Diameter-specific command lists (must match simulator)
S6A_COMMANDS = ["AIR", "ULR", "CLR", "IDR", "DSR", "PUR", "RSR", "NOR"]
GX_COMMANDS = ["CCR_I", "CCR_U", "CCR_T", "RAR"]


# ══════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    data_path: str = "diameter_dataset_v1.csv"
    validation_path: Optional[str] = None
    stack_path: Optional[str] = None
    test_path: Optional[str] = None
    additional_test_paths: Tuple[str, ...] = ()
    curve_train_fraction: Optional[float] = None
    compare_tree_reference: bool = False
    results_dir: str = "results_diameter_v1"
    plots_dir: str = "results_diameter_v1/plots"
    test_size: float = 0.25
    val_size: float = 0.15
    seed: int = 42

    # Unsupervised models
    iso_forest_trees: int = 500
    iso_forest_contamination: float = 0.05  # retained for experiment metadata
    iso_forest_max_features: float = 1.0
    iso_forest_bootstrap: bool = False
    iso_forest_clip_lower_q: float = 0.001
    iso_forest_clip_upper_q: float = 0.999
    ocsvm_nu: float = 0.05
    ocsvm_kernel: str = "rbf"

    # Neural models — compact capacity and aggressive validation guards.
    ae_hidden_dims: List[int] = field(default_factory=lambda: [40, 20])
    ae_latent_dim: int = 8
    ae_epochs: int = 60
    ae_batch: int = 256
    ae_lr: float = 4e-4
    ae_dropout: float = 0.10
    ae_weight_decay: float = 2e-4
    ae_patience: int = 4
    ae_min_delta: float = 5e-4

    # VAE — stronger latent regularisation and short overfit tail.
    vae_hidden_dims: List[int] = field(default_factory=lambda: [40, 20])
    vae_latent_dim: int = 8
    vae_epochs: int = 80
    vae_batch: int = 256
    vae_lr: float = 4e-4
    vae_kl_weight: float = 0.60
    vae_dropout: float = 0.10
    vae_weight_decay: float = 2e-4
    vae_patience: int = 4
    vae_min_delta: float = 5e-4

    # LSTM-AE — smaller recurrent state and earlier plateau stop.
    lstm_hidden: int = 28
    lstm_layers: int = 2
    lstm_seq_len: int = 10
    lstm_epochs: int = 55
    lstm_batch: int = 256
    lstm_lr: float = 5e-4
    lstm_dropout: float = 0.35
    lstm_weight_decay: float = 2e-4
    lstm_patience: int = 5
    lstm_min_delta: float = 5e-4

    # Temporal-attention AE — reduced width + stronger dropout.
    tat_hidden: int = 40
    tat_heads: int = 4
    tat_layers: int = 1
    tat_dropout: float = 0.30
    tat_epochs: int = 65
    tat_batch: int = 256
    tat_lr: float = 2e-4
    tat_weight_decay: float = 2e-4
    tat_patience: int = 5
    tat_min_delta: float = 5e-4

    # Supervised models — constrain tree capacity to reduce memorisation.
    rf_trees: int = 450
    rf_max_depth: int = 12
    rf_min_samples_leaf: int = 8
    rf_min_samples_split: int = 16
    rf_max_features: str = "sqrt"
    rf_max_samples: float = 0.75

    gb_estimators: int = 260
    gb_lr: float = 0.035
    gb_max_depth: int = 3
    gb_min_samples_leaf: int = 24
    gb_subsample: float = 0.80

    # Supervised decision thresholds are tuned only on validation labels,
    # never on test, with an explicit false-positive-rate budget.
    supervised_max_fpr: float = 0.02

    # Cascade / node-compromise branch.
    cascade_rf_trees: int = 500
    cascade_gb_estimators: int = 300
    compromise_gb_positive_weight: float = 6.0
    compromise_general_fpr_budget: float = 0.004
    compromise_total_fpr_budget: float = 0.020
    recurrence_calibration_budget: float = 0.010
    compromise_target_recall: float = 0.85

    threshold_percentile: float = 95.0


# ══════════════════════════════════════════════════════════
#  Data Pipeline — Diameter features
# ══════════════════════════════════════════════════════════

class DataPipeline:
    """
    Feature engineering for Diameter signaling data.

    Feature groups (analogous to SS7 but with Diameter-specific fields):
    1. CU interaction (11 features) — same concept as SS7
    2. Reported by slave (19+ features) — Diameter interfaces & commands
    3. Temporal statistics (7 features) — same concept
    4. Spatial/zone features (13 features) — same concept
    """

    # ── Build feature column lists dynamically ──

    # Group 1: CU interaction features (11)
    CU_FEATURES = [
        "cu_delivered", "cu_rtt_delay_ms", "cu_req_delay_ms",
        "cu_resp_delay_ms", "cu_combined_loss_prob", "integrity_check",
        "consecutive_cu_losses", "cu_processing_delay_ms",
        "cu_response_time_jitter", "cu_staleness", "cu_consistency_error",
    ]

    # Group 2: Reported traffic features
    # Interface counts
    REPORTED_COUNTS = [
        "obs_total_messages", "obs_s6a_count", "obs_gx_count",
        "obs_rx_count", "obs_s13_count", "obs_other_count",
    ]
    REPORTED_STATS = [
        "obs_entropy", "obs_inbound_outbound_ratio",
        "obs_international_fraction", "obs_n_unique_peers",
        "obs_s6a_dominance_ratio", "obs_s6a_top2_ratio", "obs_s6a_gini",
    ]
    # S6a command ratios
    S6A_RATIO_FEATURES = [f"obs_ratio_s6a_{c.lower()}" for c in S6A_COMMANDS]
    # Gx command ratios
    GX_RATIO_FEATURES = [f"obs_ratio_gx_{c.lower()}" for c in GX_COMMANDS]

    # Group 3: Z-score features
    Z_FEATURES = [
        "z_obs_total_messages", "z_obs_s6a_count",
        "z_obs_s6a_dominance_ratio",
        "z_obs_ratio_s6a_ulr", "z_obs_ratio_s6a_idr",
    ]

    # Group 4: Temporal features (7)
    TEMPORAL_FEATURES = [
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio",
        "var_obs_s6a_dominance_ratio",
        "autocorr_obs_total_messages",
        "integrity_fail_rate", "obs_peers_cv",
    ]

    # Group 5: Zone deviation features
    ZONE_FEATURES = [
        "dev_total_messages_vs_zone", "dev_s6a_count_vs_zone",
        "dev_gx_count_vs_zone",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone",
        "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
    ]

    GROUND_TRUTH_COLS = [
        "gt_total_messages", "gt_s6a_count", "gt_gx_count",
        "gt_rx_count", "gt_s13_count", "gt_other_count",
        "gt_entropy", "gt_inbound_outbound_ratio",
        "gt_international_fraction", "gt_n_unique_peers",
    ]

    LABEL_COL = "is_anomaly"
    ATTACK_COL = "attack_type"

    # Compromise-specific features (subset for cascade stage 2)
    COMPROMISE_FEATURES = [
        "obs_inbound_outbound_ratio", "integrity_check",
        "integrity_fail_rate", "cu_consistency_error",
        "cu_processing_delay_ms", "cu_response_time_jitter",
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio", "var_obs_s6a_dominance_ratio",
        "autocorr_obs_total_messages", "obs_peers_cv",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone",
        "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
        "consecutive_cu_losses", "cu_staleness",
        "dev_total_messages_vs_zone", "dev_s6a_count_vs_zone",
    ]

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_cols: List[str] = []
        self.compromise_scaler = StandardScaler()
        self.compromise_imputer = SimpleImputer(strategy="median")
        self.compromise_feature_cols: List[str] = []

    def _build_feature_cols(self, df: pd.DataFrame):
        """Build feature column lists from what's actually in the data."""
        all_candidate = (
            self.CU_FEATURES + self.REPORTED_COUNTS + self.REPORTED_STATS +
            self.S6A_RATIO_FEATURES + self.GX_RATIO_FEATURES +
            self.Z_FEATURES + self.ZONE_FEATURES + self.TEMPORAL_FEATURES
        )
        self.feature_cols = [c for c in all_candidate if c in df.columns]
        if "node_type_enc" in df.columns:
            self.feature_cols.append("node_type_enc")

        self.compromise_feature_cols = [
            c for c in self.COMPROMISE_FEATURES if c in df.columns
        ]
        if "node_type_enc" in df.columns:
            self.compromise_feature_cols.append("node_type_enc")

    def load_and_prepare(self):
        config = self.config
        optional_paths = (config.validation_path, config.stack_path, config.test_path)
        if any(optional_paths) or config.additional_test_paths:
            if not all(optional_paths):
                raise ValueError("Provide validation, stacking and test files together")
            paths = [Path(p).resolve() for p in (
                config.data_path, *optional_paths, *config.additional_test_paths)]
            if len(set(paths)) != len(paths):
                raise ValueError("Each partition requires a separate simulation file")
            simulations = [pd.read_csv(path) for path in paths]
            if any(simulations[i].equals(simulations[j])
                   for i in range(len(simulations)) for j in range(i)):
                raise ValueError("Duplicate simulations in evaluation partitions")
            test_names = [p.stem.removesuffix(".csv") for p in paths[3:]]
            if len(set(test_names)) != len(test_names):
                raise ValueError("Test simulation filenames must be distinct")
            test_frames = []
            next_node_id = 0
            for name, frame in zip(test_names, simulations[3:]):
                if set(frame[self.LABEL_COL].unique()) != {0, 1}:
                    raise ValueError(f"{name} needs both benign and anomalous records")
                if len(test_names) > 1:
                    # Give each simulation its own node namespace for both
                    # sequence construction and report recurrence.
                    codes, nodes = pd.factorize(frame["node_id"], sort=True)
                    if (codes < 0).any():
                        raise ValueError(f"Missing node IDs in {name}")
                    frame = frame.assign(node_id=codes + next_node_id)
                    next_node_id += len(nodes)
                test_frames.append(frame.assign(_test_simulation=name))
            partitions = (*simulations[:3], pd.concat(test_frames, ignore_index=True))
            self.split_protocol = "separate simulation files"
        else:
            df = pd.read_csv(config.data_path)
            # Keep all nodes in each time block in the same partition.
            blocks = df["interval"] // 120
            block_labels = df.groupby(blocks)[self.LABEL_COL].max()
            fit_val, test = train_test_split(
                block_labels, test_size=config.test_size,
                stratify=block_labels, random_state=config.seed)
            fit, val_stack = train_test_split(
                fit_val, test_size=config.val_size / (1 - config.test_size),
                stratify=fit_val, random_state=config.seed)
            validation, stack = train_test_split(
                val_stack, test_size=0.5, stratify=val_stack, random_state=config.seed)
            partitions = tuple(df[blocks.isin(group.index)].copy()
                               for group in (fit, validation, stack, test))
            self.split_protocol = "disjoint 30-minute blocks of one simulation"
            partitions[-1]["_test_simulation"] = "test"

        train = partitions[0]
        if "node_type" in train and "node_type_enc" not in train:
            codes = {name: i for i, name in enumerate(sorted(train["node_type"].unique()))}
            for df in partitions:
                df["node_type_enc"] = df["node_type"].map(codes).fillna(-1)
        self._build_feature_cols(train)
        if any(c.startswith("gt_") for c in self.feature_cols):
            raise ValueError("Ground-truth columns found in feature set")
        for name, df in zip(("train", "validation", "stack", "test"), partitions):
            if "is_compromised" in df:
                validate_compromise_state(df)
            if set(df[self.LABEL_COL].unique()) != {0, 1}:
                raise ValueError(f"{name} needs both benign and anomalous records")
            print(f"{name}: {len(df)} rows; anomaly rate {df[self.LABEL_COL].mean():.4f}")
        print(f"Split protocol: {self.split_protocol}")
        print(f"Observable features: {len(self.feature_cols)}")
        return partitions

    def get_features_labels(self, df, fit=False):
        X = df[self.feature_cols].copy()
        if fit:
            X_imp = self.imputer.fit_transform(X)
            X_sc = self.scaler.fit_transform(X_imp)
        else:
            X_imp = self.imputer.transform(X)
            X_sc = self.scaler.transform(X_imp)
        return X_sc, df[self.LABEL_COL].values

    def get_compromise_features(self, df, fit=False):
        X = df[self.compromise_feature_cols].copy()
        if fit:
            X_imp = self.compromise_imputer.fit_transform(X)
            X_sc = self.compromise_scaler.fit_transform(X_imp)
        else:
            X_imp = self.compromise_imputer.transform(X)
            X_sc = self.compromise_scaler.transform(X_imp)
        y = (df[self.ATTACK_COL] == "node_compromise").astype(int).values
        return X_sc, y

    def get_normal_data(self, df, fit=False):
        normal = df[df[self.LABEL_COL] == 0]
        return self.get_features_labels(normal, fit=fit)

    def get_sequential_data(self, df, seq_len, include_simulation=False):
        ordered = df.sort_values(["node_id", "interval"], kind="stable")
        X_all, y_all = self.get_features_labels(ordered)
        seqs, labels, attack_types, benign, simulations = [], [], [], [], []
        for positions in ordered.groupby("node_id", sort=False).indices.values():
            if len(positions) < seq_len:
                continue
            rows = ordered.iloc[positions]
            times = rows["interval"].to_numpy()
            if np.any(np.diff(times) <= 0):
                raise ValueError("Duplicate interval for a Diameter node")
            contiguous = times[seq_len - 1:] - times[:len(times) - seq_len + 1] == seq_len - 1
            windows = np.lib.stride_tricks.sliding_window_view(
                X_all[positions].astype(np.float32), seq_len, axis=0).swapaxes(1, 2)
            y_node = y_all[positions]
            seqs.append(windows[contiguous])
            labels.append(y_node[seq_len - 1:][contiguous])
            attack_types.append(rows[self.ATTACK_COL].to_numpy()[seq_len - 1:][contiguous])
            benign.append(~np.lib.stride_tricks.sliding_window_view(
                y_node.astype(bool), seq_len).any(axis=1)[contiguous])
            if include_simulation:
                simulations.append(rows["_test_simulation"].to_numpy()[seq_len - 1:][contiguous])
        if not seqs:
            empty = (np.empty((0, seq_len, X_all.shape[1]), dtype=np.float32),
                     np.array([], dtype=int), np.array([], dtype=str), np.array([], dtype=bool))
            return (*empty, np.array([], dtype=str)) if include_simulation else empty
        groups = (seqs, labels, attack_types, benign)
        return tuple(np.concatenate(parts) for parts in (
            (*groups, simulations) if include_simulation else groups))


# ══════════════════════════════════════════════════════════
#  MS (Master-Slave) Feature List for Ablation
# ══════════════════════════════════════════════════════════

MS_FEATURES = (
    DataPipeline.CU_FEATURES +
    DataPipeline.REPORTED_COUNTS +
    ["obs_inbound_outbound_ratio"] +
    [f for f in DataPipeline.ZONE_FEATURES] +
    DataPipeline.TEMPORAL_FEATURES
)


# ══════════════════════════════════════════════════════════
#  Feature Groups for Specialized Ensemble
# ══════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "s6a_profile": (
        DataPipeline.S6A_RATIO_FEATURES +
        ["obs_s6a_dominance_ratio", "obs_s6a_top2_ratio", "obs_s6a_gini",
         "z_obs_s6a_dominance_ratio", "z_obs_ratio_s6a_ulr",
         "z_obs_ratio_s6a_idr"]
    ),
    "gx_profile": DataPipeline.GX_RATIO_FEATURES,
    "volume_timing": [
        "obs_total_messages", "obs_s6a_count", "obs_gx_count",
        "obs_rx_count", "obs_s13_count", "obs_entropy",
        "z_obs_total_messages", "z_obs_s6a_count",
    ],
    "control_unit": DataPipeline.CU_FEATURES,
    "international": [
        "obs_international_fraction", "obs_n_unique_peers",
    ],
    "baseline_deviation": [
        "dev_total_messages_vs_zone", "dev_s6a_count_vs_zone",
        "dev_gx_count_vs_zone", "obs_inbound_outbound_ratio",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone",
    ],
    "compromise_indicators": DataPipeline.COMPROMISE_FEATURES,
}

ATTACK_LABELS = {
    'location_tracking': 'Отслеживание',
    'sms_data_interception': 'Перехват SMS/данных',
    'signaling_dos': 'Сигнальный DoS',
    'fraud_profile': 'Мошенничество (IDR)',
    'node_compromise': 'Компрометация узла',
}


# ══════════════════════════════════════════════════════════
#  Neural Network Models
# ══════════════════════════════════════════════════════════

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.0):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc.extend([
                nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h),
                nn.Dropout(dropout),
            ])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([
                nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h),
                nn.Dropout(dropout),
            ])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.0):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc.extend([
                nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h),
                nn.Dropout(dropout),
            ])
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([
                nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h),
                nn.Dropout(dropout),
            ])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decoder(z), mu, lv


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, seq_len,
                 dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers,
                               batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                               batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_in = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(dec_in, (h, c))
        return self.output_layer(dec_out)


class TemporalAttentionAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=4, n_layers=2,
                 seq_len=10, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.output_proj(self.transformer(self.input_proj(x)))


# ══════════════════════════════════════════════════════════
#  Trainers
# ══════════════════════════════════════════════════════════

class AETrainer:
    def __init__(self, model, lr, epochs, batch_size, patience=15,
                 weight_decay=1e-4, min_delta=1e-4, grad_clip=1.0,
                 device=None):
        self.model = model
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model.to(self.device)
        self.train_losses, self.train_eval_losses, self.val_losses = [], [], []

    def fit(self, X_train, X_val=None):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2,
            threshold=self.min_delta, min_lr=max(self.lr * 0.05, 1e-6))
        crit = nn.MSELoss()
        dl = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=self.batch_size, shuffle=True)
        best_val, significant_val = float("inf"), float("inf")
        pat_cnt, best_state = 0, None
        self.best_epoch = None
        self.train_losses, self.train_eval_losses, self.val_losses = [], [], []

        for epoch in range(self.epochs):
            self.model.train()
            tl = 0.0
            for (batch,) in dl:
                batch = batch.to(self.device)
                loss = crit(self.model(batch), batch)
                opt.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                opt.step()
                tl += loss.item() * len(batch)
            tl /= len(X_train)
            self.train_losses.append(tl)

            # Compare train and validation in identical eval mode.  The old
            # curve mixed dropout-active training loss with dropout-disabled
            # validation loss, which overstated/obscured generalization gaps.
            train_eval = self._eval(X_train, crit)
            self.train_eval_losses.append(train_eval)
            if X_val is not None:
                vl = self._eval(X_val, crit)
                self.val_losses.append(vl)
                sched.step(vl)
                if vl < best_val:
                    best_val, self.best_epoch = vl, epoch + 1
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                if vl < significant_val - self.min_delta:
                    significant_val, pat_cnt = vl, 0
                else:
                    pat_cnt += 1
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}: "
                          f"train_eval={train_eval:.6f}, val={vl:.6f}")
                if pat_cnt >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def _eval(self, X, crit):
        self.model.eval()
        total, count = 0.0, 0
        dl = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            batch_size=max(self.batch_size, 1024), shuffle=False)
        with torch.no_grad():
            for (batch,) in dl:
                batch = batch.to(self.device)
                loss = crit(self.model(batch), batch)
                total += loss.item() * len(batch)
                count += len(batch)
        return total / max(count, 1)

    def reconstruction_error(self, X):
        self.model.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(X), 2048):
                t = torch.as_tensor(X[start:start + 2048], dtype=torch.float32, device=self.device)
                err = torch.mean((t - self.model(t)) ** 2, dim=-1)
                if err.dim() > 1:
                    err = err.mean(dim=-1)
                errors.append(err.cpu().numpy())
        return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)


class VAETrainer:
    def __init__(self, model, lr, epochs, batch_size, kl_weight=0.5,
                 patience=15, weight_decay=1e-4, min_delta=1e-4,
                 grad_clip=1.0, device=None):
        self.model = model
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.kl_weight, self.patience = kl_weight, patience
        self.weight_decay = weight_decay
        self.min_delta = min_delta
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model.to(self.device)
        self.train_losses, self.train_eval_losses, self.val_losses = [], [], []

    def _loss(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.kl_weight * kl, recon, kl

    def fit(self, X_train, X_val=None):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2,
            threshold=self.min_delta, min_lr=max(self.lr * 0.05, 1e-6))
        dl = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=self.batch_size, shuffle=True)
        best_val, significant_val = float("inf"), float("inf")
        pat_cnt, best_state = 0, None
        self.best_epoch = None
        self.train_losses, self.train_eval_losses, self.val_losses = [], [], []

        for epoch in range(self.epochs):
            self.model.train()
            tl = 0.0
            for (batch,) in dl:
                batch = batch.to(self.device)
                x_hat, mu, lv = self.model(batch)
                loss, _, _ = self._loss(batch, x_hat, mu, lv)
                opt.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                opt.step()
                tl += loss.item() * len(batch)
            tl /= len(X_train)
            self.train_losses.append(tl)

            train_eval = self._eval_val(X_train)
            self.train_eval_losses.append(train_eval)
            if X_val is not None:
                vl = self._eval_val(X_val)
                self.val_losses.append(vl)
                sched.step(vl)
                if vl < best_val:
                    best_val, self.best_epoch = vl, epoch + 1
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                if vl < significant_val - self.min_delta:
                    significant_val, pat_cnt = vl, 0
                else:
                    pat_cnt += 1
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}: "
                          f"train_eval={train_eval:.6f}, val={vl:.6f}")
                if pat_cnt >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def _eval_val(self, X):
        # Deterministic and batched evaluation: use posterior mean instead of
        # resampling z, and avoid version/device-dependent full-tensor peaks.
        self.model.eval()
        total, count = 0.0, 0
        dl = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            batch_size=max(self.batch_size, 1024), shuffle=False)
        with torch.no_grad():
            for (batch,) in dl:
                batch = batch.to(self.device)
                mu, lv = self.model.encode(batch)
                x_hat = self.model.decoder(mu)
                loss, _, _ = self._loss(batch, x_hat, mu, lv)
                total += loss.item() * len(batch)
                count += len(batch)
        return total / max(count, 1)

    def anomaly_score(self, X):
        self.model.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(X), 2048):
                t = torch.as_tensor(X[start:start + 2048], dtype=torch.float32, device=self.device)
                mu, lv = self.model.encode(t)
                x_hat = self.model.decoder(mu)
                recon = torch.mean((t - x_hat) ** 2, dim=-1)
                kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=-1)
                errors.append((recon + self.kl_weight * kl).cpu().numpy())
        return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)


# ══════════════════════════════════════════════════════════
#  Robust Isolation-Forest preprocessing
# ══════════════════════════════════════════════════════════

def fit_iso_robust_bounds(X_normal, lower_q=0.001, upper_q=0.999):
    """Fit wide winsorization bounds on benign TRAINING observations only."""
    X = np.asarray(X_normal, dtype=float)
    low = np.nanquantile(X, lower_q, axis=0)
    high = np.nanquantile(X, upper_q, axis=0)
    return low, high


def apply_iso_robust_bounds(X, low, high):
    """Clip representation artifacts without using validation/test statistics."""
    X = np.asarray(X, dtype=float)
    return np.minimum(np.maximum(X, low), high)


# ══════════════════════════════════════════════════════════
#  Evaluation Utilities
# ══════════════════════════════════════════════════════════

def find_threshold(scores, y_true, percentile=95.0):
    normal_scores = scores[y_true == 0]
    th_pct = np.percentile(normal_scores, percentile)
    prec, rec, ths = precision_recall_curve(y_true, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    th_f1 = ths[min(np.argmax(f1s), len(ths) - 1)]
    y_pct = (scores >= th_pct).astype(int)
    y_f1 = (scores >= th_f1).astype(int)
    if f1_score(y_true, y_f1, zero_division=0) >= \
       f1_score(y_true, y_pct, zero_division=0):
        return th_f1, "max_f1"
    return th_pct, "percentile"


def select_supervised_threshold(scores, y_true, max_fpr=0.02):
    """Choose the validation threshold with maximum F1 under an FPR budget.

    This uses validation labels only.  The returned threshold is frozen before
    the untouched test split is scored.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, scores)
    positives = max(1, int((y_true == 1).sum()))
    negatives = max(1, int((y_true == 0).sum()))

    tp = tpr_arr * positives
    fp = fpr_arr * negatives
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tpr_arr
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    feasible = np.isfinite(thresholds) & (fpr_arr <= max_fpr)
    if not feasible.any():
        # Conservative fallback: threshold at the validation benign quantile
        # corresponding to the FPR budget.
        return float(np.quantile(
            scores[y_true == 0], 1.0 - max_fpr)), "validation_fpr_quantile"

    idxs = np.flatnonzero(feasible)
    best = idxs[int(np.argmax(f1s[feasible]))]
    return float(thresholds[best]), "validation_max_f1_with_fpr_cap"


def select_compromise_aware_thresholds(
        p_general, p_compromise, y_any, y_compromise,
        general_fpr_budget=0.012, total_fpr_budget=0.020,
        target_recall=0.85):
    """Select two validation-only thresholds for the cascade.

    The general branch receives only part of the false-positive budget.  The
    remaining budget is allocated to the node-compromise branch.  The
    compromise threshold is made as permissive as validation benign FPR
    allows, which maximises compromise recall without touching the test split.
    """
    p_general = np.asarray(p_general, dtype=float)
    p_compromise = np.asarray(p_compromise, dtype=float)
    y_any = np.asarray(y_any, dtype=int)
    y_compromise = np.asarray(y_compromise, dtype=int)

    th_general, general_method = select_supervised_threshold(
        p_general, y_any, general_fpr_budget)
    base_pred = p_general >= th_general
    benign = y_any == 0
    comp = y_compromise == 1
    n_benign = max(1, int(benign.sum()))

    max_fp = int(np.floor(total_fpr_budget * n_benign))
    base_fp = int(np.sum(base_pred & benign))
    remaining_fp = max(0, max_fp - base_fp)

    candidates = np.sort(p_compromise[benign & ~base_pred])
    if remaining_fp <= 0 or len(candidates) == 0:
        th_compromise = float(np.nextafter(
            np.max(p_compromise) if len(p_compromise) else 1.0, np.inf))
    elif remaining_fp >= len(candidates):
        th_compromise = float(np.nextafter(candidates[0], -np.inf))
    else:
        # Strictly above the cutoff avoids exceeding the budget when values tie.
        cutoff = candidates[-remaining_fp]
        th_compromise = float(np.nextafter(cutoff, np.inf))

    combined = base_pred | (p_compromise >= th_compromise)
    validation_fpr = float(np.mean(combined[benign])) if benign.any() else 0.0
    validation_compromise_recall = (
        float(np.mean(combined[comp])) if comp.any() else 0.0)
    validation_f1 = f1_score(y_any, combined, zero_division=0)

    # Normalised score keeps a single threshold=1.0 for downstream metrics.
    eps = 1e-12
    comp_den = th_compromise if np.isfinite(th_compromise) else 1e12
    score = np.maximum(
        p_general / max(th_general, eps),
        p_compromise / max(comp_den, eps),
    )

    info = {
        "general_threshold": float(th_general),
        "compromise_threshold": float(th_compromise),
        "general_threshold_method": general_method,
        "validation_fpr": validation_fpr,
        "validation_compromise_recall": validation_compromise_recall,
        "validation_f1": float(validation_f1),
        "target_compromise_recall": float(target_recall),
        "target_met": bool(validation_compromise_recall >= target_recall),
        "general_fpr_budget": float(general_fpr_budget),
        "total_fpr_budget": float(total_fpr_budget),
        "base_benign_fp": base_fp,
        "allowed_benign_fp": max_fp,
    }
    return float(th_general), float(th_compromise), score, info


def combine_cascade_scores(p_general, p_compromise,
                           general_threshold, compromise_threshold):
    eps = 1e-12
    comp_den = (
        compromise_threshold if np.isfinite(compromise_threshold) else 1e12)
    return np.maximum(
        np.asarray(p_general, dtype=float) / max(general_threshold, eps),
        np.asarray(p_compromise, dtype=float) / max(comp_den, eps),
    )


def calibrate_cascade_across_domains(domains, config, fpr_budget):
    """Use the most conservative cutoff for each branch across validation days."""
    if not 0 < config.compromise_general_fpr_budget <= fpr_budget <= config.compromise_total_fpr_budget:
        raise ValueError("Invalid cascade calibration FPR budget")
    cutoffs = [
        select_compromise_aware_thresholds(
            general, specific, y, compromised,
            general_fpr_budget=config.compromise_general_fpr_budget,
            total_fpr_budget=fpr_budget,
            target_recall=config.compromise_target_recall,
        )[:2]
        for _, general, specific, y, compromised in domains
    ]
    general_cut, specific_cut = map(max, zip(*cutoffs))
    measured = {}
    for name, general, specific, y, compromised in domains:
        alerts = combine_cascade_scores(general, specific, general_cut, specific_cut) >= 1
        measured[name] = {
            "benign_fpr": float(alerts[y == 0].mean()),
            "compromise_recall": (float(alerts[compromised == 1].mean())
                                  if (compromised == 1).any() else None),
        }
    return {
        "thresholds": [general_cut, specific_cut],
        "calibration_fpr_budget": fpr_budget,
        "total_fpr_budget": config.compromise_total_fpr_budget,
        "target_compromise_recall": config.compromise_target_recall,
        "per_simulation": measured,
    }


def evaluate_model(name, scores, y_true, attack_types,
                   threshold=None, percentile=95.0):
    if threshold is None:
        threshold, method = find_threshold(scores, y_true, percentile)
    else:
        method = "provided"
    y_pred = (scores >= threshold).astype(int)
    auc_roc = (roc_auc_score(y_true, scores)
               if len(np.unique(y_true)) > 1 else 0.)
    auc_pr = (average_precision_score(y_true, scores)
              if len(np.unique(y_true)) > 1 else 0.)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.

    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  AUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}  |  "
          f"F1: {f1:.4f}")
    print(f"  FPR: {fpr:.4f}  |  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  Threshold: {threshold:.4f} ({method})")

    per_attack = {}
    print(f"  Detection by attack type:")
    for at in sorted(set(attack_types)):
        if at == "none":
            continue
        m = attack_types == at
        if m.sum() > 0:
            det = y_pred[m].sum()
            tot = m.sum()
            per_attack[at] = (det, tot, 100 * det / tot)
            label = ATTACK_LABELS.get(at, at)
            print(f"    {label:30s}: {det}/{tot} = "
                  f"{100 * det / tot:.1f}%")

    return {
        "name": name, "auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1,
        "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold, "method": method,
        "per_attack": per_attack,
        "scores": scores, "y_pred": y_pred,
    }


# ══════════════════════════════════════════════════════════
#  Specialized Ensemble
# ══════════════════════════════════════════════════════════

class CompactAE(nn.Module):
    def __init__(self, input_dim, bottleneck):
        super().__init__()
        mid = max(bottleneck + 1, (input_dim + bottleneck) // 2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, mid), nn.ReLU(),
                                     nn.Linear(mid, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, mid), nn.ReLU(),
                                     nn.Linear(mid, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SpecializedDetector:
    def __init__(self, name, feature_indices, seed=42):
        self.name, self.feature_indices, self.seed = name, feature_indices, seed
        self.models = {}

    def transform(self, X):
        values = X[:, self.feature_indices]
        # S6a dominance Z-scores can be very large when local variance is tiny.
        return np.arcsinh(values) if self.name == "s6a_profile" else values

    def fit(self, X_normal, X_val_normal):
        X = self.transform(X_normal)
        X_val = self.transform(X_val_normal)
        n = X.shape[1]
        self.models["iso"] = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed).fit(X)
        self.models["svm"] = OneClassSVM(
            nu=0.05, kernel="rbf", gamma="scale").fit(X)
        trainer = AETrainer(
            CompactAE(n, max(2, n // 3)), 4e-4, 60, 256, patience=4,
            weight_decay=2e-4, min_delta=5e-4)
        trainer.fit(X, X_val)
        self.models["ae_trainer"] = trainer
        self.score_low, score_high = np.percentile(self.raw_scores(X), [1, 99], axis=0)
        self.score_span = np.maximum(score_high - self.score_low, 1e-9)

    def raw_scores(self, X):
        return np.column_stack([
            -self.models["iso"].score_samples(X),
            -self.models["svm"].score_samples(X),
            self.models["ae_trainer"].reconstruction_error(X),
        ])

    def score(self, X):
        scores = self.raw_scores(self.transform(X))
        return np.max(np.clip((scores - self.score_low) / self.score_span, 0, 1), axis=1)


class EnsembleDetector:
    def __init__(self, groups, all_cols, seed=42):
        self.detectors = {}
        for gn, cols in groups.items():
            idx = [all_cols.index(c) for c in cols if c in all_cols]
            if idx:
                self.detectors[gn] = SpecializedDetector(gn, idx, seed)

    def fit(self, X_normal, X_val_normal):
        for n, d in self.detectors.items():
            print(f"  Fitting: {n} ({len(d.feature_indices)} features)")
            d.fit(X_normal, X_val_normal)

    def score(self, X):
        r = {n: d.score(X) for n, d in self.detectors.items()}
        a = np.column_stack(list(r.values()))
        r["ensemble_max"] = np.max(a, axis=1)
        r["ensemble_mean"] = np.mean(a, axis=1)
        return r


# ══════════════════════════════════════════════════════════
#  Cascade Compromise Detector
# ══════════════════════════════════════════════════════════

class CascadeCompromiseDetector:
    def __init__(self, config, seed=42):
        self.config = config
        self.stage1_rf = RandomForestClassifier(
            n_estimators=config.cascade_rf_trees, random_state=seed,
            class_weight="balanced_subsample", n_jobs=-1,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            min_samples_split=config.rf_min_samples_split,
            max_features=config.rf_max_features,
            bootstrap=True, max_samples=config.rf_max_samples)

        # Dedicated compromise expert. Histogram boosting ranks the subtle
        # node-compromise class better than the generic stage-2 ensemble
        # while retaining explicit regularisation.
        self.stage2_hgbs = [
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.06,
                max_leaf_nodes=23,
                max_depth=6,
                min_samples_leaf=30,
                l2_regularization=3.0,
                class_weight="balanced",
                random_state=seed + offset,
            )
            for offset in (15, 16, 17, 18, 19)
        ]

    def fit(self, X_full, y, X_comp, y_comp):
        print("  Cascade Stage 1: General RF...")
        self.stage1_rf.fit(X_full, y)
        print("  Cascade Stage 2: five-seed full-feature compromise HistGB...")
        for model in self.stage2_hgbs:
            model.fit(X_full, y_comp)

    def predict_components(self, X_full, X_comp):
        p1 = self.stage1_rf.predict_proba(X_full)[:, 1]
        p2 = np.mean([
            model.predict_proba(X_full)[:, 1]
            for model in self.stage2_hgbs
        ], axis=0)
        return p1, p2

    def predict_proba(self, X_full, X_comp):
        p1, p2 = self.predict_components(X_full, X_comp)
        return np.maximum(p1, p2)

    def predict_proba_stage1(self, X):
        return self.stage1_rf.predict_proba(X)[:, 1]

    def predict_proba_stage2(self, X_comp, X_full):
        return np.mean([
            model.predict_proba(X_full)[:, 1]
            for model in self.stage2_hgbs
        ], axis=0)


# ══════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════

class ResultsVisualizer:
    def __init__(self, plots_dir):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def _save(self, fig, name):
        for fmt in ["png", "svg"]:
            path = os.path.join(self.plots_dir, f"{name}.{fmt}")
            fig.savefig(path, bbox_inches='tight', facecolor='white',
                        format=fmt)
        plt.close(fig)
        print(f"  Saved: {name}.png / .svg")

    def plot_roc_curves(self, results_list, y_test, main_models):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results_list:
            if r['name'] not in main_models:
                continue
            fpr_arr, tpr_arr, _ = roc_curve(y_test, r['scores'])
            ax.plot(fpr_arr, tpr_arr, linewidth=1.8,
                    label=f"{r['name']} (AUC={r['auc_roc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC-кривые — Diameter'); ax.legend(loc='lower right')
        self._save(fig, '01_roc_curves')

    def plot_pr_curves(self, results_list, y_test, main_models):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results_list:
            if r['name'] not in main_models:
                continue
            prec, rec, _ = precision_recall_curve(y_test, r['scores'])
            ax.plot(rec, prec, linewidth=1.8,
                    label=f"{r['name']} (AP={r['auc_pr']:.3f})")
        ax.axhline(y=y_test.mean(), color='k', ls='--', alpha=0.3,
                   label=f'Baseline ({y_test.mean():.3f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall — Diameter')
        ax.legend(loc='lower left')
        self._save(fig, '02_precision_recall')

    def plot_attack_heatmap(self, results_list, main_models):
        attacks = ['location_tracking', 'sms_data_interception',
                   'signaling_dos', 'fraud_profile', 'node_compromise']
        matrix, names = [], []
        for r in results_list:
            if r['name'] not in main_models:
                continue
            row = [r['per_attack'].get(a, (0, 1, 0))[2] for a in attacks]
            matrix.append(row)
            names.append(r['name'])
        if not matrix:
            return
        mat = np.array(matrix)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mat, cmap='Greys', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks],
                           rotation=30, ha='right')
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(attacks)):
                v = mat[i, j]
                c = 'white' if v > 60 else 'black'
                ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=c)
        plt.colorbar(im, ax=ax, label='Detection %', shrink=0.8)
        ax.set_title('Полнота обнаружения по типам атак — Diameter')
        self._save(fig, '03_attack_heatmap')

    def plot_summary_bars(self, results_list, main_models):
        names, rocs, prs, f1s = [], [], [], []
        for r in results_list:
            if r['name'] in main_models:
                names.append(r['name'])
                rocs.append(r['auc_roc'])
                prs.append(r['auc_pr'])
                f1s.append(r['f1'])
        if not names:
            return
        x = np.arange(len(names)); w = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w, rocs, w, label='AUC-ROC', color='#2f2f2f', alpha=0.9)
        ax.bar(x, prs, w, label='AUC-PR', color='#7a7a7a', alpha=0.9)
        ax.bar(x + w, f1s, w, label='F1', color='#b0b0b0', alpha=0.95)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right')
        ax.set_ylabel('Metric Value')
        ax.set_title('Сравнение моделей — Diameter')
        ax.set_ylim([0, 1.08]); ax.legend(loc='lower right')
        for i in range(len(names)):
            ax.text(i + w, f1s[i] + 0.01, f'{f1s[i]:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        self._save(fig, '04_summary_bars')

    def plot_feature_importance(self, feature_names, importances,
                                ms_features, top_n=20):
        idx = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in idx]
        top_imps = importances[idx]
        is_ms = [n in ms_features for n in top_names]
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = ['#2f2f2f' if ms else '#a6a6a6' for ms in is_ms]
        ax.barh(range(len(top_names)), top_imps, color=colors, alpha=0.85)
        ax.set_yticks(range(len(top_names))); ax.set_yticklabels(top_names)
        ax.set_xlabel('Gini Importance')
        ax.set_title(f'Top-{top_n} Feature Importance (RF) — Diameter')
        for i, v in enumerate(top_imps):
            ax.text(v + 0.001, i, f'{100*v:.1f}%', va='center', fontsize=7)
        legend_elements = [
            Patch(facecolor='#2f2f2f', alpha=0.85, label='MS features'),
            Patch(facecolor='#a6a6a6', alpha=0.85, label='Other features'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        self._save(fig, '05_feature_importance')

    def plot_ablation(self, ablation_results):
        configs = [a['config'] for a in ablation_results]
        f1s = [a['f1'] for a in ablation_results]
        comp_rates = [a['compromise_rate'] for a in ablation_results]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(configs))
        cs = ['#1f1f1f', '#4a4a4a', '#757575', '#9a9a9a', '#c0c0c0']
        ax1.bar(x, f1s, color=cs[:len(x)], alpha=0.85)
        ax1.set_xticks(x); ax1.set_xticklabels(configs, rotation=35,
                                                 ha='right')
        ax1.set_ylabel('F1'); ax1.set_title('Ablation: F1')
        ax1.set_ylim([0, 1.1])
        for i, v in enumerate(f1s):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

        ax2.bar(x, comp_rates, color=cs[:len(x)], alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(configs, rotation=35,
                                                 ha='right')
        ax2.set_ylabel('Detection %')
        ax2.set_title('Ablation: Node Compromise')
        ax2.set_ylim([0, 110])
        for i, v in enumerate(comp_rates):
            ax2.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontweight='bold')
        fig.tight_layout()
        self._save(fig, '06_ablation')

    def plot_training_curves(self, trainers_dict):
        if not trainers_dict:
            return
        n = len(trainers_dict)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (name, trainer) in zip(axes, trainers_dict.items()):
            epochs = range(1, len(trainer.train_losses) + 1)
            ax.plot(epochs, trainer.train_losses, 'k-', lw=1.5,
                    label='Train')
            if trainer.val_losses:
                ve = range(1, len(trainer.val_losses) + 1)
                ax.plot(ve, trainer.val_losses, color='#7a7a7a', ls='--',
                        lw=1.5, label='Val')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(name); ax.legend(); ax.set_yscale('log')
        fig.suptitle('Training Curves — Diameter', fontsize=13)
        fig.tight_layout()
        self._save(fig, '07_training_curves')

    def plot_confusion_matrices(self, results_list, y_test, models):
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5))
        if len(models) == 1:
            axes = [axes]
        for ax, mname in zip(axes, models):
            r = next((r for r in results_list if r['name'] == mname), None)
            if r is None:
                continue
            cm = confusion_matrix(y_test, r['y_pred'], labels=[0, 1])
            ax.imshow(cm, cmap='Greys', aspect='auto')
            for i in range(2):
                for j in range(2):
                    c = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
                    ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                            fontsize=11, fontweight='bold', color=c)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Normal', 'Anomaly'])
            ax.set_yticklabels(['Normal', 'Anomaly'])
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title(f'{mname}\nF1={r["f1"]:.3f}')
        fig.suptitle('Confusion Matrices — Diameter', y=1.02)
        fig.tight_layout()
        self._save(fig, '08_confusion_matrices')


# ══════════════════════════════════════════════════════════
#  MAIN TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


METRIC_TABLES = (
    "full_comparison_diameter.csv", "metrics_by_simulation.csv",
    "recall_by_simulation_and_attack.csv",
)


def evaluate_ablation_variants(config, pipeline, train, validation, test, attack_types,
                               stage1=None):
    """Evaluate the three ablations using the same validation FPR budget."""
    X_train, y_train = train
    X_val, y_val = validation
    X_test, y_test = test
    template = CascadeCompromiseDetector(config, config.seed).stage1_rf
    if stage1 is None:
        stage1 = template.fit(X_train, y_train)

    variants = []
    for name, block in (("RF (no MS)", MS_FEATURES),
                        ("RF (no temporal)", DataPipeline.TEMPORAL_FEATURES)):
        features = [i for i, name in enumerate(pipeline.feature_cols) if name not in block]
        model = clone(template).set_params(n_estimators=config.rf_trees)
        model.fit(X_train[:, features], y_train)
        threshold, _ = select_supervised_threshold(
            model.predict_proba(X_val[:, features])[:, 1], y_val, config.supervised_max_fpr)
        variants.append(evaluate_model(
            name, model.predict_proba(X_test[:, features])[:, 1], y_test,
            attack_types, threshold=threshold))
    threshold, _ = select_supervised_threshold(
        stage1.predict_proba(X_val)[:, 1], y_val, config.supervised_max_fpr)
    variants.append(evaluate_model(
        "Cascade-S1", stage1.predict_proba(X_test)[:, 1], y_test,
        attack_types, threshold=threshold))
    return variants


def collect_metric_tables(results_list, test_df, y_test, attack_types_test, sequences=None):
    save_cols = ['name', 'auc_roc', 'auc_pr', 'f1', 'fpr',
                 'tp', 'fp', 'fn', 'tn']
    res_df = pd.DataFrame([{k: r[k] for k in save_cols}
                           for r in results_list])
    test_runs = test_df["_test_simulation"].to_numpy()
    metrics_rows, attack_rows = [], []
    for result in results_list:
        if "y_true" in result:
            runs, labels, attacks = sequences
        else:
            runs, labels, attacks = test_runs, y_test, attack_types_test
        for run in pd.unique(runs):
            mask = runs == run
            scores, pred = result["scores"][mask], result["y_pred"][mask]
            y = labels[mask]
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            metrics_rows.append({
                "simulation": run, "name": result["name"], "windows": len(y),
                "auc_roc": roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.0,
                "auc_pr": average_precision_score(y, scores) if len(np.unique(y)) > 1 else 0.0,
                "f1": f1_score(y, pred, zero_division=0),
                "fpr": fp / (fp + tn) if fp + tn else 0.0,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": result["threshold"],
            })
            for attack in sorted(set(attacks[mask]) - {"none"}):
                attack_mask = attacks[mask] == attack
                hits = int(pred[attack_mask].sum())
                count = int(attack_mask.sum())
                attack_rows.append({"simulation": run, "name": result["name"],
                                    "attack_type": attack, "detected": hits,
                                    "windows": count, "recall": hits / count})
    return res_df, pd.DataFrame(metrics_rows), pd.DataFrame(attack_rows)


def main(config=None):
    config = config or TrainingConfig()
    set_global_seed(config.seed)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)

    pipeline = DataPipeline(config)
    train_df, val_df, stack_df, test_df = pipeline.load_and_prepare()
    if config.curve_train_fraction is not None:
        fraction = config.curve_train_fraction
        if not 0 < fraction < 1:
            raise ValueError("Curve training fraction must be between zero and one")
        groups = train_df.groupby(pipeline.ATTACK_COL, sort=True)
        sampled = [
            group.sample(frac=1, random_state=config.seed).iloc[:max(1, round(len(group) * fraction))]
            for _, group in groups
        ]
        train_df = pd.concat(sampled).sort_index()
        print(f"Curve training rows: {len(train_df)}; fraction={fraction}")

    X_train, y_train = pipeline.get_features_labels(train_df, fit=True)
    X_val, y_val = pipeline.get_features_labels(val_df)
    X_stack, y_stack = pipeline.get_features_labels(stack_df)
    X_test, y_test = pipeline.get_features_labels(test_df)

    X_train_comp, y_train_comp = pipeline.get_compromise_features(
        train_df, fit=True)
    X_val_comp, y_val_comp = pipeline.get_compromise_features(val_df)
    X_test_comp, y_test_comp = pipeline.get_compromise_features(test_df)

    X_train_normal, _ = pipeline.get_normal_data(train_df, fit=False)

    # Compress heavy-tailed standardized telemetry for squared-error models.
    # The signed monotone transform uses the training-fitted scaler and has
    # no parameters to re-estimate on validation, stacking, or test.
    X_train_neural, X_val_neural, X_stack_neural, X_test_neural = (
        np.arcsinh(X) for X in (X_train_normal, X_val, X_stack, X_test)
    )

    attack_types_test = test_df["attack_type"].values
    n_feat = X_train.shape[1]

    results_list = []
    trainers_dict = {}

    print("\n" + "=" * 60)
    print("  Training Models — Diameter v1")
    print("=" * 60)

    # ── 1. Robust Isolation Forest ──
    print("\n[1] Robust Isolation Forest...")
    iso_low, iso_high = fit_iso_robust_bounds(
        X_train_normal,
        config.iso_forest_clip_lower_q,
        config.iso_forest_clip_upper_q,
    )
    X_train_iso = apply_iso_robust_bounds(X_train, iso_low, iso_high)
    X_train_normal_iso = apply_iso_robust_bounds(X_train_normal, iso_low, iso_high)
    X_val_iso = apply_iso_robust_bounds(X_val, iso_low, iso_high)
    X_test_iso = apply_iso_robust_bounds(X_test, iso_low, iso_high)

    iso = IsolationForest(
        n_estimators=config.iso_forest_trees,
        contamination="auto",
        max_features=config.iso_forest_max_features,
        bootstrap=config.iso_forest_bootstrap,
        random_state=config.seed,
        n_jobs=-1,
    )
    iso.fit(X_train_normal_iso)
    iso_sv = -iso.score_samples(X_val_iso)
    iso_st = -iso.score_samples(X_test_iso)
    # Calibrate only on benign validation observations.  This avoids selecting
    # the threshold with attack labels and makes the intended FPR explicit.
    th_iso = float(np.percentile(
        iso_sv[y_val == 0], config.threshold_percentile
    ))
    iso_result = evaluate_model(
        "Isolation Forest", iso_st, y_test,
        attack_types_test, threshold=th_iso
    )
    iso_result["robustness_method"] = (
        "benign-train-only 0.1/99.9% winsorization + "
        "full-feature IsolationForest + benign-validation percentile threshold"
    )
    results_list.append(iso_result)

    # ── 2. One-Class SVM ──
    print("\n[2] One-Class SVM...")
    svm = OneClassSVM(nu=config.ocsvm_nu, kernel=config.ocsvm_kernel,
                      gamma="scale")
    svm.fit(X_train_normal)
    svm_sv = -svm.score_samples(X_val)
    svm_st = -svm.score_samples(X_test)
    th_svm = float(np.percentile(
        svm_sv[y_val == 0], config.threshold_percentile))
    results_list.append(evaluate_model("One-Class SVM", svm_st, y_test,
                                       attack_types_test, threshold=th_svm))

    # ── 3. Autoencoder ──
    print("\n[3] Autoencoder...")
    ae = Autoencoder(
        n_feat, config.ae_hidden_dims, config.ae_latent_dim,
        dropout=config.ae_dropout)
    ae_trainer = AETrainer(
        ae, config.ae_lr, config.ae_epochs, config.ae_batch,
        patience=config.ae_patience,
        weight_decay=config.ae_weight_decay,
        min_delta=config.ae_min_delta)
    X_val_normal = X_val[y_val == 0]
    ae_trainer.fit(X_train_neural, X_val_neural[y_val == 0])
    trainers_dict['Autoencoder'] = ae_trainer
    ae_sv = ae_trainer.reconstruction_error(X_val_neural)
    ae_st = ae_trainer.reconstruction_error(X_test_neural)
    th_ae = float(np.percentile(
        ae_sv[y_val == 0], config.threshold_percentile))
    results_list.append(evaluate_model("Autoencoder", ae_st, y_test,
                                       attack_types_test, threshold=th_ae))

    # ── 4. VAE ──
    print("\n[4] VAE...")
    vae = VAE(
        n_feat, config.vae_hidden_dims, config.vae_latent_dim,
        dropout=config.vae_dropout)
    vae_trainer = VAETrainer(
        vae, config.vae_lr, config.vae_epochs, config.vae_batch,
        kl_weight=config.vae_kl_weight,
        patience=config.vae_patience,
        weight_decay=config.vae_weight_decay,
        min_delta=config.vae_min_delta)
    vae_trainer.fit(X_train_neural, X_val_neural[y_val == 0])
    trainers_dict['VAE'] = vae_trainer
    vae_sv = vae_trainer.anomaly_score(X_val_neural)
    vae_st = vae_trainer.anomaly_score(X_test_neural)
    th_vae = float(np.percentile(
        vae_sv[y_val == 0], config.threshold_percentile))
    results_list.append(evaluate_model("VAE", vae_st, y_test,
                                       attack_types_test, threshold=th_vae))

    seq_test_runs = seq_y_test = seq_at_test = None
    if config.curve_train_fraction is None:
        # ── 5. LSTM-AE ──
        print("\n[5] LSTM-Autoencoder...")
        seq_train, _, _, benign_train = pipeline.get_sequential_data(
            train_df, config.lstm_seq_len)
        seq_train = np.arcsinh(seq_train[benign_train])
        seq_val, _, _, benign_val_windows = pipeline.get_sequential_data(
            val_df, config.lstm_seq_len)
        seq_val = np.arcsinh(seq_val)
        seq_val_normal = seq_val[benign_val_windows]
        seq_test, seq_y_test, seq_at_test, _, seq_test_runs = pipeline.get_sequential_data(
            test_df, config.lstm_seq_len, include_simulation=True)
        seq_test = np.arcsinh(seq_test)
        lstm_done = False
        if len(seq_train) > 100 and len(seq_val_normal) and len(seq_test):
            lstm = LSTMAutoencoder(n_feat, config.lstm_hidden,
                                   config.lstm_layers,
                                   config.lstm_seq_len,
                                   dropout=config.lstm_dropout)
            lstm_trainer = AETrainer(
                lstm, config.lstm_lr, config.lstm_epochs, config.lstm_batch,
                patience=config.lstm_patience,
                weight_decay=config.lstm_weight_decay,
                min_delta=config.lstm_min_delta)
            lstm_trainer.fit(seq_train, seq_val_normal)
            trainers_dict['LSTM-AE'] = lstm_trainer

            lstm_st = lstm_trainer.reconstruction_error(seq_test)
            lstm_sv = lstm_trainer.reconstruction_error(seq_val)
            th_lstm = float(np.percentile(
                lstm_sv[benign_val_windows], config.threshold_percentile))
            results_list.append(evaluate_model(
                "LSTM-Autoencoder", lstm_st, seq_y_test,
                seq_at_test, threshold=th_lstm))
            results_list[-1]["y_true"] = seq_y_test
            lstm_done = True
        else:
            print("  Skipped: not enough sequences.")

        # ── 6. Temporal-Attention-AE ──
        print("\n[6] Temporal-Attention-AE...")
        if lstm_done and len(seq_train) > 100:
            tat = TemporalAttentionAE(
                n_feat, config.tat_hidden, config.tat_heads, config.tat_layers,
                config.lstm_seq_len, config.tat_dropout)
            tat_trainer = AETrainer(
                tat, config.tat_lr, config.tat_epochs, config.tat_batch,
                patience=config.tat_patience,
                weight_decay=config.tat_weight_decay,
                min_delta=config.tat_min_delta)
            tat_trainer.fit(seq_train, seq_val_normal)
            trainers_dict['TA-AE'] = tat_trainer
            tat_st = tat_trainer.reconstruction_error(seq_test)
            tat_sv = tat_trainer.reconstruction_error(seq_val)
            th_tat = float(np.percentile(
                tat_sv[benign_val_windows], config.threshold_percentile))
            results_list.append(evaluate_model(
                "Temporal-Attention-AE", tat_st, seq_y_test,
                seq_at_test, threshold=th_tat))
            results_list[-1]["y_true"] = seq_y_test
            del seq_train, seq_val, seq_val_normal, seq_test

    # ── 7. Random Forest ──
    print("\n[7] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=config.rf_trees,
        random_state=config.seed,
        class_weight="balanced_subsample", n_jobs=-1,
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        min_samples_split=config.rf_min_samples_split,
        max_features=config.rf_max_features,
        bootstrap=True, max_samples=config.rf_max_samples)
    rf.fit(X_train, y_train)
    rf_pv = rf.predict_proba(X_val)[:, 1]
    rf_pt = rf.predict_proba(X_test)[:, 1]
    th_rf, th_rf_method = select_supervised_threshold(
        rf_pv, y_val, config.supervised_max_fpr)
    print(f"  RF validation threshold: {th_rf:.6f} ({th_rf_method})")
    results_list.append(evaluate_model("Random Forest", rf_pt, y_test,
                                       attack_types_test, threshold=th_rf))

    importances = rf.feature_importances_
    print("\n  Feature importance (top 20):")
    feat_imp = sorted(zip(pipeline.feature_cols, importances),
                      key=lambda x: x[1], reverse=True)
    for fn, imp in feat_imp[:20]:
        print(f"    {fn:45s} {imp:.4f} ({100*imp:.1f}%)")

    ms_imp = sum(imp for fn, imp in feat_imp
                 if fn in MS_FEATURES)
    print(f"\n  MS features total importance: {ms_imp:.4f} "
          f"({100 * ms_imp:.1f}%)")

    # ── 8. Gradient Boosting ──
    print("\n[8] Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=config.gb_estimators,
        learning_rate=config.gb_lr,
        max_depth=config.gb_max_depth,
        min_samples_leaf=config.gb_min_samples_leaf,
        subsample=config.gb_subsample,
        random_state=config.seed)
    gb.fit(X_train, y_train)
    gb_pv = gb.predict_proba(X_val)[:, 1]
    gb_pt = gb.predict_proba(X_test)[:, 1]
    th_gb, th_gb_method = select_supervised_threshold(
        gb_pv, y_val, config.supervised_max_fpr)
    print(f"  GB validation threshold: {th_gb:.6f} ({th_gb_method})")
    results_list.append(evaluate_model("Gradient Boosting", gb_pt, y_test,
                                       attack_types_test, threshold=th_gb))

    # ── 9. Cascade ──
    print("\n[9] Cascade Compromise Detector...")
    cascade = CascadeCompromiseDetector(config, config.seed)
    cascade.fit(X_train, y_train, X_train_comp, y_train_comp)
    casc1_pv, casc2_pv = cascade.predict_components(X_val, X_val_comp)
    casc1_pt, casc2_pt = cascade.predict_components(X_test, X_test_comp)
    th_casc1, th_casc2, casc_pv, compromise_threshold_info = (
        select_compromise_aware_thresholds(
            casc1_pv, casc2_pv, y_val, y_val_comp,
            general_fpr_budget=config.compromise_general_fpr_budget,
            total_fpr_budget=config.compromise_total_fpr_budget,
            target_recall=config.compromise_target_recall,
        )
    )
    casc_pt = combine_cascade_scores(
        casc1_pt, casc2_pt, th_casc1, th_casc2)
    print(
        f"  Cascade dual thresholds: general={th_casc1:.6f}, "
        f"compromise={th_casc2:.6f}; "
        f"validation compromise recall="
        f"{compromise_threshold_info['validation_compromise_recall']:.3f}, "
        f"validation FPR={compromise_threshold_info['validation_fpr']:.4f}"
    )
    casc_result = evaluate_model(
        "Cascade", casc_pt, y_test, attack_types_test, threshold=1.0)
    casc_result["compromise_threshold_info"] = compromise_threshold_info
    results_list.append(casc_result)

    # An alternative second stage scores persistence of repeated fresh
    # report profiles. Stage 1 and the validation budget match the baseline.
    stack_general = cascade.predict_proba_stage1(X_stack)
    if all(RECURRENCE_COLUMNS <= set(frame.columns)
           for frame in (val_df, stack_df, test_df)):
        recurrence_domains = [
            ("validation", casc1_pv, report_recurrence(val_df), y_val, y_val_comp),
            ("stack", stack_general, report_recurrence(stack_df), y_stack,
             stack_df[pipeline.ATTACK_COL].eq("node_compromise").to_numpy()),
        ]
        rec_validation = calibrate_cascade_across_domains(
            recurrence_domains, config, config.recurrence_calibration_budget)
        recurrence_t = combine_cascade_scores(
            casc1_pt, report_recurrence(test_df), *rec_validation["thresholds"])
        recurrence_result = evaluate_model(
            RECURRENCE_MODEL, recurrence_t, y_test, attack_types_test, threshold=1.0)
        recurrence_result["compromise_threshold_info"] = rec_validation
        results_list.append(recurrence_result)

    # ── 10. Specialized Ensemble ──
    print("\n[10] Specialized Ensemble...")
    ensemble = EnsembleDetector(FEATURE_GROUPS, pipeline.feature_cols,
                                config.seed)
    ensemble.fit(X_train_normal, X_val_normal)
    ens_sv = ensemble.score(X_val)
    ens_st = ensemble.score(X_test)
    ens_results = []
    for en in sorted(ens_st.keys()):
        sv, st = ens_sv[en], ens_st[en]
        th = float(np.percentile(
            sv[y_val == 0], config.threshold_percentile))
        r = evaluate_model(f"Ens:{en}", st, y_test, attack_types_test,
                           threshold=th)
        results_list.append(r)
        ens_results.append(r)

    # ── 11. Meta-Ensemble ──
    print("\n[11] Meta-Ensemble...")

    # Use raw cascade branch probabilities, independent of validation labels.
    stack_compromise = cascade.predict_proba_stage2(None, X_stack)
    meta_stack = np.column_stack([
        -iso.score_samples(apply_iso_robust_bounds(X_stack, iso_low, iso_high)),
        -svm.score_samples(X_stack),
        ae_trainer.reconstruction_error(X_stack_neural), vae_trainer.anomaly_score(X_stack_neural),
        stack_general, stack_compromise,
        rf.predict_proba(X_stack)[:, 1], gb.predict_proba(X_stack)[:, 1],
    ])
    meta_val = np.column_stack([
        iso_sv, svm_sv, ae_sv, vae_sv, casc1_pv, casc2_pv, rf_pv, gb_pv,
    ])
    meta_test = np.column_stack([
        iso_st, svm_st, ae_st, vae_st, casc1_pt, casc2_pt, rf_pt, gb_pt,
    ])
    meta_gb = GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.03, max_depth=2,
        min_samples_leaf=30, subsample=0.80,
        random_state=config.seed)
    meta_gb.fit(meta_stack, y_stack)
    meta_fit_scores = meta_gb.predict_proba(meta_stack)[:, 1]
    meta_val_scores = meta_gb.predict_proba(meta_val)[:, 1]
    meta_pt = meta_gb.predict_proba(meta_test)[:, 1]
    _, th_meta, _, meta_threshold_info = select_compromise_aware_thresholds(
        meta_val_scores, meta_val_scores, y_val, y_val_comp,
        general_fpr_budget=config.compromise_general_fpr_budget,
        total_fpr_budget=config.compromise_total_fpr_budget,
        target_recall=config.compromise_target_recall,
    )
    # The two inputs are identical, so their OR uses the lower threshold.
    th_meta = min(th_meta, meta_threshold_info["general_threshold"])
    meta_threshold_info["decision_threshold"] = th_meta
    best_meta_recall = meta_threshold_info["validation_compromise_recall"]
    print(f"  Meta threshold: {th_meta:.6f}; validation compromise recall="
          f"{best_meta_recall:.3f}")

    meta_result = evaluate_model(
        "Meta-Ensemble", meta_pt, y_test, attack_types_test,
        threshold=th_meta)
    meta_result["validation_compromise_recall"] = best_meta_recall
    results_list.append(meta_result)

    # ══════════════════════════════════════════════════
    #  Ablation Studies
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Ablation Studies")
    print("=" * 60)

    ablation = []
    comp_mask = attack_types_test == "node_compromise"
    comp_total = comp_mask.sum()

    # Full
    comp_full = (rf_pt[comp_mask] >= th_rf).sum()
    rf_r = next(r for r in results_list if r['name'] == 'Random Forest')
    ablation.append({
        'config': f'Full\n({len(pipeline.feature_cols)} feat)',
        'f1': rf_r['f1'],
        'compromise_rate': 100 * comp_full / max(1, comp_total),
    })

    temporal = DataPipeline.TEMPORAL_FEATURES
    variants = evaluate_ablation_variants(
        config, pipeline, (X_train, y_train), (X_val, y_val), (X_test, y_test),
        attack_types_test, stage1=cascade.stage1_rf)
    for result, label in zip(variants, (
        f"No MS\n({sum(c not in MS_FEATURES for c in pipeline.feature_cols)} feat)",
        f"No temporal\n({sum(c not in temporal for c in pipeline.feature_cols)} feat)",
        "Cascade\nStage-1 only",
    )):
        ablation.append({
            "config": label, "f1": result["f1"],
            "compromise_rate": result["per_attack"].get("node_compromise", (0, 0, 0))[2],
        })
    results_list.extend(variants)

    # Cascade full
    casc_r = next(r for r in results_list if r['name'] == 'Cascade')
    comp_casc = (casc_pt[comp_mask] >= 1.0).sum()
    ablation.append({
        'config': 'Cascade\nBoth levels',
        'f1': casc_r['f1'],
        'compromise_rate': 100 * comp_casc / max(1, comp_total),
    })

    # ══════════════════════════════════════════════════
    #  Generalization / overfitting diagnostics
    # ══════════════════════════════════════════════════
    overfit_diag = {
        "method": {
            "neural_early_stopping": "best benign-validation loss, restored weights",
            "unsupervised_thresholds": "95th percentile of benign validation only",
            "split_protocol": pipeline.split_protocol,
            "sequences": "consecutive intervals; wholly benign fit and validation windows",
            "neural_scaling": "signed asinh of training-standardized features",
            "specialized_scaling": "frozen quantiles from benign training scores",
            "meta_stacking": "fit on separate stacking predictions; threshold on validation",
            "supervised_thresholds": (
                f"validation-only max-F1 subject to FPR <= "
                f"{config.supervised_max_fpr:.3f}"
            ),
            "test_role": "untouched until final evaluation",
        },
        "neural": {},
        "supervised": {},
    }

    diagnostic_trainers = {
        **trainers_dict,
        **{f"Ens:{name}": detector.models["ae_trainer"]
           for name, detector in ensemble.detectors.items()},
    }
    for name, trainer in diagnostic_trainers.items():
        vals = np.asarray(trainer.val_losses, dtype=float)
        trs = np.asarray(
            getattr(trainer, "train_eval_losses", trainer.train_losses),
            dtype=float)
        if len(vals) and len(trs):
            best_idx = trainer.best_epoch - 1
            overfit_diag["neural"][name] = {
                "epochs_run": int(len(trs)),
                "best_epoch": best_idx + 1,
                "train_loss_at_best": float(trs[best_idx]),
                "best_val_loss": float(vals[best_idx]),
                "absolute_gap_at_best": float(vals[best_idx] - trs[best_idx]),
                "final_train_loss": float(trs[-1]),
                "final_val_loss": float(vals[-1]),
            }

    def ap_across_splits(p_train, p_val, p_test):
        train_ap = float(average_precision_score(y_train, p_train))
        val_ap = float(average_precision_score(y_val, p_val))
        test_ap = float(average_precision_score(y_test, p_test))
        return {
            "train_auc_pr": train_ap,
            "val_auc_pr": val_ap,
            "test_auc_pr": test_ap,
            "train_minus_val_auc_pr": train_ap - val_ap,
            "val_minus_test_auc_pr": val_ap - test_ap,
        }

    overfit_diag["supervised"]["Random Forest"] = ap_across_splits(
        rf.predict_proba(X_train)[:, 1],
        rf.predict_proba(X_val)[:, 1],
        rf_pt,
    )
    overfit_diag["supervised"]["Gradient Boosting"] = ap_across_splits(
        gb.predict_proba(X_train)[:, 1],
        gb.predict_proba(X_val)[:, 1],
        gb_pt,
    )
    casc1_train, casc2_train = cascade.predict_components(
        X_train, X_train_comp)
    casc_train = combine_cascade_scores(
        casc1_train, casc2_train, th_casc1, th_casc2)
    overfit_diag["supervised"]["Cascade"] = ap_across_splits(
        casc_train,
        casc_pv,
        casc_pt,
    )
    meta_fit_ap = float(average_precision_score(y_stack, meta_fit_scores))
    meta_val_ap = float(average_precision_score(y_val, meta_val_scores))
    meta_test_ap = float(average_precision_score(y_test, meta_pt))
    overfit_diag["supervised"]["Meta-Ensemble"] = {
        "fit_auc_pr_on_stack": meta_fit_ap,
        "val_auc_pr": meta_val_ap,
        "test_auc_pr": meta_test_ap,
        "fit_minus_val_auc_pr": meta_fit_ap - meta_val_ap,
        "val_minus_test_auc_pr": meta_val_ap - meta_test_ap,
    }

    if config.compare_tree_reference:
        print("\n  Comparing previous tree capacities on the same partitions")
        forest_params = dict(max_depth=16, min_samples_leaf=3,
                             min_samples_split=6, max_samples=0.85)
        ref_rf = clone(rf).set_params(**forest_params).fit(X_train, y_train)
        ref_gb = clone(gb).set_params(
            max_depth=4, min_samples_leaf=12, subsample=0.90).fit(X_train, y_train)
        ref_cascade = CascadeCompromiseDetector(config, config.seed)
        ref_cascade.stage1_rf.set_params(**forest_params)
        for expert in ref_cascade.stage2_hgbs:
            expert.set_params(max_leaf_nodes=31, max_depth=8,
                              min_samples_leaf=20, l2_regularization=1.0)
        ref_cascade.fit(X_train, y_train, X_train_comp, y_train_comp)
        ref_v1, ref_v2 = ref_cascade.predict_components(X_val, X_val_comp)
        ref_t1, ref_t2, ref_v, _ = select_compromise_aware_thresholds(
            ref_v1, ref_v2, y_val, y_val_comp,
            general_fpr_budget=config.compromise_general_fpr_budget,
            total_fpr_budget=config.compromise_total_fpr_budget,
            target_recall=config.compromise_target_recall,
        )
        ref_train = combine_cascade_scores(
            *ref_cascade.predict_components(X_train, X_train_comp), ref_t1, ref_t2)
        ref_test = combine_cascade_scores(
            *ref_cascade.predict_components(X_test, X_test_comp), ref_t1, ref_t2)
        overfit_diag["supervised_reference"] = {
            "Random Forest": ap_across_splits(*[
                ref_rf.predict_proba(X)[:, 1] for X in (X_train, X_val, X_test)]),
            "Gradient Boosting": ap_across_splits(*[
                ref_gb.predict_proba(X)[:, 1] for X in (X_train, X_val, X_test)]),
            "Cascade": {
                **ap_across_splits(ref_train, ref_v, ref_test),
                "test_compromise_recall": float(np.mean(ref_test[y_test_comp == 1] >= 1)),
                "test_benign_fpr": float(np.mean(ref_test[y_test == 0] >= 1)),
            },
        }

    # Explicit node-compromise recall report for reproducible acceptance.
    comp_test_mask = attack_types_test == "node_compromise"
    cascade_test_pred = casc_pt >= 1.0
    meta_test_pred = meta_pt >= th_meta
    compromise_diag = {
        "validation": compromise_threshold_info,
        "meta_validation": meta_threshold_info,
        "test": {
            "cascade_compromise_recall": float(
                np.mean(cascade_test_pred[comp_test_mask])
                if comp_test_mask.any() else 0.0),
            "cascade_benign_fpr": float(
                np.mean(cascade_test_pred[y_test == 0])
                if np.any(y_test == 0) else 0.0),
            "meta_compromise_recall": float(
                np.mean(meta_test_pred[comp_test_mask])
                if comp_test_mask.any() else 0.0),
            "meta_benign_fpr": float(
                np.mean(meta_test_pred[y_test == 0])
                if np.any(y_test == 0) else 0.0),
        },
        "targets": {
            "compromise_recall": config.compromise_target_recall,
            "benign_fpr_max": config.compromise_total_fpr_budget,
        },
    }
    recurrence = next((r for r in results_list if r["name"] == RECURRENCE_MODEL), None)
    if recurrence is not None:
        hits, total, _ = recurrence["per_attack"].get("node_compromise", (0, 0, 0))
        compromise_diag["recurrence_validation"] = recurrence["compromise_threshold_info"]
        compromise_diag["test"].update({
            "recurrence_compromise_recall": float(hits / total) if total else None,
            "recurrence_benign_fpr": float(recurrence["fpr"]),
        })
    with open(os.path.join(
            config.results_dir, "compromise_recall_diagnostics.json"),
            "w", encoding="utf-8") as f:
        json.dump(compromise_diag, f, indent=2, ensure_ascii=False)
    print("\n  Compromise recall diagnostics:")
    print(json.dumps(compromise_diag, indent=2, ensure_ascii=False))

    diag_path = os.path.join(config.results_dir, "overfitting_diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(overfit_diag, f, indent=2, ensure_ascii=False)
    print("\n  Overfitting diagnostics:")
    print(json.dumps(overfit_diag, indent=2, ensure_ascii=False))

    # ══════════════════════════════════════════════════
    #  Save Results
    # ══════════════════════════════════════════════════
    metric_tables = collect_metric_tables(
        results_list, test_df, y_test, attack_types_test,
        (seq_test_runs, seq_y_test, seq_at_test))
    for name, table in zip(METRIC_TABLES, metric_tables):
        table.to_csv(Path(config.results_dir) / name, index=False)
    if config.curve_train_fraction is not None:
        metadata = {"fraction": config.curve_train_fraction,
                    "train_rows": len(train_df), "test_rows": len(test_df)}
        (Path(config.results_dir) / "curve_point.json").write_text(
            json.dumps(metadata), encoding="utf-8")
        return
    recurrence = next((r for r in results_list if r["name"] == RECURRENCE_MODEL), None)
    if recurrence is not None:
        print(f"\n  {RECURRENCE_MODEL} thresholds: "
              f"{recurrence['compromise_threshold_info']['thresholds']}")

    with open(os.path.join(config.results_dir,
                           "feature_columns_diameter.json"), "w") as f:
        json.dump({
            "observable_features": pipeline.feature_cols,
            "compromise_features": pipeline.compromise_feature_cols,
            "ms_features": [c for c in MS_FEATURES
                            if c in pipeline.feature_cols],
            "temporal_features": [c for c in temporal
                                  if c in pipeline.feature_cols],
            "feature_groups": {k: [c for c in v
                                   if c in pipeline.feature_cols]
                               for k, v in FEATURE_GROUPS.items()},
            "isolation_forest_robustness": {
                "trees": config.iso_forest_trees,
                "max_features": config.iso_forest_max_features,
                "bootstrap": config.iso_forest_bootstrap,
                "clip_lower_q": config.iso_forest_clip_lower_q,
                "clip_upper_q": config.iso_forest_clip_upper_q,
                "threshold_percentile": config.threshold_percentile,
                "threshold_calibration": "benign validation only",
            },
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY — Diameter v1")
    print("=" * 70)
    print(f"{'Model':<35s} {'AUC-ROC':>8s} {'AUC-PR':>8s} "
          f"{'F1':>8s} {'FPR':>8s}")
    print("-" * 67)
    for r in results_list:
        print(f"{r['name']:<35s} {r['auc_roc']:8.4f} "
              f"{r['auc_pr']:8.4f} {r['f1']:8.4f} {r['fpr']:8.4f}")

    # ══════════════════════════════════════════════════
    #  Save raw artifacts for the dedicated plot generator
    # ══════════════════════════════════════════════════
    # Canonical publication figures are generated only by
    # diameter/diameter_generate_plots.py.  Keeping plotting outside the
    # training pipeline makes the figures reproducible without retraining.
    import pickle

    # История обучения нейросетевых моделей
    trainers_history = {}
    for name, trainer in diagnostic_trainers.items():
        trainers_history[name] = {
            "train_losses": list(trainer.train_losses),
            "train_eval_losses": list(
                getattr(trainer, "train_eval_losses", trainer.train_losses)),
            "val_losses": list(trainer.val_losses),
        }

    # Для последовательных моделей y_test отличается (seq_y_test).
    # Приводим scores к длине y_test, где это возможно; иначе скрипт
    # визуализации сам отфильтрует такие модели по длине.
    artifacts = {
        "results_list": results_list,
        "ens_results": ens_results,
        "ablation_results": ablation,
        "y_test": y_test,
        "attack_types_test": attack_types_test,
        "importances": importances,
        "feature_cols": pipeline.feature_cols,
        "cascade_s1": cascade.predict_proba_stage1(X_test),
        "cascade_s2": cascade.predict_proba_stage2(
            X_test_comp, X_test),
        "cascade_comb": casc_pt,
        "trainers_history": trainers_history,
    }

    artifacts_path = os.path.join(config.results_dir, "plot_artifacts.pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)

    # Portable plotting payload.  JSON avoids pandas/numpy pickle ABI
    # compatibility problems when figures are regenerated in another
    # environment or after a dependency upgrade.
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                return _json_safe(value.tolist())
            except Exception:
                pass
        return value

    artifacts_json_path = os.path.join(
        config.results_dir, "plot_artifacts.json")
    with open(artifacts_json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(artifacts), f, ensure_ascii=False)
    print(f"\n  Plot artifacts saved to {artifacts_path}")
    print(f"  Portable plot artifacts saved to {artifacts_json_path}")


    print(f"\n  All results saved to {config.results_dir}/")
    print("  Plot generation delegated to diameter/diameter_generate_plots.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=TrainingConfig.data_path)
    parser.add_argument("--validation-data")
    parser.add_argument("--stack-data")
    parser.add_argument("--test-data", nargs="+")
    parser.add_argument("--curve-train-fraction", type=float)
    parser.add_argument("--compare-tree-reference", action="store_true")
    parser.add_argument("--results-dir", default=TrainingConfig.results_dir)
    args = parser.parse_args()
    main(TrainingConfig(
        data_path=args.train_data, validation_path=args.validation_data,
        stack_path=args.stack_data, test_path=args.test_data[0] if args.test_data else None,
        additional_test_paths=tuple(args.test_data[1:]) if args.test_data else (),
        curve_train_fraction=args.curve_train_fraction,
        results_dir=args.results_dir, compare_tree_reference=args.compare_tree_reference,
        plots_dir=str(Path(args.results_dir) / "plots"),
    ))
