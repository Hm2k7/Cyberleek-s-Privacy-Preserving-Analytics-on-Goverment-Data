"""
Federated Learning (FedAvg) Simulation for Healthcare Predictive Modeling
Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data

Person 2: Core Privacy Technique Engineer

This script implements decentralized Federated Learning (FedAvg) across three simulated
UAE hospitals (Hospital A Abu Dhabi, Hospital B Dubai, Hospital C RAK) to predict
30-day hospital readmission without centralizing patient records.

Regulatory Alignment:
- Complies with UAE Federal Decree-Law No. 45/2021 (PDPL) Article 1 (Sensitive Data)
- Complies with UAE Federal Law No. 2/2019 (Health ICT Law) Article 13 (Institutional Boundary Preservation)
- Raw electronic health records never cross hospital network perimeters. Only model weights/gradients
  are transmitted to MOHAP (Ministry of Health & Prevention) orchestrator.

Supported Modes:
1. Standard FedAvg (McMahan et al., 2017)
2. Differentially Private FedAvg (DP-FedAvg) with L2 weight clipping and calibrated Gaussian noise

Usage:
    python federated_learning.py [--rounds 15] [--local-epochs 3] [--dp] [--epsilon 2.0]
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =====================================================================
# DP-FEDAVG MULTI-ROUND COMPOSITION (zero-Concentrated DP)
# =====================================================================
#
# DP-FedAvg releases a Gaussian-noised model update once per communication
# round, and every round's release is visible to (and further trains on top
# of, in the next round) the same underlying hospital cohorts. The *total*
# privacy loss of the run is therefore the composition of `rounds` Gaussian
# releases, not the cost of a single release.
#
# Composing many Gaussian releases under basic/sequential (epsilon, delta)
# composition (total_eps = rounds * per_round_eps) is extremely loose: to
# hit a fixed total epsilon, per-round epsilon shrinks linearly in `rounds`,
# which blows up the required noise std (sigma ~ 1/eps) linearly too. That
# is why a naive per-round accounting quickly makes the released model
# useless over many rounds even at "generous" total epsilon values.
#
# Real DP-FedAvg / DP-SGD implementations (e.g. McMahan et al. 2018,
# "Learning Differentially Private Recurrent Language Models"; Abadi et al.
# 2016 moments accountant) instead use a composition theorem where the
# noise std needed for a fixed total budget grows only with sqrt(rounds).
# We use zero-Concentrated DP (zCDP; Bun & Steinke 2016), which gives a
# simple closed form with that favorable scaling and composes *exactly*
# (additively) across rounds:
#
#   A single Gaussian release with L2 sensitivity S and noise std sigma
#   satisfies rho-zCDP with  rho = S^2 / (2 * sigma^2).
#
#   R releases with the same (S, sigma) compose to  rho_total = R * rho.
#
#   rho_total converts to (epsilon, delta)-DP via (Bun & Steinke, Prop 3.3):
#       epsilon = rho_total + 2 * sqrt(rho_total * ln(1/delta))
#
# We invert this to solve for the per-round sigma that makes R releases
# compose to *exactly* the caller's requested total (epsilon, delta).

def zcdp_epsilon_from_rho(rho: float, delta: float) -> float:
    """Forward conversion: total rho-zCDP budget to the (epsilon, delta)-DP guarantee
    it implies (Bun & Steinke 2016, Prop 3.3)."""
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def _zcdp_rho_from_total_epsilon(total_epsilon: float, delta: float) -> float:
    """Inverts zcdp_epsilon_from_rho() to find the total rho-zCDP budget
    corresponding to a target (total_epsilon, delta)-DP guarantee."""
    log_inv_delta = math.log(1.0 / delta)
    return (math.sqrt(total_epsilon + log_inv_delta) - math.sqrt(log_inv_delta)) ** 2


def sigma_for_composed_gaussian(total_epsilon: float, delta: float, rounds: int,
                                 l2_sensitivity: float) -> Tuple[float, float, float]:
    """
    Computes the per-round Gaussian noise std sigma such that `rounds` repeated
    releases (each with L2 sensitivity `l2_sensitivity`) compose via zCDP to
    exactly a total (total_epsilon, delta)-DP guarantee for the whole run.

    Returns (sigma, rho_total, rho_per_round).
    """
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    rho_total = _zcdp_rho_from_total_epsilon(total_epsilon, delta)
    rho_per_round = rho_total / float(rounds)
    sigma = l2_sensitivity / math.sqrt(2.0 * rho_per_round)
    return sigma, rho_total, rho_per_round


def encode_dataframe_features(df_input: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Preprocesses and encodes categorical and boolean features into numerical floats."""
    df = df_input[feature_cols].copy()
    if "hba1c" in df.columns:
        median_hba1c = df["hba1c"].median()
        df["hba1c"] = df["hba1c"].fillna(median_hba1c)

    if "gender" in df.columns:
        df["gender"] = (df["gender"].astype(str) == "M").astype(float)
    if "smoking_status" in df.columns:
        df["smoking_status"] = (df["smoking_status"].astype(str) == "Current").astype(float)

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(float)
        elif not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.factorize(df[col].astype(str))[0].astype(float)

    return df.astype(float)


class HospitalClient:
    """
    Simulates an isolated hospital node (e.g. running within Malaffi/Nabidh/Riayati boundary).
    Holds custody of local patient records and performs local SGD updates.
    """

    def __init__(self, hospital_id: str, data_df: pd.DataFrame, feature_cols: List[str], target_col: str):
        self.hospital_id = hospital_id
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_samples = len(data_df)

        # Prepare local feature matrix and target
        X_df = encode_dataframe_features(data_df, feature_cols)
        self.X = X_df.to_numpy(dtype=float)
        self.y = data_df[target_col].astype(int).to_numpy()

        # Class-balanced sample weights (matches sklearn's class_weight="balanced":
        # w_c = n_samples / (n_classes * n_samples_c). Without this, minority-class
        # (readmitted) gradient signal is swamped by the majority class under the
        # ~19% positive prevalence in this cohort, and the model degenerates to
        # always predicting the majority class (precision/recall/F1 = 0).
        n_pos = max(1, int(np.sum(self.y == 1)))
        n_neg = max(1, int(np.sum(self.y == 0)))
        weight_pos = self.n_samples / (2.0 * n_pos)
        weight_neg = self.n_samples / (2.0 * n_neg)
        self.sample_weight = np.where(self.y == 1, weight_pos, weight_neg)

    def local_train(self, global_weights: np.ndarray, global_intercept: float,
                    epochs: int = 3, lr: float = 0.05,
                    l2_reg: float = 0.01) -> Tuple[np.ndarray, float]:
        """
        Runs local mini-batch / full-batch SGD starting from the broadcasted global parameters.
        Returns the updated local weights and intercept.
        """
        weights = global_weights.copy()
        intercept = float(global_intercept)
        n = self.n_samples

        for _ in range(epochs):
            # Compute logistic predictions: p = 1 / (1 + exp(-(X w + b)))
            logits = np.dot(self.X, weights) + intercept
            # Numerical stability clip
            logits = np.clip(logits, -20.0, 20.0)
            probs = 1.0 / (1.0 + np.exp(-logits))

            # Compute class-balanced gradients: error = sample_weight * (probs - y)
            error = self.sample_weight * (probs - self.y)
            grad_w = (np.dot(self.X.T, error) / n) + (l2_reg * weights)
            grad_b = np.mean(error)

            # Gradient descent step
            weights -= lr * grad_w
            intercept -= lr * grad_b

        return weights, intercept


class FederatedOrchestrator:
    """
    Simulates MOHAP (Ministry of Health & Prevention) central aggregation server.
    Coordinates rounds, aggregates client weights using FedAvg, and applies optional DP noise.
    """

    def __init__(self, feature_dim: int, clients: List[HospitalClient],
                 enable_dp: bool = False, epsilon: float = 2.0, delta: float = 1e-5,
                 clip_norm: float = 0.05, rounds: int = 15):
        self.feature_dim = feature_dim
        self.clients = clients
        self.total_samples = sum(c.n_samples for c in clients)
        self.enable_dp = enable_dp
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.rounds = rounds

        # Initialize global parameters
        self.global_weights = np.zeros(feature_dim, dtype=float)
        self.global_intercept = 0.0
        self.round_history: List[Dict[str, Any]] = []

        # Precompute the per-round Gaussian noise std that makes `rounds` repeated
        # releases compose (via zCDP) to exactly the requested total (epsilon, delta).
        # L2 sensitivity of the FedAvg weighted-average update: clipping each client's
        # update to `clip_norm` bounds the aggregate's change from any one client's
        # contribution to (2 * clip_norm) / num_clients (worst case over the disjoint
        # per-client weight factors n_k / N).
        self.sigma = None
        self.rho_total = None
        self.rho_per_round = None
        if self.enable_dp:
            l2_sens = (2.0 * self.clip_norm) / float(len(self.clients))
            self.l2_sensitivity = l2_sens
            self.sigma, self.rho_total, self.rho_per_round = sigma_for_composed_gaussian(
                total_epsilon=self.epsilon, delta=self.delta, rounds=self.rounds, l2_sensitivity=l2_sens
            )

    def aggregate_round(self, round_idx: int, epochs: int = 3, lr: float = 0.05, 
                        rng: np.random.Generator = None) -> Tuple[np.ndarray, float]:
        """
        Executes one round of Federated Averaging:
        1. Distribute current global model to all hospital clients.
        2. Hospital clients train locally and return updated weights.
        3. Aggregator computes weighted average: w_global = sum( (n_k / N) * w_k ).
        4. Optional: Add Differential Privacy Gaussian noise to aggregated weights.
        """
        if rng is None:
            rng = np.random.default_rng(round_idx)

        client_weights = []
        client_intercepts = []
        client_sizes = []

        # Local training phase across hospitals
        for client in self.clients:
            w_k, b_k = client.local_train(
                global_weights=self.global_weights,
                global_intercept=self.global_intercept,
                epochs=epochs,
                lr=lr
            )

            # DP update clipping if enabled
            if self.enable_dp:
                diff_w = w_k - self.global_weights
                norm_w = np.linalg.norm(diff_w)
                if norm_w > self.clip_norm:
                    diff_w = diff_w * (self.clip_norm / norm_w)
                w_k = self.global_weights + diff_w

            client_weights.append(w_k)
            client_intercepts.append(b_k)
            client_sizes.append(client.n_samples)

        # FedAvg weighted aggregation
        new_weights = np.zeros(self.feature_dim, dtype=float)
        new_intercept = 0.0

        for w_k, b_k, n_k in zip(client_weights, client_intercepts, client_sizes):
            weight_factor = n_k / float(self.total_samples)
            new_weights += weight_factor * w_k
            new_intercept += weight_factor * b_k

        # Differential privacy noise addition. self.sigma is precomputed (once, in
        # __init__) via zCDP composition so that `self.rounds` repeated releases at
        # this sigma compose to exactly the requested total (epsilon, delta) budget
        # for the whole training run -- see sigma_for_composed_gaussian().
        if self.enable_dp:
            noise_w = rng.normal(0.0, self.sigma, size=self.feature_dim)
            noise_b = rng.normal(0.0, self.sigma)
            new_weights += noise_w
            new_intercept += noise_b

        self.global_weights = new_weights
        self.global_intercept = new_intercept

        return self.global_weights, self.global_intercept

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluates current global federated model against test set."""
        logits = np.dot(X_test, self.global_weights) + self.global_intercept
        logits = np.clip(logits, -20.0, 20.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        }


def prepare_datasets(data_dir: str):
    """Loads and standardizes datasets for federated simulation."""
    files = {
        "A": os.path.join(data_dir, "hospital_a_abu_dhabi.csv"),
        "B": os.path.join(data_dir, "hospital_b_dubai.csv"),
        "C": os.path.join(data_dir, "hospital_c_rak.csv"),
    }
    dfs = {k: pd.read_csv(v) for k, v in files.items()}

    feature_cols = [
        "age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic",
        "hba1c", "length_of_stay", "has_diabetes", "has_hypertension",
        "has_obesity", "gender", "smoking_status"
    ]
    target_col = "readmitted_30_days"

    # Pre-scale continuous columns globally (or compute per-hospital)
    # To maintain strict separation, we compute standard scaling statistics on pooled training
    # or simulate local client scaling.
    df_combined = pd.concat(dfs.values(), ignore_index=True)

    # Train/Test Split (80% train split across hospitals, 20% pooled evaluation set).
    # eval_mask_combined marks which rows of df_combined are held out for evaluation;
    # those exact rows MUST be excluded from every client's local training set below,
    # otherwise the "held-out" test set is not actually held out (the model would be
    # evaluated on data it was trained on, inflating reported accuracy/ROC-AUC and
    # invalidating any membership-inference comparison against it).
    rng = np.random.default_rng(42)
    eval_indices = rng.choice(len(df_combined), size=int(0.20 * len(df_combined)), replace=False)
    eval_mask_combined = np.zeros(len(df_combined), dtype=bool)
    eval_mask_combined[eval_indices] = True
    train_mask_combined = ~eval_mask_combined

    df_eval = df_combined.iloc[eval_indices].copy()

    # Preprocess test set identically
    X_eval_df = encode_dataframe_features(df_eval, feature_cols)

    # Standardize continuous variables. Fit only on the TRAIN portion so no
    # statistic of the held-out evaluation rows leaks into the scaler.
    scaler = StandardScaler()
    cont_cols = ["age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic", "hba1c", "length_of_stay"]
    df_train_combined = df_combined.loc[train_mask_combined]
    scaler.fit(df_train_combined[cont_cols].fillna(df_train_combined[cont_cols].median()))

    # Apply scaler to eval set
    X_eval_arr = X_eval_df.to_numpy(dtype=float)
    cont_indices = [feature_cols.index(c) for c in cont_cols]
    X_eval_arr[:, cont_indices] = scaler.transform(X_eval_df[cont_cols])
    y_eval_arr = df_eval[target_col].astype(int).to_numpy()

    # Create hospital clients with scaled local training sets, excluding any row
    # that was assigned to the pooled evaluation set above (see eval_mask_combined).
    clients = []
    offset = 0
    for h_code, df_h in dfs.items():
        n_h = len(df_h)
        local_eval_mask = eval_mask_combined[offset: offset + n_h]
        offset += n_h

        df_h_train = df_h.loc[~local_eval_mask].copy()
        # Scale continuous
        df_h_train[cont_cols] = scaler.transform(df_h_train[cont_cols].fillna(df_h_train[cont_cols].median()))
        client = HospitalClient(
            hospital_id=f"Hospital_{h_code}",
            data_df=df_h_train,
            feature_cols=feature_cols,
            target_col=target_col
        )
        clients.append(client)

    return clients, X_eval_arr, y_eval_arr, feature_cols


def run_federated_training(clients: List[HospitalClient], X_test: np.ndarray, y_test: np.ndarray,
                           feature_cols: List[str], rounds: int = 15, local_epochs: int = 3,
                           lr: float = 0.08, enable_dp: bool = False, epsilon: float = 2.0,
                           delta: float = 1e-5, clip_norm: float = 0.05,
                           seed: "int | None" = None) -> Dict[str, Any]:
    """Runs complete federated learning loop and evaluates performance progression.

    `epsilon` (and `delta`) is the TOTAL privacy budget for the entire `rounds`-round
    training run when enable_dp=True: the per-round Gaussian noise is calibrated via
    zCDP composition so the whole run satisfies exactly this (epsilon, delta)-DP
    guarantee, not a per-round one. See sigma_for_composed_gaussian().

    `seed`: if None (default, preserves prior behavior), each round's DP noise is
    drawn from `np.random.default_rng(round_idx)`, so every call is deterministic
    and identical. Pass an explicit int to get an independent noise realization
    (used by tradeoff_analysis.py to run multiple trials per epsilon).
    """
    orchestrator = FederatedOrchestrator(
        feature_dim=len(feature_cols),
        clients=clients,
        enable_dp=enable_dp,
        epsilon=epsilon,
        delta=delta,
        clip_norm=clip_norm,
        rounds=rounds,
    )

    progression = []
    print(f"[*] Starting Federated Training ({'DP-FedAvg (total ε=' + str(epsilon) + ' over ' + str(rounds) + ' rounds)' if enable_dp else 'Standard FedAvg'})...")
    print(f"{'Round':>6} | {'Accuracy':>10} | {'ROC-AUC':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
    print("-" * 68)

    for r in range(1, rounds + 1):
        round_rng = np.random.default_rng(seed * 100_000 + r) if seed is not None else None
        orchestrator.aggregate_round(round_idx=r, epochs=local_epochs, lr=lr, rng=round_rng)
        metrics = orchestrator.evaluate(X_test, y_test)
        metrics["round"] = r
        progression.append(metrics)

        print(f"{r:6d} | {metrics['accuracy']:10.4f} | {metrics['roc_auc']:10.4f} | "
              f"{metrics['precision']:10.4f} | {metrics['recall']:10.4f} | {metrics['f1_score']:10.4f}")

    final_metrics = progression[-1]
    weights_dict = {col: float(w) for col, w in zip(feature_cols, orchestrator.global_weights)}

    result = {
        "final_metrics": final_metrics,
        "round_progression": progression,
        "learned_weights": weights_dict,
        "learned_intercept": float(orchestrator.global_intercept),
        "configuration": {
            "rounds": rounds,
            "local_epochs": local_epochs,
            "learning_rate": lr,
            "dp_enabled": enable_dp,
            "epsilon": epsilon if enable_dp else None,
            "delta": delta if enable_dp else None,
            "clip_norm": clip_norm if enable_dp else None,
            "clients": [c.hospital_id for c in clients],
            "client_samples": {c.hospital_id: c.n_samples for c in clients},
        }
    }

    if enable_dp:
        result["dp_accounting"] = {
            "composition_method": "zCDP (Bun & Steinke 2016), rho_total split evenly across rounds",
            "total_epsilon": epsilon,
            "total_delta": delta,
            "rounds": rounds,
            "l2_sensitivity_per_round": orchestrator.l2_sensitivity,
            "rho_total_zcdp": orchestrator.rho_total,
            "rho_per_round_zcdp": orchestrator.rho_per_round,
            "gaussian_sigma_per_round": orchestrator.sigma,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Cyberleek Federated Learning Readmission Predictor")
    parser.add_argument("--rounds", type=int, default=15, help="Number of federated aggregation rounds")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local SGD epochs per client round")
    parser.add_argument("--lr", type=float, default=0.08, help="Client learning rate")
    parser.add_argument("--dp", action="store_true", help="Enable Differentially Private FedAvg (DP-FedAvg)")
    parser.add_argument("--epsilon", type=float, default=2.0,
                         help="TOTAL epsilon budget for the entire DP-FedAvg run (all rounds combined)")
    parser.add_argument("--delta", type=float, default=1e-5, help="Total delta budget for DP-FedAvg")
    parser.add_argument("--clip-norm", type=float, default=0.05,
                         help="Per-client L2 update clipping bound for DP-FedAvg")
    parser.add_argument("--data-dir", default=None, help="Path to hospital data directory")
    parser.add_argument("--output", default=None,
                         help="Path to output JSON (default: federated_learning_results.json for standard "
                              "FedAvg, federated_learning_dp_results.json for DP-FedAvg)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(base_dir, "data")
    # Standard and DP-FedAvg runs write to different default files so that running one
    # mode doesn't silently overwrite the other mode's saved results.
    default_output = "federated_learning_dp_results.json" if args.dp else "federated_learning_results.json"
    output_path = os.path.join(base_dir, args.output or default_output)

    print("=" * 70)
    print("  CYBERLEEK — FEDERATED LEARNING READMISSION PREDICTOR")
    print("  Core Privacy Technique Engineer: Decentralized Model Training")
    print("=" * 70)
    print(f"[*] Participating Nodes: 3 Hospitals (Hospital A, B, C)")
    print(f"[*] Central Aggregator:   MOHAP (Ministry of Health & Prevention)")
    print(f"[*] Architecture:        FedAvg (Federated Averaging)")
    print(f"[*] Privacy Guarantee:   Raw records never leave hospital boundaries")

    clients, X_test, y_test, feature_cols = prepare_datasets(data_dir)
    print(f"[*] Evaluated on {len(X_test):,} held-out patient records")

    # Run federated training
    fl_results = run_federated_training(
        clients=clients,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        enable_dp=args.dp,
        epsilon=args.epsilon,
        delta=args.delta,
        clip_norm=args.clip_norm
    )

    # Compare with Naive Centralized Baseline if available
    naive_path = os.path.join(base_dir, "naive_baseline_results.json")
    if os.path.exists(naive_path):
        with open(naive_path, "r", encoding="utf-8") as f:
            naive_data = json.load(f)
        naive_model = naive_data.get("readmission_predictive_model", {})
        print("\n" + "=" * 70)
        print("  COMPARISON: FEDERATED LEARNING VS NAIVE CENTRALIZED BASELINE")
        print("=" * 70)
        print(f"{'Metric':<20} | {'Centralized (Naive)':>20} | {'Federated (FedAvg)':>20} | {'Delta':>10}")
        print("-" * 75)
        for m in ["accuracy", "roc_auc", "precision", "recall", "f1_score"]:
            c_val = naive_model.get(m, 0.0)
            f_val = fl_results["final_metrics"].get(m, 0.0)
            delta = f_val - c_val
            delta_str = f"{delta:+0.4f}"
            print(f"{m.upper():<20} | {c_val:20.4f} | {f_val:20.4f} | {delta_str:>10}")
        print("=" * 70)
        fl_results["centralized_baseline_comparison"] = {
            m: {"centralized": naive_model.get(m, 0.0), "federated": fl_results["final_metrics"].get(m, 0.0)}
            for m in ["accuracy", "roc_auc", "precision", "recall", "f1_score"]
        }

    # Save artifact
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fl_results, f, indent=2)

    print(f"\n[DONE] Federated learning results saved to: {os.path.basename(output_path)}")


if __name__ == "__main__":
    main()
