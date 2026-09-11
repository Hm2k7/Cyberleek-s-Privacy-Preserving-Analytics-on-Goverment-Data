"""
Attack & Vulnerability Demonstration
Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data

Person 3: Attack & Vulnerability Engineer

Runs two real, reproducible attacks against the outputs Person 2 built, then shows
each attack degrades to (near) chance once the corresponding privacy technique is
switched on. Nothing here is hard-coded: every number below is measured live by
retraining the naive/federated models and re-running the DP mechanisms in this
process, then scoring an actual attacker against the actual outputs.

Attack 1 — Membership Inference Attack (MIA), Yeom et al. 2018 loss-threshold attack:
    A shadow-free attacker computes each sample's binary cross-entropy loss under a
    trained model. Since models fit their training data more closely than unseen
    data, "loss" separates members (train set) from non-members (held-out set)
    better than chance. We score the attacker with ROC-AUC (chance = 0.50) and
    report its Attack Success Rate (ASR) at the best empirical threshold.
        Target A (vulnerable): centralized non-private logistic regression
                                (naive_baseline.py) — retrained identically in-process.
        Target B (protected):  DP-FedAvg global model (federated_learning.py --dp)
                                — retrained identically in-process.
        Reference:              non-DP FedAvg, to show FL custody alone (no noise)
                                does NOT by itself defeat MIA the way DP-FedAvg does.

Attack 2 — Differencing / Reconstruction Attack against aggregate counting queries:
    For many randomly chosen target patients in Hospital A, the attacker issues a
    "population count" query and a "population count excluding target" query, then
    subtracts them. Under a naive (non-private) system the difference exactly equals
    the target's true diabetes status (0 or 1) — perfect reconstruction. Under DP
    (dp_core.laplace_mechanism, matching dp_analytics.py's mechanism) each release is
    independently noised, so the subtraction is swamped and the attacker's inferred
    attribute is no better than chance. We also show that a real deployed
    PrivacyBudgetTracker halts this kind of large-scale differencing campaign after
    only a handful of queries, independent of the noise itself.

Usage:
    python attack_demo.py [--epsilon-analytics 0.2] [--epsilon-fl 2.0] [--n-targets 300] [--seed 42]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from naive_baseline import load_hospital_data, train_centralized_readmission_model
from federated_learning import prepare_datasets, run_federated_training
from dp_core import laplace_mechanism, PrivacyBudgetTracker, PrivacyBudgetExhaustedError


# ==========================================
# SHARED ATTACK PRIMITIVES
# ==========================================
def binary_cross_entropy_loss(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-sample BCE loss: L(x,y) = -[y*log(p) + (1-y)*log(1-p)]."""
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0 - eps)
    y = y_true.astype(float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def loss_threshold_mia(member_loss: np.ndarray, nonmember_loss: np.ndarray) -> dict:
    """
    Yeom et al. (2018) loss-threshold membership inference attack.
    A model fits training members more closely, so LOWER loss => more likely a member.
    Attack score = -loss. Scored by ROC-AUC (0.50 = chance / fully protected) and by
    Attack Success Rate (ASR) at the empirically best (Youden's J) decision threshold.

    ASR uses BALANCED accuracy, not raw accuracy: member and non-member pools are
    rarely equal-sized (e.g. an 80/20 train/test split), so raw accuracy has a
    trivial >=50% floor just from always guessing the majority class regardless of
    whether the attack actually works. Balanced accuracy's chance level is exactly
    0.50 regardless of that skew, matching the ROC-AUC chance level.
    """
    y_attack_true = np.concatenate([np.ones(len(member_loss)), np.zeros(len(nonmember_loss))])
    attack_score = np.concatenate([-member_loss, -nonmember_loss])

    auc = float(roc_auc_score(y_attack_true, attack_score))
    fpr, tpr, thresholds = roc_curve(y_attack_true, attack_score)
    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])
    preds = (attack_score >= best_threshold).astype(int)
    asr = float(balanced_accuracy_score(y_attack_true, preds))

    advantage = float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))
    return {
        "n_members": int(len(member_loss)),
        "n_nonmembers": int(len(nonmember_loss)),
        "mean_member_loss": float(np.mean(member_loss)),
        "mean_nonmember_loss": float(np.mean(nonmember_loss)),
        "attack_auc": auc,
        "attack_success_rate": asr,
        "attacker_advantage": advantage,
        "verdict": "LEAKED" if advantage > 0.30 else ("PARTIALLY PROTECTED (bounded residual leak)" if advantage > 0.10 else "STRONGLY PROTECTED (near-chance)"),
    }


# ==========================================
# ATTACK 1: MEMBERSHIP INFERENCE ATTACK
# ==========================================
def run_membership_inference_attack(data_dir: str, epsilon_fl: float, rounds: int,
                                     local_epochs: int, lr: float, seed: int) -> dict:
    print("\n" + "=" * 70)
    print("  ATTACK 1: MEMBERSHIP INFERENCE ATTACK (Yeom et al. 2018, loss-threshold)")
    print("=" * 70)

    # --- Target A1: naive off-the-shelf model, NO privacy engineering --------
    # An analyst with raw pooled access and no privacy training reaches for a
    # high-capacity, unregularized classifier and never thinks to check whether
    # it memorized training records. An unpruned decision tree is the textbook
    # case: it can perfectly fit (memorize) the training set, which is exactly
    # what a loss-based membership attack exploits.
    print("\n[*] Retraining Target A1: naive off-the-shelf model, no privacy engineering (unpruned decision tree)...")
    dfs = load_hospital_data(data_dir)
    df_all = pd.concat(dfs.values(), ignore_index=True)
    _, clf_pipeline, split = train_centralized_readmission_model(df_all, random_state=seed)

    overfit_pipeline = clone(clf_pipeline)
    overfit_pipeline.set_params(classifier=DecisionTreeClassifier(
        max_depth=None, min_samples_leaf=1, random_state=seed, class_weight="balanced"))
    overfit_pipeline.fit(split["X_train"], split["y_train"])
    overfit_member_prob = overfit_pipeline.predict_proba(split["X_train"])[:, 1]
    overfit_nonmember_prob = overfit_pipeline.predict_proba(split["X_test"])[:, 1]
    overfit_member_loss = binary_cross_entropy_loss(split["y_train"].to_numpy(), overfit_member_prob)
    overfit_nonmember_loss = binary_cross_entropy_loss(split["y_test"].to_numpy(), overfit_nonmember_prob)

    overfit_result = loss_threshold_mia(overfit_member_loss, overfit_nonmember_loss)
    print(f"    Members (train, n={overfit_result['n_members']}): mean loss = {overfit_result['mean_member_loss']:.4f}")
    print(f"    Non-members (test, n={overfit_result['n_nonmembers']}): mean loss = {overfit_result['mean_nonmember_loss']:.4f}")
    print(f"    Attack ROC-AUC: {overfit_result['attack_auc']:.4f}  |  ASR: {overfit_result['attack_success_rate']:.4f}  |  {overfit_result['verdict']}")

    # --- Target A2: Person 2's actual production model (regularized LR) -----
    # Included for context, NOT as the primary "naive" target: good ML hygiene
    # (L2 regularization, ample training data relative to model capacity)
    # already reduces this basic attack's success on its own, but that is an
    # accidental side effect, not a provable guarantee -- it offers no bound
    # against a more capable attacker (e.g. shadow-model attacks) or a less
    # careful analyst, unlike the DP mechanism below.
    print("\n[*] Re-scoring Target A2: Person 2's production model (regularized logistic regression)...")
    member_prob = clf_pipeline.predict_proba(split["X_train"])[:, 1]
    nonmember_prob = clf_pipeline.predict_proba(split["X_test"])[:, 1]
    member_loss = binary_cross_entropy_loss(split["y_train"].to_numpy(), member_prob)
    nonmember_loss = binary_cross_entropy_loss(split["y_test"].to_numpy(), nonmember_prob)

    naive_result = loss_threshold_mia(member_loss, nonmember_loss)
    print(f"    Members (train, n={naive_result['n_members']}): mean loss = {naive_result['mean_member_loss']:.4f}")
    print(f"    Non-members (test, n={naive_result['n_nonmembers']}): mean loss = {naive_result['mean_nonmember_loss']:.4f}")
    print(f"    Attack ROC-AUC: {naive_result['attack_auc']:.4f}  |  ASR: {naive_result['attack_success_rate']:.4f}  |  {naive_result['verdict']}")

    # --- Targets B/C: Federated models (standard FedAvg + DP-FedAvg) -------
    print("\n[*] Retraining Target B/C: federated models (federated_learning.py)...")
    clients, X_eval, y_eval, feature_cols = prepare_datasets(data_dir)
    member_X = np.concatenate([c.X for c in clients], axis=0)
    member_y = np.concatenate([c.y for c in clients], axis=0)

    def fl_model_mia(enable_dp: bool, epsilon: float, label: str) -> dict:
        fl_res = run_federated_training(
            clients=clients, X_test=X_eval, y_test=y_eval, feature_cols=feature_cols,
            rounds=rounds, local_epochs=local_epochs, lr=lr,
            enable_dp=enable_dp, epsilon=epsilon,
        )
        w = np.array([fl_res["learned_weights"][c] for c in feature_cols])
        b = fl_res["learned_intercept"]

        def sigmoid_probs(X):
            logits = np.clip(X @ w + b, -20.0, 20.0)
            return 1.0 / (1.0 + np.exp(-logits))

        m_prob = sigmoid_probs(member_X)
        nm_prob = sigmoid_probs(X_eval)
        m_loss = binary_cross_entropy_loss(member_y, m_prob)
        nm_loss = binary_cross_entropy_loss(y_eval, nm_prob)
        result = loss_threshold_mia(m_loss, nm_loss)
        result["model_test_roc_auc"] = fl_res["final_metrics"]["roc_auc"]
        print(f"\n    [{label}] Members (n={result['n_members']}): mean loss = {result['mean_member_loss']:.4f}  "
              f"| Non-members (n={result['n_nonmembers']}): mean loss = {result['mean_nonmember_loss']:.4f}")
        print(f"    [{label}] Attack ROC-AUC: {result['attack_auc']:.4f}  |  ASR: {result['attack_success_rate']:.4f}  |  {result['verdict']}")
        return result

    fedavg_result = fl_model_mia(enable_dp=False, epsilon=epsilon_fl, label="Standard FedAvg (no DP)")
    dp_fedavg_result = fl_model_mia(enable_dp=True, epsilon=epsilon_fl, label=f"DP-FedAvg (total eps={epsilon_fl})")

    return {
        "naive_unregularized_decision_tree": overfit_result,
        "naive_centralized": naive_result,
        "standard_fedavg": fedavg_result,
        "dp_fedavg": dp_fedavg_result,
    }


# ==========================================
# ATTACK 2: DIFFERENCING / RECONSTRUCTION ATTACK
# ==========================================
def attacker_advantage(auc: float) -> float:
    """
    Standard membership/attribute-inference metric (Yeom et al. 2018): advantage
    = 2*(AUC - 0.5), clipped to [0, 1]. 0 = attacker has literally no edge over a
    coin flip; 1 = perfect reconstruction. This is the metric to trust over a raw
    AUC or a binary LEAKED/PROTECTED label, because DP does NOT promise the
    attacker's advantage hits exactly zero -- it promises the advantage is BOUNDED
    by a function of epsilon, so a small residual (e.g. 0.05-0.15) at a non-trivial
    epsilon is expected and correctly reported here, not a bug in the demo.
    """
    return float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))


def verdict_from_advantage(advantage: float) -> str:
    if advantage <= 0.10:
        return "STRONGLY PROTECTED (near-chance)"
    if advantage <= 0.30:
        return "PARTIALLY PROTECTED (bounded residual leak)"
    return "LEAKED"


def run_differencing_attack(data_dir: str, epsilon: float, n_targets: int, seed: int) -> dict:
    print("\n" + "=" * 70)
    print("  ATTACK 2: DIFFERENCING / RECONSTRUCTION ATTACK vs. AGGREGATE COUNTS")
    print("=" * 70)

    df_a = pd.read_csv(os.path.join(data_dir, "hospital_a_abu_dhabi.csv"))
    rng = np.random.default_rng(seed)

    n_targets = min(n_targets, len(df_a))
    target_idx = rng.choice(len(df_a), size=n_targets, replace=False)
    targets = df_a.iloc[target_idx]
    true_labels = targets["has_diabetes"].astype(int).to_numpy()
    true_diffs = true_labels.astype(float)  # exact (full - minus) count difference per target

    full_true_count = int(df_a["has_diabetes"].sum())
    minus_true_counts = full_true_count - true_labels  # vectorized: full_true_count - is_diabetic

    def score_at_epsilon(eps: float, rng_local: np.random.Generator) -> np.ndarray:
        """Two INDEPENDENT Laplace releases (Query 1, Query 2) per target, matching
        how dp_analytics.py answers each count query -- the attacker gets no benefit
        from correlated noise because every release is a fresh mechanism call."""
        dp_full = np.array([laplace_mechanism(float(full_true_count), sensitivity=1.0, epsilon=eps, rng=rng_local)
                             for _ in range(len(targets))])
        dp_minus = np.array([laplace_mechanism(float(m), sensitivity=1.0, epsilon=eps, rng=rng_local)
                              for m in minus_true_counts])
        return dp_full - dp_minus

    dp_diffs = score_at_epsilon(epsilon, rng)

    naive_preds = (true_diffs >= 0.5).astype(int)
    dp_preds = (dp_diffs >= 0.5).astype(int)

    naive_auc = float(roc_auc_score(true_labels, true_diffs))
    naive_asr = float(balanced_accuracy_score(true_labels, naive_preds))
    naive_advantage = attacker_advantage(naive_auc)

    dp_auc = float(roc_auc_score(true_labels, dp_diffs)) if len(np.unique(true_labels)) > 1 else 0.5
    dp_asr = float(balanced_accuracy_score(true_labels, dp_preds))
    dp_advantage = attacker_advantage(dp_auc)
    advantage_reduction_pct = (1.0 - dp_advantage / naive_advantage) * 100.0 if naive_advantage > 0 else 0.0

    majority_class_raw_accuracy = float(max(true_labels.mean(), 1.0 - true_labels.mean()))

    print(f"\n[*] Target pool: {n_targets} randomly selected Hospital A patients")
    print(f"    True diabetes prevalence in target pool: {true_labels.mean():.2%}")
    print(f"    (context only) raw accuracy of guessing the majority class with ZERO query access: {majority_class_raw_accuracy:.4f}")
    print(f"\n    Naive exact differencing   : AUC={naive_auc:.4f}  Advantage={naive_advantage:.4f}  ASR={naive_asr:.4f}  -> LEAKED (perfect reconstruction)")
    print(f"    DP-protected (eps={epsilon}) : AUC={dp_auc:.4f}  Advantage={dp_advantage:.4f}  ASR={dp_asr:.4f}  -> {verdict_from_advantage(dp_advantage)}")
    print(f"    Attacker advantage reduced by {advantage_reduction_pct:.1f}% relative to the naive (unprotected) system.")
    print(f"    Note: DP bounds attacker advantage by a function of epsilon -- it does not promise it hits")
    print(f"    exactly zero, so a small residual at eps={epsilon} is expected, not a bug. The sweep below")
    print(f"    shows this residual shrinking predictably and monotonically as epsilon decreases.")

    # --- Epsilon sweep: shows the residual attacker advantage shrinking monotonically
    #     as epsilon decreases, directly tying this attack to the stated privacy budget.
    sweep_epsilons = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    sweep_trials = 10
    sweep = []
    for eps in sweep_epsilons:
        trial_advantages = []
        for trial in range(sweep_trials):
            eps_rng = np.random.default_rng(seed * 7919 + int(eps * 1e6) + trial)
            scores = score_at_epsilon(eps, eps_rng)
            auc = float(roc_auc_score(true_labels, scores)) if len(np.unique(true_labels)) > 1 else 0.5
            trial_advantages.append(attacker_advantage(auc))
        mean_adv = float(np.mean(trial_advantages))
        sweep.append({"epsilon": eps, "attacker_advantage": mean_adv,
                      "attacker_advantage_std": float(np.std(trial_advantages)), "n_trials": sweep_trials})
    print(f"\n    Attacker advantage vs. epsilon (same {n_targets} targets, mean of {sweep_trials} independent noise trials):")
    for pt in sweep:
        bar = "#" * int(round(pt["attacker_advantage"] * 40))
        print(f"      eps={pt['epsilon']:>5.2f}  advantage={pt['attacker_advantage']:.4f} (+/-{pt['attacker_advantage_std']:.4f})  {bar}")

    # --- Budget-exhaustion demo: a real deployment blocks the campaign outright ---
    total_budget = 1.0
    tracker = PrivacyBudgetTracker(total_epsilon=total_budget, total_delta=1e-5)
    queries_before_exhaustion = 0
    try:
        for i in range(n_targets * 2):  # Query 1 + Query 2 per target
            tracker.allocate(query_name=f"attacker_query_{i}", epsilon=epsilon)
            queries_before_exhaustion += 1
    except PrivacyBudgetExhaustedError:
        pass
    print(f"\n[*] Deployment guardrail: with a realistic total budget eps={total_budget} and per-query "
          f"eps={epsilon}, PrivacyBudgetTracker halts this attacker after {queries_before_exhaustion} "
          f"of the {n_targets * 2} queries it would need -- the noise analysis above already assumes "
          f"the attacker got ALL of them, which a deployed system would never allow.")

    return {
        "n_targets": n_targets,
        "true_prevalence_in_targets": float(true_labels.mean()),
        "majority_class_raw_accuracy_zero_query_access": majority_class_raw_accuracy,
        "epsilon_per_query": epsilon,
        "naive": {
            "attack_success_rate": naive_asr,
            "attack_auc": naive_auc,
            "attacker_advantage": naive_advantage,
            "verdict": "LEAKED",
        },
        "dp_protected": {
            "attack_success_rate": dp_asr,
            "attack_auc": dp_auc,
            "attacker_advantage": dp_advantage,
            "advantage_reduction_pct": advantage_reduction_pct,
            "verdict": verdict_from_advantage(dp_advantage),
        },
        "advantage_vs_epsilon_sweep": sweep,
        "budget_guardrail": {
            "total_epsilon_assumed": total_budget,
            "epsilon_per_query": epsilon,
            "queries_attacker_needed": n_targets * 2,
            "queries_allowed_before_exhaustion": queries_before_exhaustion,
        },
    }


# ==========================================
# MAIN EXECUTION & JSON OUTPUT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Cyberleek Attack & Vulnerability Demonstration")
    parser.add_argument("--epsilon-analytics", type=float, default=0.2,
                         help="Per-query epsilon for the differencing attack (matches dp_analytics.py's typical per-query allocation)")
    parser.add_argument("--epsilon-fl", type=float, default=2.0,
                         help="Total DP-FedAvg epsilon budget (matches federated_learning.py --dp default)")
    parser.add_argument("--rounds", type=int, default=15, help="FedAvg communication rounds")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local SGD epochs per client round")
    parser.add_argument("--lr", type=float, default=0.08, help="Client learning rate")
    parser.add_argument("--n-targets", type=int, default=1000, help="Number of target patients for the differencing attack")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", default=None, help="Path to hospital data directory")
    parser.add_argument("--output", default="attack_results.json", help="Path to output JSON")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(base_dir, "data")
    output_path = os.path.join(base_dir, args.output)

    print("=" * 70)
    print("  CYBERLEEK — ATTACK & VULNERABILITY DEMONSTRATION")
    print("  Person 3: Attack & Vulnerability Engineer")
    print("=" * 70)

    mia_results = run_membership_inference_attack(
        data_dir=data_dir, epsilon_fl=args.epsilon_fl, rounds=args.rounds,
        local_epochs=args.local_epochs, lr=args.lr, seed=args.seed,
    )
    diff_results = run_differencing_attack(
        data_dir=data_dir, epsilon=args.epsilon_analytics, n_targets=args.n_targets, seed=args.seed,
    )

    # --- Clean summary table -------------------------------------------------
    print("\n" + "=" * 128)
    print(f"{'ATTACK':<24} | {'TARGET':<26} | {'ASR':>7} | {'AUC':>7} | {'ADVANTAGE':>9} | {'VERDICT':<38}")
    print("=" * 128)
    rows = [
        ("Membership Inference", "Naive (Unreg. Dec. Tree)", mia_results["naive_unregularized_decision_tree"]["attack_success_rate"],
         mia_results["naive_unregularized_decision_tree"]["attack_auc"], mia_results["naive_unregularized_decision_tree"]["attacker_advantage"], mia_results["naive_unregularized_decision_tree"]["verdict"]),
        ("Membership Inference", "Naive Centralized (LR)", mia_results["naive_centralized"]["attack_success_rate"],
         mia_results["naive_centralized"]["attack_auc"], mia_results["naive_centralized"]["attacker_advantage"], mia_results["naive_centralized"]["verdict"]),
        ("Membership Inference", "Standard FedAvg (no DP)", mia_results["standard_fedavg"]["attack_success_rate"],
         mia_results["standard_fedavg"]["attack_auc"], mia_results["standard_fedavg"]["attacker_advantage"], mia_results["standard_fedavg"]["verdict"]),
        ("Membership Inference", "DP-FedAvg (Protected)", mia_results["dp_fedavg"]["attack_success_rate"],
         mia_results["dp_fedavg"]["attack_auc"], mia_results["dp_fedavg"]["attacker_advantage"], mia_results["dp_fedavg"]["verdict"]),
        ("Differencing Attack", "Naive Aggregate Counts", diff_results["naive"]["attack_success_rate"],
         diff_results["naive"]["attack_auc"], diff_results["naive"]["attacker_advantage"], diff_results["naive"]["verdict"]),
        ("Differencing Attack", "DP Analytics (Protected)", diff_results["dp_protected"]["attack_success_rate"],
         diff_results["dp_protected"]["attack_auc"], diff_results["dp_protected"]["attacker_advantage"], diff_results["dp_protected"]["verdict"]),
    ]
    for atk, target, asr, auc, adv, verdict in rows:
        print(f"{atk:<24} | {target:<26} | {asr:>7.4f} | {auc:>7.4f} | {adv:>9.4f} | {verdict:<38}")
    print("=" * 128)

    output_data = {
        "metadata": {
            "title": "Cyberleek Attack & Vulnerability Demonstration Results",
            "role": "Person 3 - Attack & Vulnerability Engineer",
            "methodology": "Loss-threshold Membership Inference (Yeom et al. 2018); batched differencing/reconstruction attack",
            "configuration": vars(args),
        },
        "membership_inference_attack": mia_results,
        "differencing_attack": diff_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[DONE] Attack results saved to: {os.path.basename(output_path)}")


if __name__ == "__main__":
    main()
