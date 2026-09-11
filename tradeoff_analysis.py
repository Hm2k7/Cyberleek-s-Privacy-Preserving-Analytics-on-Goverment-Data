"""
Privacy-Utility Trade-Off Analysis
Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data

Person 4: Trade-Off & Evaluation Lead

Sweeps the privacy budget (epsilon) across two independent tracks and measures the
resulting utility cost, so MOHAP can pick a defensible operating point for each:

Track 1 — DP Analytics (population statistics, dp_analytics.py):
    For each epsilon, re-runs run_dp_analytics() across N_TRIALS independent seeds
    and measures Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the
    DP-released National Diabetes Prevalence against the true (naive) value.

Track 2 — DP-FedAvg (readmission risk model, federated_learning.py --dp):
    For each epsilon (interpreted as the TOTAL budget for the whole 15-round run,
    exactly as federated_learning.py defines it), re-runs run_federated_training()
    across N_TRIALS independent noise seeds and measures test ROC-AUC / accuracy.

The two tracks are NOT combined into one curve or one recommended epsilon: HANDOVER.md
documents (and this sweep independently confirms) that the DP-analytics Pareto knee
sits near eps~1.0 while the DP-FedAvg knee sits far further out (eps~5-20) -- collapsing
them into a single "recommended epsilon" would misrepresent the FL model's real
privacy cost.

Usage:
    python tradeoff_analysis.py [--trials 20] [--output-prefix tradeoff]
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dp_analytics import load_hospital_silos, run_dp_analytics
from federated_learning import prepare_datasets, run_federated_training

EPSILON_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
FL_EPSILON_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def sweep_dp_analytics(data_dir: str, n_trials: int) -> dict:
    """Track 1: epsilon vs. National Diabetes Prevalence error (DP analytics)."""
    print("\n" + "=" * 70)
    print("  TRACK 1: DP ANALYTICS -- epsilon vs. Population Statistic Error")
    print("=" * 70)

    silos = load_hospital_silos(data_dir)
    true_value = None
    curve = []

    for eps in EPSILON_GRID:
        errors = []
        for trial in range(n_trials):
            report = run_dp_analytics(silos, total_epsilon=eps, seed=1000 * trial + 1)
            q = report["queries"]["national_diabetes_prevalence"]
            if true_value is None:
                true_value = q["true_value"]
            errors.append(q["dp_estimate"] - q["true_value"])

        errors = np.array(errors)
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        std = float(np.std(errors))
        curve.append({
            "epsilon": eps, "mae": mae, "rmse": rmse, "std": std,
            "mae_pct_of_true": (mae / true_value) * 100.0,
            "n_trials": n_trials,
        })
        print(f"  eps={eps:>6.2f}  MAE={mae*100:7.4f}%  RMSE={rmse*100:7.4f}%  "
              f"(relative to true value {true_value*100:.2f}%)")

    return {"true_value": true_value, "curve": curve}


def sweep_dp_fedavg(data_dir: str, n_trials: int, rounds: int, local_epochs: int, lr: float) -> dict:
    """Track 2: epsilon vs. DP-FedAvg readmission model utility."""
    print("\n" + "=" * 70)
    print("  TRACK 2: DP-FEDAVG -- epsilon vs. Readmission Model Utility")
    print("=" * 70)

    clients, X_eval, y_eval, feature_cols = prepare_datasets(data_dir)

    # Non-DP FedAvg ceiling, for reference on the plot.
    non_dp_res = run_federated_training(
        clients=clients, X_test=X_eval, y_test=y_eval, feature_cols=feature_cols,
        rounds=rounds, local_epochs=local_epochs, lr=lr, enable_dp=False,
    )
    non_dp_auc = non_dp_res["final_metrics"]["roc_auc"]
    non_dp_acc = non_dp_res["final_metrics"]["accuracy"]
    print(f"  Non-DP FedAvg ceiling: ROC-AUC={non_dp_auc:.4f}  Accuracy={non_dp_acc:.4f}")

    curve = []
    for eps in FL_EPSILON_GRID:
        aucs, accs = [], []
        for trial in range(n_trials):
            res = run_federated_training(
                clients=clients, X_test=X_eval, y_test=y_eval, feature_cols=feature_cols,
                rounds=rounds, local_epochs=local_epochs, lr=lr,
                enable_dp=True, epsilon=eps, seed=trial + 1,
            )
            aucs.append(res["final_metrics"]["roc_auc"])
            accs.append(res["final_metrics"]["accuracy"])

        aucs = np.array(aucs)
        accs = np.array(accs)
        curve.append({
            "epsilon": eps,
            "mean_roc_auc": float(np.mean(aucs)), "std_roc_auc": float(np.std(aucs)),
            "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
            "n_trials": n_trials,
        })
        print(f"  eps={eps:>6.2f}  ROC-AUC={np.mean(aucs):.4f} (+/-{np.std(aucs):.4f})  "
              f"Accuracy={np.mean(accs):.4f} (+/-{np.std(accs):.4f})")

    return {"non_dp_ceiling": {"roc_auc": non_dp_auc, "accuracy": non_dp_acc}, "curve": curve}


def find_analytics_knee(curve: list, mae_threshold_pct: float = 0.5) -> float:
    """First epsilon (ascending) at which MAE drops below `mae_threshold_pct`% of the true value."""
    for point in curve:
        if point["mae_pct_of_true"] <= mae_threshold_pct:
            return point["epsilon"]
    return curve[-1]["epsilon"]


def find_fedavg_knee(curve: list, non_dp_auc: float, auc_gap_threshold: float = 0.03) -> float:
    """First epsilon (ascending) at which ROC-AUC comes within `auc_gap_threshold` of the non-DP ceiling."""
    for point in curve:
        if (non_dp_auc - point["mean_roc_auc"]) <= auc_gap_threshold:
            return point["epsilon"]
    return curve[-1]["epsilon"]


def plot_tradeoff_curves(analytics_result: dict, fedavg_result: dict, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Plot 1: DP Analytics error vs epsilon --------------------------------
    ax1 = axes[0]
    eps_vals = [p["epsilon"] for p in analytics_result["curve"]]
    mae_vals = [p["mae_pct_of_true"] for p in analytics_result["curve"]]
    std_vals = [p["std"] * 100.0 / analytics_result["true_value"] for p in analytics_result["curve"]]

    ax1.plot(eps_vals, mae_vals, marker="o", color="#1f77b4", label="MAE (% of true value)")
    ax1.fill_between(eps_vals,
                      [max(0, m - s) for m, s in zip(mae_vals, std_vals)],
                      [m + s for m, s in zip(mae_vals, std_vals)],
                      color="#1f77b4", alpha=0.15, label="±1 std across trials")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5% error (illustrative reference)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Privacy Budget (ε), log scale", fontsize=11)
    ax1.set_ylabel("National Diabetes Prevalence Error (% of true value)", fontsize=11)
    ax1.set_title("Track 1: DP Analytics\n(Population Statistic Utility)", fontsize=13)
    ax1.tick_params(labelsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: DP-FedAvg ROC-AUC / Accuracy vs epsilon ----------------------
    ax2 = axes[1]
    fl_eps = [p["epsilon"] for p in fedavg_result["curve"]]
    fl_auc = [p["mean_roc_auc"] for p in fedavg_result["curve"]]
    fl_auc_std = [p["std_roc_auc"] for p in fedavg_result["curve"]]
    fl_acc = [p["mean_accuracy"] for p in fedavg_result["curve"]]

    ax2.errorbar(fl_eps, fl_auc, yerr=fl_auc_std, marker="o", color="#d62728",
                 label="DP-FedAvg Test ROC-AUC", capsize=3)
    ax2.plot(fl_eps, fl_acc, marker="s", color="#ff7f0e", linestyle="--",
             label="DP-FedAvg Test Accuracy", alpha=0.8)
    ax2.axhline(fedavg_result["non_dp_ceiling"]["roc_auc"], color="#2ca02c", linestyle=":",
                label=f"Non-DP FedAvg ceiling (ROC-AUC={fedavg_result['non_dp_ceiling']['roc_auc']:.3f})")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random guessing (0.50)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Total Privacy Budget (ε) over 15 rounds, log scale", fontsize=11)
    ax2.set_ylabel("Model Performance", fontsize=11)
    ax2.set_title("Track 2: DP-FedAvg\n(Readmission Risk Model Utility)", fontsize=13)
    ax2.tick_params(labelsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Cyberleek: Privacy-Utility Trade-Off Curves (UAE MOHAP Healthcare Demonstrator)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\n[*] Trade-off curves saved to: {os.path.basename(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="Cyberleek Privacy-Utility Trade-Off Analysis")
    parser.add_argument("--trials", type=int, default=20, help="Independent trials per epsilon value")
    parser.add_argument("--rounds", type=int, default=15, help="FedAvg communication rounds")
    parser.add_argument("--local-epochs", type=int, default=3, help="Local SGD epochs per client round")
    parser.add_argument("--lr", type=float, default=0.08, help="Client learning rate")
    parser.add_argument("--data-dir", default=None, help="Path to hospital data directory")
    parser.add_argument("--output-prefix", default="tradeoff", help="Prefix for output files")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(base_dir, "data")

    print("=" * 70)
    print("  CYBERLEEK -- PRIVACY-UTILITY TRADE-OFF ANALYSIS")
    print("  Person 4: Trade-Off & Evaluation Lead")
    print("=" * 70)
    print(f"[*] Trials per epsilon: {args.trials}")

    analytics_result = sweep_dp_analytics(data_dir, args.trials)
    fedavg_result = sweep_dp_fedavg(data_dir, args.trials, args.rounds, args.local_epochs, args.lr)

    analytics_knee = find_analytics_knee(analytics_result["curve"], mae_threshold_pct=0.03)
    fedavg_knee = find_fedavg_knee(fedavg_result["curve"], fedavg_result["non_dp_ceiling"]["roc_auc"],
                                    auc_gap_threshold=0.05)

    plot_path = os.path.join(base_dir, f"{args.output_prefix}_curves.png")
    plot_tradeoff_curves(analytics_result, fedavg_result, plot_path)

    print("\n" + "=" * 70)
    print("  MOHAP POLICY OPERATING POINT RECOMMENDATIONS")
    print("=" * 70)
    print(f"  DP Analytics (population statistics):")
    print(f"    Recommended eps ~= {analytics_knee}  (<0.03% mean relative error on national diabetes prevalence, averaged over {args.trials} noise trials)")
    print(f"  DP-FedAvg (readmission risk model):")
    print(f"    Recommended eps ~= {fedavg_knee}  (mean ROC-AUC within 0.05 of the non-DP FedAvg ceiling, averaged over {args.trials} noise trials)")
    print(f"  NOTE: these are two SEPARATE recommendations for two separate deployments.")
    print(f"        Using the analytics epsilon for the FL model would badly understate its true")
    print(f"        privacy cost to utility, and vice-versa the FL epsilon is far too generous")
    print(f"        (too little noise) for a population-statistics release.")
    print("=" * 70)

    summary = {
        "metadata": {
            "title": "Cyberleek Privacy-Utility Trade-Off Summary",
            "role": "Person 4 - Trade-Off & Evaluation Lead",
            "trials_per_epsilon": args.trials,
        },
        "dp_analytics_track": analytics_result,
        "dp_fedavg_track": fedavg_result,
        "policy_recommendations": {
            "dp_analytics_operating_epsilon": analytics_knee,
            "dp_analytics_rationale": "First epsilon (ascending) at which MAE on national diabetes "
                                      "prevalence falls below 0.03% of the true value (a tiny fraction "
                                      "of a percentage point, well below the sampling noise of any real "
                                      "epidemiological survey).",
            "dp_fedavg_operating_epsilon": fedavg_knee,
            "dp_fedavg_rationale": "First epsilon (ascending) at which mean test ROC-AUC comes within "
                                   "0.05 of the non-DP FedAvg ceiling -- a gap small enough to be "
                                   "operationally indistinguishable for a clinical risk-stratification tool.",
            "note": "These are independent recommendations for two independent deployments; do not "
                    "reuse one track's epsilon for the other.",
        },
    }

    summary_path = os.path.join(base_dir, f"{args.output_prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Trade-off summary saved to: {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
