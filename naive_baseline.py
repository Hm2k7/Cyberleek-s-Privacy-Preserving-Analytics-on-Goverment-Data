"""
Naive (Non-Private) Baseline Analytics & Predictive Modeling
Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data

Person 2: Core Privacy Technique Engineer

LEGAL & PRIVACY WARNING:
This script demonstrates the "Current Broken Approach B: Centralized Data Lake".
It loads and pools raw patient records across Hospital A, Hospital B, and Hospital C
into a single centralized memory space. Under UAE Federal Decree-Law No. 45/2021 (PDPL)
and Federal Law No. 2/2019 (Health ICT Law) Article 13, transferring and pooling raw
un-anonymized healthcare records across institutional boundaries without explicit
patient consent and ministerial authorization is strictly prohibited.

This non-private baseline serves as the GROUND TRUTH benchmark against which
Differential Privacy (dp_analytics.py) and Federated Learning (federated_learning.py)
utility loss and accuracy trade-offs are evaluated.

Usage:
    python naive_baseline.py [--output naive_baseline_results.json]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_hospital_data(data_dir: str):
    """Loads datasets from the three simulated UAE hospitals."""
    files = {
        "A": os.path.join(data_dir, "hospital_a_abu_dhabi.csv"),
        "B": os.path.join(data_dir, "hospital_b_dubai.csv"),
        "C": os.path.join(data_dir, "hospital_c_rak.csv"),
    }
    dfs = {}
    for h_code, fpath in files.items():
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing data file: {fpath}. Run generate_dataset.py first.")
        df = pd.read_csv(fpath)
        df["hospital_origin"] = h_code
        dfs[h_code] = df
    return dfs


def compute_population_statistics(df_all: pd.DataFrame, dfs: dict) -> dict:
    """Computes exact non-private national and subgroup population statistics."""
    stats = {}

    # 1. Total counts
    stats["total_records"] = int(len(df_all))
    stats["hospital_counts"] = {h: int(len(df)) for h, df in dfs.items()}

    # 2. National Prevalences
    stats["national_diabetes_prevalence"] = float(df_all["has_diabetes"].mean())
    stats["national_hypertension_prevalence"] = float(df_all["has_hypertension"].mean())
    stats["national_readmission_rate"] = float(df_all["readmitted_30_days"].mean())

    # 3. Hospital-Specific Prevalences
    stats["hospital_diabetes_prevalence"] = {
        h: float(df["has_diabetes"].mean()) for h, df in dfs.items()
    }
    stats["hospital_hypertension_prevalence"] = {
        h: float(df["has_hypertension"].mean()) for h, df in dfs.items()
    }
    stats["hospital_readmission_rate"] = {
        h: float(df["readmitted_30_days"].mean()) for h, df in dfs.items()
    }

    # 4. Demographic Subgroup Statistics (Stratified Prevalence)
    # Age brackets
    df_all["age_group"] = pd.cut(
        df_all["age"],
        bins=[18, 40, 60, 100],
        labels=["<40", "40-60", ">60"],
        right=False,
    )
    stats["diabetes_by_age_group"] = {
        str(k): float(v) for k, v in df_all.groupby("age_group", observed=True)["has_diabetes"].mean().items()
    }
    stats["diabetes_by_gender"] = {
        str(k): float(v) for k, v in df_all.groupby("gender")["has_diabetes"].mean().items()
    }
    stats["diabetes_by_nationality"] = {
        str(k): float(v) for k, v in df_all.groupby("nationality")["has_diabetes"].mean().items()
    }

    # 5. Continuous Clinical Metrics
    stats["mean_age"] = float(df_all["age"].mean())
    stats["std_age"] = float(df_all["age"].std())
    stats["mean_bmi"] = float(df_all["bmi"].mean())
    stats["std_bmi"] = float(df_all["bmi"].std())
    stats["mean_los"] = float(df_all["length_of_stay"].mean())

    # Mean HbA1c (handling missing values cleanly)
    diab_mask = df_all["has_diabetes"] == True
    stats["mean_hba1c_overall"] = float(df_all["hba1c"].dropna().mean())
    stats["mean_hba1c_diabetic"] = float(df_all.loc[diab_mask, "hba1c"].dropna().mean())
    stats["mean_hba1c_nondiabetic"] = float(df_all.loc[~diab_mask, "hba1c"].dropna().mean())

    # Blood Pressure (Systolic)
    hyp_mask = df_all["has_hypertension"] == True
    stats["mean_bp_systolic_overall"] = float(df_all["blood_pressure_systolic"].mean())
    stats["mean_bp_systolic_hypertensive"] = float(df_all.loc[hyp_mask, "blood_pressure_systolic"].mean())
    stats["mean_bp_systolic_normotensive"] = float(df_all.loc[~hyp_mask, "blood_pressure_systolic"].mean())

    return stats


def train_centralized_readmission_model(df_all: pd.DataFrame, random_state: int = 42):
    """
    Trains a centralized, non-private Logistic Regression model to predict 30-day readmission.
    Simulates what an analyst would build if they had unrestricted access to pooled raw patient data.
    """
    numerical_features = [
        "age",
        "bmi",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "hba1c",
        "length_of_stay",
    ]
    categorical_features = ["gender", "nationality", "smoking_status"]
    boolean_features = ["has_diabetes", "has_hypertension", "has_obesity"]

    all_features = numerical_features + categorical_features + boolean_features
    target = "readmitted_30_days"

    X = df_all[all_features].copy()
    y = df_all[target].astype(int).copy()

    # Preprocessing pipelines
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
            ("bool", "passthrough", boolean_features),
        ]
    )

    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs",
                                           class_weight="balanced")),
    ])

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    clf_pipeline.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = clf_pipeline.predict(X_test)
    y_prob = clf_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_positive_prevalence": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Extract learned feature weights for model interpretability
    fitted_preprocessor = clf_pipeline.named_steps["preprocessor"]
    fitted_classifier = clf_pipeline.named_steps["classifier"]

    cat_cols_encoded = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features).tolist()
    feature_names = numerical_features + cat_cols_encoded + boolean_features
    coefficients = {name: float(coef) for name, coef in zip(feature_names, fitted_classifier.coef_[0])}
    metrics["model_coefficients"] = coefficients
    metrics["model_intercept"] = float(fitted_classifier.intercept_[0])

    # Train/test split exposed for downstream membership-inference attack analysis
    # (Person 3): the attacker needs per-sample predictions on both the exact
    # training members and the held-out non-members of THIS model.
    split_data = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}

    return metrics, clf_pipeline, split_data


def main():
    parser = argparse.ArgumentParser(description="Cyberleek Naive Non-Private Baseline Analytics")
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    parser.add_argument("--output", default="naive_baseline_results.json", help="Path to output JSON")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(base_dir, "data")
    output_path = os.path.join(base_dir, args.output)

    print("=" * 65)
    print("  CYBERLEEK — NAIVE (NON-PRIVATE) BASELINE BENCHMARK")
    print("  Core Privacy Technique Engineer: Ground Truth Analytics")
    print("=" * 65)
    print("[!] NOTICE: Centralized data pooling mode (Violates UAE PDPL in production)")

    # 1. Load data
    dfs = load_hospital_data(data_dir)
    df_all = pd.concat(dfs.values(), ignore_index=True)
    print(f"[*] Successfully pooled {len(df_all):,} records from 3 hospitals:")
    for h, df in dfs.items():
        print(f"    - Hospital {h}: {len(df):,} records")

    # 2. Population Statistics
    print("\n" + "-" * 50)
    print("  1. Ground Truth Population Statistics")
    print("-" * 50)
    stats = compute_population_statistics(df_all, dfs)
    print(f"  National Diabetes Prevalence:     {stats['national_diabetes_prevalence']:.2%}")
    print(f"    - Hospital A (Abu Dhabi):        {stats['hospital_diabetes_prevalence']['A']:.2%}")
    print(f"    - Hospital B (Dubai):            {stats['hospital_diabetes_prevalence']['B']:.2%}")
    print(f"    - Hospital C (RAK):              {stats['hospital_diabetes_prevalence']['C']:.2%}")
    print(f"  National Hypertension Rate:       {stats['national_hypertension_prevalence']:.2%}")
    print(f"  National 30-Day Readmission Rate: {stats['national_readmission_rate']:.2%}")
    print(f"  Mean HbA1c (Diabetic):            {stats['mean_hba1c_diabetic']:.2f}%")
    print(f"  Mean HbA1c (Non-Diabetic):        {stats['mean_hba1c_nondiabetic']:.2f}%")
    print(f"  Mean Systolic BP (Hypertensive):  {stats['mean_bp_systolic_hypertensive']:.1f} mmHg")
    print(f"  Mean Systolic BP (Normotensive):  {stats['mean_bp_systolic_normotensive']:.1f} mmHg")

    # 3. Readmission Prediction Model
    print("\n" + "-" * 50)
    print("  2. Centralized Readmission Classifier (Non-Private Logistic Regression)")
    print("-" * 50)
    metrics, _, _ = train_centralized_readmission_model(df_all)
    print(f"  Test Samples:      {metrics['test_samples']:,}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC Score:     {metrics['roc_auc']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1_score']:.4f}")

    # Top predictive risk factors
    top_coeffs = sorted(metrics["model_coefficients"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print("\n  Top 5 Learned Risk Factors (Log-Odds):")
    for feat, weight in top_coeffs:
        direction = "+" if weight > 0 else "-"
        print(f"    {feat:28s}: {direction}{abs(weight):.4f}")

    # 4. Save results to JSON
    benchmark_payload = {
        "metadata": {
            "title": "Cyberleek Naive Non-Private Baseline",
            "role": "Person 2 - Core Privacy Technique Engineer",
            "privacy_guarantee": "NONE (Centralized Raw Data Pooling)",
            "compliance_status": "NON-COMPLIANT (Violates UAE PDPL Art. 1 / Health ICT Law Art. 13)",
            "purpose": "Ground truth reference for measuring DP utility loss and FL accuracy",
        },
        "population_statistics": stats,
        "readmission_predictive_model": metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  [DONE] Baseline benchmark saved to: {os.path.basename(output_path)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
