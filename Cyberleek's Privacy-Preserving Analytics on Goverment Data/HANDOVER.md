# CYBERLEEK PROJECT HANDOVER DOCUMENT
**Project:** Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data  
**Context:** UAE Ministry of Health & Prevention (MOHAP) Healthcare Privacy Demonstrator  
**From:** Person 1 (Data & Scenario Lead) & Person 2 (Core Privacy Technique Engineer)  
**To:** Person 3 (Attack & Vulnerability Engineer) & Person 4 (Trade-Off & Evaluation Lead)  
**Status:** All four workstreams (Person 1–4) are **100% complete, verified, and tested**. `attack_demo.py` (Person 3) and `tradeoff_analysis.py` (Person 4) are implemented, run end-to-end, and their outputs are folded into `Cyberleek_Report.pdf`. This document is kept as the detailed build history and API reference; `README.md` is the up-to-date top-level entry point.

**Revision note (post-review):** An independent end-to-end verification pass found and fixed three real bugs in the original Person 2 deliverables:
1. Both `naive_baseline.py`'s and `federated_learning.py`'s readmission classifiers were **degenerate at the default 0.5 threshold** (precision/recall/F1 = 0.0 — the model never predicted the minority "readmitted" class due to the ~19% class imbalance with no reweighting). Fixed with class-balanced loss weighting (`class_weight="balanced"` in sklearn; an equivalent inverse-frequency sample weighting in the hand-rolled FedAvg SGD).
2. `federated_learning.py --dp` **mis-accounted its privacy budget across communication rounds**: it recalibrated noise using the full stated ε at *every one* of the 15 rounds instead of composing across them, which (a) understated the true cumulative privacy loss and (b) combined with a `clip_norm` default 15–70x larger than real update magnitudes, made the DP-protected model **worse than random guessing** at any ε a normal sweep would test. Fixed with proper zCDP composition across rounds (see §5 below) and a clip norm recalibrated to real update magnitudes.
3. `federated_learning.py`'s `prepare_datasets()` held out a 20% evaluation split from the pooled dataframe but then trained every hospital client on its **full, unfiltered** local dataframe — meaning every "held-out" evaluation record was also a training record, silently inflating reported FedAvg accuracy/ROC-AUC and invalidating any train/non-member comparison. Fixed by excluding each client's evaluation rows before local training (and fitting the feature scaler on the train-only rows). The effect on the reported metrics turned out to be small (ROC-AUC moved from 0.6398 to 0.6393) but the leakage was real and is now closed — this fix is also what makes Person 3's membership-inference attack against the FL models valid (a genuine, disjoint member/non-member split). A `seed` parameter was also added to `run_federated_training()` so DP-FedAvg runs can be repeated with independent noise draws (previously every DP run was 100% deterministic given fixed `round_idx`-seeded RNG, which blocked the multi-trial sweep `tradeoff_analysis.py` needed). All ground-truth figures in this document reflect the corrected code.

---

## 1. Executive Summary & Current Repository State

The foundation for the Cyberleek demonstrator is in place:
1. **Person 1** created the realistic synthetic UAE healthcare dataset (15,000 records across 3 hospitals), legal problem framing (UAE PDPL 45/2021, Health ICT Law 2/2019), and validation suite (42/42 checks passing).
2. **Person 2** built the non-private baseline benchmark, the core Differential Privacy engine (Laplace & Gaussian mechanisms, sensitivity clipping, cryptographic budget tracker), the decentralized Federated Learning (FedAvg) simulation, 11/11 passing mathematical, budget, and DP-FedAvg composition tests, and deep technical/regulatory documentation.

### Repository Layout
```text
Cyberleek's Privacy-Preserving Analytics on Goverment Data/
├── data/
│   ├── hospital_a_abu_dhabi.csv         # 8,000 records (older / Emirati skewed)
│   ├── hospital_b_dubai.csv             # 5,000 records (diverse / younger cohort)
│   ├── hospital_c_rak.csv               # 2,000 records (mixed demographics)
│   └── README.md                        # Complete schema definitions
├── docs/
│   └── dp_mechanism_guide.md            # Person 2 technical depth & legal report
├── generate_dataset.py                  # Person 1 data generator (seedable)
├── validate_dataset.py                  # Person 1 data validator (42 tests)
├── scenario.md                          # Person 1 UAE business & legal case
├── naive_baseline.py                    # Person 2 centralized non-private benchmark
├── naive_baseline_results.json          # Person 2 baseline results & model weights
├── dp_core.py                           # Person 2 DP primitives & budget ledger
├── dp_analytics.py                      # Person 2 DP analytics engine & CLI
├── dp_analytics_results.json            # Person 2 DP query results & audit log
├── federated_learning.py                # Person 2 multi-client FedAvg simulation
├── federated_learning_results.json      # Person 2 standard FedAvg training logs & comparison
├── federated_learning_dp_results.json   # Person 2 DP-FedAvg training logs & zCDP accounting
├── test_privacy.py                      # Person 2 test suite (11/11 passing)
├── requirements.txt                     # numpy, pandas, scipy, scikit-learn, matplotlib
├── README.md                            # Comprehensive project overview
└── HANDOVER.md                          # This document
```

---

## 2. Ground Truth & Baseline Metrics Reference

When evaluating attacks or trade-offs, compare against these certified ground truth figures:

| Metric | Ground Truth (Naive Centralized) | DP Estimate ($\epsilon=1.0$) | FedAvg Model (Decentralized) |
| :--- | :---: | :---: | :---: |
| **Total Cohort Size** | 15,000 patients | 15,000 patients | 12,000 train / 3,000 test |
| **National Diabetes Prev** | **$26.63\%$** | **$26.66\%$** (Error: $0.026\%$) | N/A |
| **Hospital A Diabetes** | $30.18\%$ | $30.17\%$ | Local node custody |
| **Hospital B Diabetes** | $20.90\%$ | $21.03\%$ | Local node custody |
| **Hospital C Diabetes** | $26.80\%$ | $26.93\%$ | Local node custody |
| **National Hypertension** | $31.82\%$ | $31.75\%$ | N/A |
| **National Readmission** | $19.11\%$ | $19.25\%$ | N/A |
| **Mean HbA1c (Diabetic)** | $7.80\%$ | $7.80\%$ | N/A |
| **Mean HbA1c (Non-Diab)** | $5.40\%$ | $5.38\%$ | N/A |
| **Readmission Accuracy** | **$61.20\%$** | N/A | **$58.87\%$** (Delta: $-2.33$ pp) |
| **Readmission ROC-AUC** | **$0.6256$** | N/A | **$0.6393$** |
| **Readmission Precision** | $0.2634$ | N/A | $0.2708$ |
| **Readmission Recall** | $0.5742$ | N/A | $0.6303$ |
| **Readmission F1** | $0.3611$ | N/A | $0.3788$ |

Both classifiers use `class_weight="balanced"` (or an equivalent inverse-frequency
sample weighting in the federated SGD) because the raw readmission label is only
~19% positive; without it, accuracy alone looks better (~81%) but the model never
predicts the minority class at all (precision/recall/F1 = 0), which is useless for
the risk-prediction use case and for Person 3's planned attack work. ROC-AUC (the
threshold-independent metric) is essentially unchanged by this fix, as expected.

**DP-FedAvg** (`federated_learning.py --dp --epsilon 2.0`, total privacy budget for
the *entire* 15-round run — see §5): Accuracy $42.07\%$, ROC-AUC $0.4807$, Precision
$0.1899$, Recall $0.5882$, F1 $0.2871$. This is a real, steep privacy cost — DP-FedAvg
utility only approaches the non-DP FedAvg ceiling (ROC-AUC $\approx 0.64$) around
$\epsilon \approx 50$–$100$; at "small" epsilons (0.01–2.0) it stays near or below
random-guessing (ROC-AUC $\approx 0.45$–$0.48$). **Person 4: do not assume the
$\epsilon=1.0$ "knee" claimed for the DP analytics track (population statistics)
also applies to the DP-FedAvg model — it does not.** The FL classifier's Pareto knee
sits much further out (empirically $\epsilon \approx 5$–$20$); report the FL curve
separately from the DP-analytics curve, and pick a policy operating point using
whichever curve is actually being deployed.

---

## 3. Ready-to-Use APIs for Person 3 & Person 4

Person 2 has exposed clean, modular programmatic interfaces ready to import:

### A. Importing DP Primitives & Budget Ledger (`dp_core.py`)
```python
from dp_core import (
    PrivacyBudgetTracker,
    laplace_mechanism,
    laplace_confidence_interval,
    gaussian_mechanism,
    clip_values,
    dp_proportion,
    dp_mean,
    dp_count,
    PrivacyBudgetExhaustedError,
)

# Example: Track privacy expenditure
tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
noisy_prev, ci_95 = dp_proportion(
    raw_numerator_count=3995, 
    total_population=15000, 
    epsilon=0.2, 
    tracker=tracker, 
    query_name="national_diabetes"
)
```

### B. Running DP Analytics Programmatically (`dp_analytics.py`)
```python
from dp_analytics import load_hospital_silos, run_dp_analytics

silos = load_hospital_silos("data")
results = run_dp_analytics(silos, total_epsilon=0.5, seed=42)

# Access queries and error metrics
nat_diab_est = results["queries"]["national_diabetes_prevalence"]["dp_estimate"]
nat_diab_err = results["queries"]["national_diabetes_prevalence"]["absolute_error"]
```

### C. Running Federated Learning Programmatically (`federated_learning.py`)
```python
from federated_learning import prepare_datasets, run_federated_training

clients, X_test, y_test, feature_cols = prepare_datasets("data")

# Run standard FedAvg
fed_res = run_federated_training(
    clients=clients, X_test=X_test, y_test=y_test, 
    feature_cols=feature_cols, rounds=15, local_epochs=3, lr=0.08
)

# Run Differentially Private FedAvg (DP-FedAvg).
# `epsilon` is the TOTAL (epsilon, delta)-DP budget for the entire `rounds`-round run --
# the per-round Gaussian noise is calibrated via zCDP composition (see dp_accounting
# in the returned dict) so all `rounds` releases together satisfy exactly this budget.
dp_fed_res = run_federated_training(
    clients=clients, X_test=X_test, y_test=y_test, 
    feature_cols=feature_cols, rounds=15, local_epochs=3, lr=0.08,
    enable_dp=True, epsilon=1.0
)
```

### D. Reading Stored Baseline Benchmark (`naive_baseline_results.json`)
```python
import json

with open("naive_baseline_results.json", "r") as f:
    baseline = json.load(f)

true_stats = baseline["population_statistics"]
naive_model = baseline["readmission_predictive_model"]
coefficients = naive_model["model_coefficients"]
```

---

## 4. Specific Action Plan for Person 3 — Attack & Vulnerability Engineer

**✅ COMPLETED.** `attack_demo.py` retrains the real models in-process (no hard-coded numbers) and runs both attacks below. Headline results: MIA attack AUC 0.65 (naive, unregularized model) → 0.49 (DP-FedAvg, chance level); differencing attack attacker-advantage 1.00 (naive, perfect reconstruction) → 0.05 (DP analytics, a 95% reduction), with a live ε-sweep confirming the residual advantage shrinks monotonically as ε decreases. Full numbers in `attack_results.json` and `Cyberleek_Report.pdf` §4.1.

**Brief Goal:** Implement `attack_demo.py` to demonstrate that naive non-private systems leak individual patient records, and prove that Differential Privacy successfully defends against the attack.

### Recommended Implementation Roadmap:
1. **Membership Inference Attack (MIA) against Predictive Model:**
   - **Target A (Vulnerable):** Centralized non-private logistic regression model (`naive_baseline.py`).
     - Extract member samples (from train set) and non-member samples (from test set).
     - Calculate loss / prediction confidence: $\mathcal{L}(x, y) = - [y \log \hat{p} + (1-y) \log (1-\hat{p})]$.
     - Show that the non-private model fits members more closely than non-members $\implies$ attacker threshold achieves **high Attack ROC-AUC ($>0.65-0.75$)** and high precision.
   - **Target B (Protected):** DP-FedAvg model (`federated_learning.py --dp --epsilon 1.0`).
     - Run the same attack on the DP-protected model.
     - Show that the membership attack success drops to **$\approx 50.0\%$ (random guessing)**, proving mathematical membership privacy.

2. **Differencing / Reconstruction Attack against Aggregate Queries:**
   - Demonstrate a targeted differencing query:
     - Query 1: Number of diabetic patients in Hospital A meeting specific criteria (e.g. Emirati females aged 62).
     - Query 2: Same query with target individual $x^*$ omitted.
     - In a naive system without DP, $(\text{Query 1} - \text{Query 2}) \in \{0, 1\}$ completely reveals $x^*$'s diagnosis.
     - Under `dp_analytics.py`, the Laplace noise ($\pm \frac{1}{\epsilon}$) swamps the individual signal, rendering the subtraction useless to the attacker.

3. **Deliverables Expected:**
   - `attack_demo.py`: Runnable CLI with clean tables showing:
     - Attack Type
     - Target (Naive Centralized vs DP-Protected)
     - Attack Success Rate (ASR)
     - Attack ROC-AUC
     - Privacy Protection Verdict
   - `attack_results.json`: Serialized attack metrics.

---

## 5. Specific Action Plan for Person 4 — Trade-Off & Evaluation Lead

**✅ COMPLETED.** `tradeoff_analysis.py` sweeps both tracks (20 independent trials per ε) and produces `tradeoff_curves.png` / `tradeoff_summary.json`. Measured (not assumed) recommendations: DP analytics ε≈5 (<0.03% mean error on national diabetes prevalence across trials; ε=1.0 already gives <0.13% mean error / 0.026% on the single audited run), DP-FedAvg ε≈20 (mean ROC-AUC within 0.05 of the non-DP ceiling) — both measured directly rather than estimated, and confirming the two tracks need genuinely different budgets, as flagged below.

**Brief Goal:** Implement `tradeoff_analysis.py` to evaluate and visualize privacy-utility trade-off curves, providing actionable policy recommendations for UAE health authorities (MOHAP).

### Recommended Implementation Roadmap:
1. **Privacy-Utility Trade-Off Curve Generation:**
   - Loop over an $\epsilon$ grid: $\epsilon \in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]$.
   - For each $\epsilon$:
     - Run `run_dp_analytics(silos, total_epsilon=eps, seed=s)` across multiple seeds ($N=20$ trials).
     - Compute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) on National Diabetes Prevalence.
     - Run `run_federated_training(..., enable_dp=True, epsilon=eps)` and record test ROC-AUC / Accuracy.
2. **Generate Publication-Grade Plots:**
   - Use `matplotlib`:
     - Plot 1: $\epsilon$ (x-axis, log scale) vs. National Diabetes Absolute Error (%) (y-axis), with $95\%$ theoretical confidence error bands.
     - Plot 2: $\epsilon$ vs. Readmission Risk Model ROC-AUC / Accuracy.
     - Save figures as `tradeoff_curves.png`.
3. **MOHAP Policy Operating Point Recommendation:**
   - Identify the "knee" of the Pareto curve **separately for each track** — they are not
     the same $\epsilon$. For DP analytics (population statistics), the knee is around
     $\epsilon \approx 1.0$ ($<0.1\%$ error on diabetes prevalence, see §2 above). For
     DP-FedAvg (the predictive model), the knee is much further out, empirically around
     $\epsilon \approx 5$–$20$ (see §2's DP-FedAvg note) — using $\epsilon=1.0$ as the
     "recommended operating point" for the FL model would badly understate its true
     privacy cost to utility.
   - Conclude that at $\epsilon = 1.0$, MOHAP's DP *analytics* obtain $<0.1\%$ error on
     diabetes prevalence and $>98\%$ relative statistical utility. State the FL model's
     trade-off as a separate, distinct recommendation with its own operating point.
4. **Deliverables Expected:**
   - `tradeoff_analysis.py`: Executable analysis script.
   - `tradeoff_curves.png`: Exported charts.
   - `tradeoff_summary.json`: Serialized sweep results.

---

## 6. How to Verify All Components Right Now

From the directory `Cyberleek's Privacy-Preserving Analytics on Goverment Data/`:

```powershell
# 1. Run unit test suite (11/11 tests pass)
python test_privacy.py

# 2. Run naive baseline benchmark
python naive_baseline.py

# 3. Run differentially private analytics engine
python dp_analytics.py --epsilon 1.0

# 4. Run federated learning simulation (standard) -> federated_learning_results.json
python federated_learning.py --rounds 15 --local-epochs 3

# 5. Run federated learning simulation (DP-FedAvg) -> federated_learning_dp_results.json
#    epsilon is the TOTAL privacy budget for the whole 15-round run (see §5), not per-round.
python federated_learning.py --rounds 15 --local-epochs 3 --dp --epsilon 2.0

# 6. Run the attack demonstration (Person 3) -> attack_results.json
python attack_demo.py

# 7. Run the privacy-utility trade-off sweep (Person 4, ~45s) -> tradeoff_curves.png / tradeoff_summary.json
python tradeoff_analysis.py --trials 20
```

Note: standard and DP-FedAvg runs now write to **different default output files**
(`federated_learning_results.json` vs. `federated_learning_dp_results.json`) so that
running one mode no longer silently overwrites the other's saved results. Both are
committed in the repo.

---

## 7. Environment & Compatibility Notes

- **Python Version:** 3.10+ (tested on Python 3.13 on Windows 64-bit).
- **Encoding:** All scripts include `sys.stdout.reconfigure(encoding="utf-8")` to guarantee flawless execution in Windows PowerShell and Unix terminals.
- **Dependencies:** Standard libraries only (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`). No obscure external dependencies.
- **Seeding:** All randomized operations support explicit `--seed` arguments for deterministic grading and reproducible benchmarking.

**Cyberleek is complete: all 4 roles delivered, tested, and reproducible end-to-end.** See `README.md` for the current top-level summary and `Cyberleek_Report.pdf` for the assembled submission write-up.
