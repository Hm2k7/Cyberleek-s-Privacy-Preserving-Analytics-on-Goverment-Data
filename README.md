# Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data

**Team Name:** Cyberleek
**Domain:** UAE Healthcare Data Privacy (MOHAP, DOH Abu Dhabi, DHA Dubai)
**Status:** All four workstreams complete, tested, and reproducible end-to-end.

## Brief Description
A demonstrator that computes actionable population health analytics and trains a predictive readmission-risk model across sensitive (simulated) UAE healthcare datasets held by three separate hospitals, while provably limiting what any party — including the central aggregator — can learn about an individual. Powered by mathematically verified **Differential Privacy** and decentralized **Federated Learning**, and validated with two real adversarial attacks that succeed against the naive baseline and collapse under the privacy technique.

## Who Buys This / What It Replaces
See `scenario.md` for the full legal and business case. In short: **MOHAP, DOH Abu Dhabi, and DHA Dubai** need cross-hospital diabetes/hypertension/readmission statistics for national health planning, but UAE PDPL (Federal Decree-Law No. 45/2021) and Health ICT Law No. 2/2019 Art. 13 forbid pooling raw patient records across institutional boundaries. Today they rely on (A) manual quarterly aggregate reporting — slow, error-prone, and still re-identifiable at small cell counts — or (B) an illegal centralized data lake. Cyberleek replaces both with cryptographically-bounded DP releases and FL model training that keep raw records inside each hospital's perimeter.

## Quick Start

```bash
# 1. Set up environment (fresh — no dependencies are pre-installed)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. [Person 1] Generate & validate synthetic UAE hospital data
.venv/bin/python generate_dataset.py
.venv/bin/python validate_dataset.py

# 3. [Person 2] Verify DP mathematical core & budget accounting (11/11 tests)
.venv/bin/python test_privacy.py

# 4. [Person 2] Run naive (non-private) baseline benchmark
.venv/bin/python naive_baseline.py

# 5. [Person 2] Run differentially private analytics engine
.venv/bin/python dp_analytics.py --epsilon 1.0

# 6. [Person 2] Run federated learning: standard FedAvg, then DP-FedAvg
.venv/bin/python federated_learning.py --rounds 15 --local-epochs 3
.venv/bin/python federated_learning.py --rounds 15 --local-epochs 3 --dp --epsilon 2.0

# 7. [Person 3] Run the attack demonstration (real MIA + differencing attack)
.venv/bin/python attack_demo.py

# 8. [Person 4] Run the privacy-utility trade-off sweep (~45s, 20 trials/epsilon)
.venv/bin/python tradeoff_analysis.py --trials 20
```

Every script is independently seeded and deterministic — `generate_dataset.py` reproduces byte-identical CSVs on every run, and each downstream script prints its own pass/fail or verdict summary so a grader can verify each step without reading code.

## Project Structure

```text
.
├── data/                                # Synthetic datasets
│   ├── hospital_a_abu_dhabi.csv         # 8,000 records (older / Emirati skewed)
│   ├── hospital_b_dubai.csv             # 5,000 records (diverse / younger population)
│   ├── hospital_c_rak.csv               # 2,000 records (mixed demographics)
│   └── README.md                        # Dataset documentation & schemas
├── docs/
│   └── dp_mechanism_guide.md            # [Person 2] Mathematical & legal justification
├── generate_dataset.py                  # [Person 1] Synthetic data generator (seedable, deterministic)
├── validate_dataset.py                  # [Person 1] Dataset validation (42/42 checks)
├── scenario.md                          # [Person 1] UAE legal & business problem context
├── naive_baseline.py                    # [Person 2] Non-private centralized baseline (ground truth)
├── naive_baseline_results.json          # [Person 2] Baseline population stats + model metrics
├── dp_core.py                           # [Person 2] DP primitives, sensitivity clipping, budget ledger
├── dp_analytics.py                      # [Person 2] Provable DP analytics CLI & API
├── dp_analytics_results.json            # [Person 2] DP query results & audit log
├── federated_learning.py                # [Person 2] Multi-client FedAvg & DP-FedAvg simulation
├── federated_learning_results.json      # [Person 2] Standard FedAvg results
├── federated_learning_dp_results.json   # [Person 2] DP-FedAvg results + zCDP accounting
├── test_privacy.py                      # [Person 2] Property & statistical verification tests (11/11)
├── attack_demo.py                       # [Person 3] Real membership-inference + differencing attacks
├── attack_results.json                  # [Person 3] Attack ROC-AUC / attacker-advantage / verdicts
├── tradeoff_analysis.py                 # [Person 4] Epsilon sweep across both privacy tracks
├── tradeoff_curves.png                  # [Person 4] Published privacy-utility trade-off plots
├── tradeoff_summary.json                # [Person 4] Full sweep data + MOHAP policy recommendation
├── Cyberleek_Report.pdf                 # 5-page submission report (objective → validation → results)
├── Cyberleek_Report.docx                # Editable source of the submission report
├── requirements.txt                     # Dependencies (numpy, pandas, scipy, scikit-learn, matplotlib)
├── README.md                            # Project overview (this file)
└── HANDOVER.md                          # Full build history, bug-fix log, and API reference
```

## Roles & Responsibilities (all complete)

### Person 1 — Data & Scenario Lead
- **Synthetic Dataset:** 15,000 realistic EHR records split across 3 UAE hospitals with realistic demographic distributions, clinical correlations, and non-identical schemas.
- **Scenario Framing:** UAE MOHAP/DOH/DHA business case under PDPL and Health ICT Law (`scenario.md`).
- **Validation:** `validate_dataset.py`, 42/42 statistical consistency checks.

### Person 2 — Core Privacy Technique Engineer
- **Naive Benchmark:** Non-private ground truth for population statistics and centralized readmission prediction.
- **Differential Privacy Engine:** Laplace & Gaussian mechanisms from first principles, hard sensitivity clipping, cryptographic `PrivacyBudgetTracker` enforcing sequential (Σεᵢ) and parallel (max εᵢ on disjoint hospital silos) composition.
- **Federated Learning:** FedAvg and DP-FedAvg (Gaussian noise calibrated via zCDP composition across all 15 rounds to one stated total (ε, δ) budget) — raw records never leave hospital custody, only weight updates are exchanged.
- **Testing:** `test_privacy.py`, 11/11 tests passing.

### Person 3 — Attack & Vulnerability Engineer
- `attack_demo.py` retrains the real models in-process and runs two real attacks — no numbers are hard-coded:
  1. **Membership Inference (Yeom et al. 2018, loss-threshold):** an unregularized off-the-shelf model leaks membership (attack AUC 0.65); the DP-FedAvg model collapses this to chance (AUC 0.49).
  2. **Differencing/Reconstruction:** naive aggregate counts perfectly reconstruct a target patient's diabetes status (AUC 1.00, attacker advantage 1.00); DP analytics cuts attacker advantage by ~95% at the same per-query ε the analytics engine actually uses, and a live epsilon sweep shows that residual shrinking monotonically as ε decreases. A `PrivacyBudgetTracker` demo also shows a real deployment halts the query campaign after a handful of queries, independent of the noise itself.

### Person 4 — Trade-off & Business Impact Lead
- `tradeoff_analysis.py` sweeps ε across both tracks (20 independent trials per point) and plots MAE-vs-ε for DP analytics alongside ROC-AUC/accuracy-vs-ε for DP-FedAvg (`tradeoff_curves.png`), then derives two **separate** MOHAP operating-point recommendations (see below) — reusing one track's ε for the other would misstate its real privacy cost.

## Concrete Tasks Solved

1. **National & Hospital-Level Diabetes Prevalence:** Computed under pure ε-DP (ε = 1.0) with absolute error <0.03% (single audited run) / <0.13% (mean of 20 noise trials) and full 95% theoretical confidence bounds.
2. **30-Day Hospital Readmission Risk:** Decentralized FedAvg training achieves **58.9% accuracy / 0.639 ROC-AUC** across 3 hospitals with zero raw data sharing, matching the centralized baseline's discriminative power (0.626 ROC-AUC). Under DP-FedAvg (total ε=2.0 over the full run), utility drops substantially (ROC-AUC ≈ 0.48) — see HANDOVER.md §2 for the full trade-off and why the DP-analytics "ε≈1" operating point does not carry over to this model (its own knee sits around ε≈20, per the trade-off sweep).
3. **Attack validation:** both a membership-inference and a differencing/reconstruction attack succeed against the naive, non-private outputs and collapse to near-chance under the corresponding privacy technique — see `attack_results.json` and Section 4 of `Cyberleek_Report.pdf`.

## Grading Rubric Coverage

| Minimum deliverable (from the brief) | Where it's satisfied |
| :--- | :--- |
| Realistic synthetic dataset across 2+ entities, no raw-data sharing | `data/*.csv` (3 hospitals), `generate_dataset.py`, 42/42 checks in `validate_dataset.py` |
| One core technique correctly implemented, stated ε budget | **Both** DP (`dp_core.py`, `dp_analytics.py` — cryptographic `PrivacyBudgetTracker`) and FL (`federated_learning.py` — local training, only weight updates shared), plus DP-FedAvg combining both with zCDP composition |
| Concrete task with a measurable answer | Population statistics (diabetes/hypertension/readmission prevalence, mean HbA1c/BP) **and** a trained classifier (30-day readmission risk, ROC-AUC 0.639) |
| Privacy demonstration: attack succeeds on naive, fails under technique | `attack_demo.py` — MIA (naive 0.65 AUC → DP-FedAvg 0.49 AUC) and differencing attack (naive 1.00 AUC/advantage → DP 0.05 advantage, 95% reduction) |
| Utility/privacy trade-off shown explicitly, assumptions stated | `tradeoff_analysis.py` → `tradeoff_curves.png` / `tradeoff_summary.json`, with separate policy recommendations per track and stated methodology (20 trials/ε, threshold definitions) |

See `Cyberleek_Report.pdf` for the assembled 5-page write-up (objective → proposed solution → validation → results → conclusions & limitations).
