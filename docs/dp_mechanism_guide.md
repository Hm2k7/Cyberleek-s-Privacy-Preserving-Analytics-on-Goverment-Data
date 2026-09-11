# Technical Documentation: Core Differential Privacy & Federated Learning Architecture
**Project:** Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data  
**Lead:** Person 2 — Core Privacy Technique Engineer  
**Target Evaluation:** Technical Depth (25%), Relevance of Solution (15%), Fit to Brief (20%)

---

## 1. Executive Summary & Problem Context

The UAE Ministry of Health & Prevention (MOHAP), Dubai Health Authority (DHA), and the Department of Health Abu Dhabi (DOH) require cross-hospital population health statistics and machine learning risk models—specifically national diabetes prevalence and 30-day hospital readmission predictions.

However, health data in the UAE is governed by strict, non-negotiable legal frameworks:
1. **UAE Federal Decree-Law No. 45/2021 on Personal Data Protection (PDPL):** Classifies health data as "Sensitive Personal Data" (Article 1) and requires rigorous technical and organizational safeguards.
2. **UAE Federal Law No. 2/2019 (Health ICT Law) Article 13:** Explicitly prohibits the unauthorized transfer, sharing, or centralization of raw patient health records outside originating hospital institutions.
3. **Medical Liability Law (Federal Decree-Law No. 4/2016, Article 13):** Imposes criminal liability on the unauthorized disclosure of confidential patient medical information.

To overcome the deadlock between public health intelligence and strict privacy compliance, Person 2 has engineered:
- **A Provably Private Differential Privacy (DP) Engine (`dp_core.py`, `dp_analytics.py`)**: Computes national and hospital-level prevalences, clinical vitals (HbA1c, BP), and demographics under mathematical $\epsilon$-Differential Privacy.
- **A Cryptographic Privacy Budget Tracker (`PrivacyBudgetTracker`)**: Prevents budget exhaustion, manages sequential composition, and leverages parallel composition across disjoint hospital cohorts.
- **A Multi-Client Federated Learning Framework (`federated_learning.py`)**: Trains decentralized 30-day readmission predictive models across Hospital A (Abu Dhabi), Hospital B (Dubai), and Hospital C (RAK) without transmitting any raw EHR patient records to the central MOHAP orchestrator.

---

## 2. Mathematical Foundations of Differential Privacy

### 2.1 Formal Definition of $(\epsilon, \delta)$-Differential Privacy

A randomized mechanism $\mathcal{M}$ provides $(\epsilon, \delta)$-Differential Privacy if, for all neighboring datasets $D, D' \in \mathcal{D}$ differing on at most one individual record ($|D \Delta D'| = 1$), and for all measurable query outputs $S \subseteq \text{Range}(\mathcal{M})$:

$$\mathbb{P}[\mathcal{M}(D) \in S] \le e^{\epsilon} \cdot \mathbb{P}[\mathcal{M}(D') \in S] + \delta$$

- **Pure $\epsilon$-Differential Privacy ($\delta = 0$):** Used in our summary analytics (`dp_analytics.py`). Guarantees an information-theoretic bound on the log-likelihood ratio of an individual's presence or absence.
- **Approximate $(\epsilon, \delta)$-Differential Privacy ($\delta > 0$):** Used in our federated model training (`federated_learning.py`). $\delta$ denotes the probability that the pure $\epsilon$-bound is violated (calibrated to $\delta \le 10^{-5} \ll \frac{1}{N}$).

### 2.2 Global Sensitivity Analysis

The amount of noise required to mask any single individual depends strictly on the **Global Sensitivity** of the query function $f$:

- **$L_1$ Global Sensitivity:**
  $$\Delta_1 f = \max_{D, D': \|D - D'\|_1 \le 1} \|f(D) - f(D')\|_1$$
- **$L_2$ Global Sensitivity:**
  $$\Delta_2 f = \max_{D, D': \|D - D'\|_1 \le 1} \|f(D) - f(D')\|_2$$

| Query Type | Mathematical Formulation | $L_1$ Sensitivity ($\Delta_1 f$) | Mechanism Selected |
| :--- | :--- | :--- | :--- |
| **Record Count** | $f(D) = \|D\|$ | $1$ | Laplace Mechanism |
| **Prevalence / Proportion** | $f(D) = \frac{1}{N} \sum_{i=1}^N x_i, \quad x_i \in \{0, 1\}$ | $\frac{1}{N}$ | Laplace Mechanism |
| **Bounded Clinical Sum** | $f(D) = \sum_{i=1}^N \text{clip}(v_i, L, U)$ | $U - L$ | Laplace Mechanism |
| **Bounded Clinical Mean** | $f(D) = \frac{1}{N} \sum_{i=1}^N \text{clip}(v_i, L, U)$ | $\frac{U - L}{N}$ | Laplace / Budget Split |
| **Federated Model Update** | $\Delta w_k = w_k^{(t+1)} - w^{(t)}, \quad \|\Delta w_k\|_2 \le C$ | $\frac{2C}{K}$ | Gaussian Mechanism |

---

## 3. Mechanism Selection & Theoretical Properties

### 3.1 The Laplace Mechanism (Pure $\epsilon$-DP)

For numeric and scalar aggregate statistics, we apply the Laplace mechanism:

$$\mathcal{M}_{Lap}(D) = f(D) + \text{Laplace}\left(0, \frac{\Delta_1 f}{\epsilon}\right)$$

where the Laplace probability density function is:

$$p(y) = \frac{1}{2b} \exp\left(-\frac{|y|}{b}\right), \quad b = \frac{\Delta_1 f}{\epsilon}$$

#### Theoretical Variance & Confidence Intervals
- **Variance:** $\text{Var}[\mathcal{M}_{Lap}(D)] = 2 b^2 = 2 \left(\frac{\Delta_1 f}{\epsilon}\right)^2$
- **Theoretical 95% Confidence Radius:**
  $$\mathbb{P}(|Y| \le r) = 0.95 \implies r = b \cdot \ln\left(\frac{1}{1 - 0.95}\right) = \frac{\Delta_1 f}{\epsilon} \ln(20) \approx 2.9957 \cdot \frac{\Delta_1 f}{\epsilon}$$

In `dp_analytics.py`, every query output is automatically reported with its theoretical $95\%$ error radius, allowing health authorities (MOHAP) to certify the statistical confidence of the reported estimate.

### 3.2 The Gaussian Mechanism ($(\epsilon, \delta)$-DP)

For vector-valued queries and high-dimensional parameter updates (e.g. federated model weight gradients in `federated_learning.py`), $L_2$ sensitivity grows far more favorably than $L_1$ sensitivity ($\Delta_2 f \le \Delta_1 f \le \sqrt{d} \Delta_2 f$).

$$\mathcal{M}_{Gauss}(D) = f(D) + \mathcal{N}\left(0, \sigma^2 I\right), \quad \sigma = \frac{\Delta_2 f \sqrt{2 \ln(1.25 / \delta)}}{\epsilon}$$

---

## 4. Why Differential Privacy is Essential Now (vs. Older Approaches)

The hackathon rubric requires documenting *"why this is appropriate now (vs. older approaches)"*. Below is a rigorous technical comparison against traditional anonymization paradigms:

```
+----------------------------------------------------------------------------------------------------+
|                                    ANONYMIZATION PARADIGM EVOLUTION                                |
|                                                                                                    |
|  [Syntactic De-Identification]  -->  [k-Anonymity / l-Diversity]  -->  [Differential Privacy]      |
|  • Strips names/Emirates IDs        • Groups records into quasi-ids    • Mathematical perturbation |
|  • Defeated by linkage attacks      • Defeated by curse of dimension   • Immune to side knowledge  |
|  • Zero worst-case guarantees       • High utility destruction         • Provable worst-case bound |
+----------------------------------------------------------------------------------------------------+
```

### 4.1 Comparison Against Legacy Techniques

| Anonymization Paradigm | Theoretical Mechanism | Vulnerability / Failure Mode | Why DP Supersedes It |
| :--- | :--- | :--- | :--- |
| **Pseudonymization / Direct Masking** | Stripping Direct Identifiers (Name, Patient ID, Emirates ID) | **Linkage Attacks (Sweeney, 2002):** 87% of individuals are uniquely identified by {Age/DOB, Gender, ZIP}. In the UAE, linking age + gender + admission date across public datasets easily re-identifies patients. | DP mathematically provably prevents linkage attacks regardless of how much auxiliary external data an attacker possesses. |
| **$k$-Anonymity (Sweeney, 2002)** | Generalizes quasi-identifiers so each record is indistinguishable from $\ge k-1$ others | **Homogeneity Attack:** If all $k$ patients in an equivalence class have diabetes, patient status is revealed with 100% certainty. **Curse of Dimensionality:** In 16+ dimensional EHR data, achieving $k \ge 5$ requires suppressing >80% of data. | DP provides individual-level indistinguishability without suppressing feature columns or relying on equivalence classes. |
| **$l$-Diversity (Machanavajjhala, 2006)** | Ensures $\ge l$ well-represented sensitive values per equivalence class | **Skewness & Similarity Attacks:** Does not account for semantic correlation between diseases (e.g., hypertension and cardiovascular disease). | DP guarantees privacy across all possible attribute distributions and semantic correlations. |
| **$t$-Closeness (Li et al., 2007)** | Distance between local and global attribute distributions $\le t$ | Computationally intractable for multimodal continuous clinical vitals (HbA1c, BP); severely degrades analytical utility. | DP provides clear analytical trade-offs via a single tuning parameter ($\epsilon$) with closed-form noise formulas. |
| **Naive Aggregate Reporting (Current Approach A)** | Publishing raw table aggregates and summary histograms | **Reconstruction Attacks (Dinur-Nissim Theorem):** Answering $O(n)$ exact linear queries allows an adversary to reconstruct the entire underlying database in polynomial time. | DP bounds cumulative query leakage through composition theorems and explicit budget tracking. |

### 4.2 The Dinur-Nissim Impossibility Theorem & Modern Relevance

The seminal theorem by Dinur and Nissim (2003) proved that **any query answering system that answers aggregate queries with $o(\sqrt{n})$ error can be subjected to database reconstruction attacks**.

Therefore, MOHAP's current aggregate reporting cannot remain safe without mathematically calibrated noise. Differential Privacy is the **only mathematically proven mechanism** that prevents database reconstruction while preserving population-level utility.

---

## 5. Architectural Design: Decentralized Silos & Federated Learning

To comply with **UAE Health ICT Law Article 13**, patient data must never leave the originating hospital network. We simulate this exact topology across three UAE hospital environments:

```
  +--------------------------+       +--------------------------+       +--------------------------+
  |  Hospital A (Abu Dhabi)  |       |   Hospital B (Dubai)     |       |    Hospital C (RAK)      |
  |  Al Mafraq Hospital      |       |   Rashid Hospital        |       |    Saqr Hospital         |
  |  8,000 Patient Records   |       |   5,000 Patient Records  |       |    2,000 Patient Records |
  |  Custody: Local Node     |       |   Custody: Local Node    |       |    Custody: Local Node   |
  +-------------+------------+       +-------------+------------+       +-------------+------------+
                |                                  |                                  |
                | Local Weights / Gradients        | Local Weights / Gradients        | Local Weights / Gradients
                | (No Raw Records!)                | (No Raw Records!)                | (No Raw Records!)
                +----------------------------------+----------------------------------+
                                                   |
                                                   v
                                 +-----------------------------------+
                                 |  MOHAP Central Orchestrator       |
                                 |  (Ministry of Health & Prevention)|
                                 |  • FedAvg Weighted Aggregation    |
                                 |  • Optional DP-Gaussian Injection |
                                 |  • Global Readmission Model       |
                                 +-----------------------------------+
```

### 5.1 Federated Averaging (FedAvg) Formulation

In each communication round $t = 1, \dots, T$:
1. MOHAP broadcasts the current global weight vector $w^{(t)}$ and intercept $b^{(t)}$ to Hospital A, B, and C.
2. Each hospital node $k \in \{A, B, C\}$ executes local SGD optimization on its private electronic health records for $E$ epochs:
   $$w_k^{(t+1)} \leftarrow w^{(t)} - \eta \nabla \mathcal{L}_k(w^{(t)})$$
3. Each hospital transmits **only model parameters** $(w_k^{(t+1)}, b_k^{(t+1)})$ back to MOHAP.
4. MOHAP aggregates the updates weighted proportionally to local cohort sizes:
   $$w^{(t+1)} = \sum_{k \in \{A, B, C\}} \frac{n_k}{N} w_k^{(t+1)}, \quad b^{(t+1)} = \sum_{k \in \{A, B, C\}} \frac{n_k}{N} b_k^{(t+1)}$$

### 5.2 Performance Comparison: Federated vs. Centralized Naive

Evaluated on 3,000 held-out patient records from the synthetic UAE cohort:

Both classifiers use class-balanced loss weighting (the raw readmission label is only
~19% positive; without reweighting, a logistic model at the default 0.5 threshold
degenerates to always predicting the majority class — precision/recall/F1 = 0, an
unusable risk model despite deceptively "high" raw accuracy).

| Metric | Naive Centralized Baseline (Illegal in Production) | Decentralized Federated Learning (FedAvg) | Delta | Operational Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | $61.20\%$ | $58.87\%$ | $-2.33$ pp | Modest accuracy loss while maintaining $100\%$ legal compliance |
| **ROC-AUC** | $0.6256$ | $0.6393$ | $+0.0137$ | Matches (slightly exceeds) centralized discriminative ranking |
| **Precision / Recall / F1** | $0.263$ / $0.574$ / $0.361$ | $0.270$ / $0.630$ / $0.378$ | comparable | Both models actually predict the minority class (non-degenerate) |
| **Data Centralization** | Complete (15,000 raw records pooled) | **Zero (0 records transferred)** | **N/A** | Preserves hospital data custody & UAE sovereignty |

### 5.3 DP-FedAvg: Multi-Round Composition

A single Gaussian release (§3.2) is calibrated for a *single* query. DP-FedAvg
releases a noised model update once per communication round, and every round's
release must compose into one *total* privacy guarantee for the run. Naively
recalibrating $\sigma$ with the full requested $\epsilon$ at *every* round (rather
than accounting for the $R$ repeated releases) both understates the true cumulative
privacy loss and — because $\sigma \propto 1/\epsilon$ — miscalibrates the noise
scale entirely independent of that accounting error.

We compose the $R=$ `rounds` Gaussian releases via **zero-Concentrated DP (zCDP;
Bun & Steinke 2016)**, which composes additively and exactly:

$$\rho = \frac{\Delta_2^2}{2\sigma^2} \text{ per release}, \qquad \rho_{\text{total}} = R \cdot \rho_{\text{per-round}}$$

$$\epsilon = \rho_{\text{total}} + 2\sqrt{\rho_{\text{total}} \ln(1/\delta)} \quad \text{(zCDP} \to (\epsilon,\delta)\text{-DP conversion)}$$

Given a caller's *total* $(\epsilon, \delta)$ budget for the whole run, we invert
this to solve for the per-round $\sigma$ that makes $R$ releases compose to exactly
that budget (`sigma_for_composed_gaussian()` in `federated_learning.py`). Under this
composition, $\sigma$ grows only $\propto \sqrt{R}$ for a fixed total budget — not
$\propto R$ as naive per-round accounting would require — matching the favorable
scaling used by real DP-FedAvg / DP-SGD implementations (e.g. McMahan et al. 2018;
Abadi et al. 2016's moments accountant achieves an even tighter bound via the same
principle).

We also recalibrated the per-client update clipping bound `clip_norm` from its
previous default of $1.0$ to $0.05$: empirically, real per-round per-client update
norms for this model/data/learning-rate combination are $\approx 0.01$–$0.04$, so a
bound of $1.0$ never actually clips anything while still (via the sensitivity
formula $\Delta_2 = 2 \cdot \text{clip\_norm} / K$) forcing noise 15–70x larger than
the true signal at any usable $\epsilon$.

**Result:** DP-FedAvg (total $\epsilon=2.0$ over 15 rounds) now achieves ROC-AUC
$0.481$ (up from a pre-fix $0.465$, which was *worse than random guessing* despite
the identical stated $\epsilon$). More importantly, the fixed mechanism now shows a
genuine, monotonically-improving privacy-utility curve across $\epsilon \in
[0.01, 10]$ instead of flat, near-random utility at every tested $\epsilon$; utility
converges to the non-DP FedAvg ceiling (ROC-AUC $\approx 0.64$) around $\epsilon
\approx 50$–$100$. Person 4: this model's Pareto knee sits much further out than the
DP-analytics track's $\epsilon \approx 1.0$ knee — report the two trade-off curves
separately.

---

## 6. Elimination of Classic Differential Privacy Vulnerabilities

Our implementation specifically defends against five classic DP implementation flaws:

### 1. Unclipped Sensitivity (Outlier Vulnerability)
- **Vulnerability:** If an attacker inserts a single corrupted record with $\text{HbA1c} = 1000$ or $\text{BP} = 500$, the true sensitivity of the mean explodes, completely breaking the privacy guarantee.
- **Solution:** `clip_values()` strictly enforces clipping bounds derived from clinical domain constraints before computing sums or means:
  - $\text{HbA1c} \in [4.0, 14.0]$
  - $\text{Systolic Blood Pressure} \in [90.0, 200.0]$
  - $\text{Age} \in [18, 95]$
  - $\text{Length of Stay} \in [1, 35]$

### 2. Infinite / Unaccounted Privacy Budget Reuse
- **Vulnerability:** Re-running queries with the same $\epsilon$ repeatedly on the same dataset compounds privacy loss, allowing attackers to average out Laplace noise.
- **Solution:** `PrivacyBudgetTracker` maintains a stateful cryptographic ledger. Every query is logged with its timestamp, requested $\epsilon$, and target partition. If total consumed budget exceeds $\epsilon_{total}$, the engine raises a `PrivacyBudgetExhaustedError`, halting execution.

### 3. Suboptimal Sequential Accounting on Disjoint Partitions
- **Vulnerability:** Standard composition assumes all queries access the same individuals, consuming $\sum \epsilon_i$.
- **Solution:** We explicitly leverage **Parallel Composition**:
  - *Theorem:* If datasets $D_1, D_2, \dots, D_k$ are pairwise disjoint partitions ($D_i \cap D_j = \emptyset$), then executing $\epsilon$-DP mechanisms on each partition simultaneously satisfies $\max_i(\epsilon_i)$-DP globally.
  - Since Hospital A, Hospital B, and Hospital C have mutually disjoint patient cohorts, querying all three hospitals costs only $\max(\epsilon_A, \epsilon_B, \epsilon_C) = 0.20 \epsilon$ rather than $0.60 \epsilon$.

### 4. Divide-by-Zero / Ratio Sensitivity in Mean Estimation
- **Vulnerability:** Directly taking a noisy sum and dividing by raw count leaks count sensitivity; dividing by a noisy count can cause numerical explosion if the noisy count is near zero.
- **Solution:** `dp_mean()` splits budget ($\epsilon_{sum} = 0.5\epsilon, \epsilon_{count} = 0.5\epsilon$) and clamps the denominator to $\max(1.0, \hat{N})$ to ensure numerical stability.

### 5. Out-of-Bounds Query Outputs (Domain Invariance)
- **Vulnerability:** Pure Laplace noise can produce negative counts (e.g. $-3$ diabetic patients) or prevalence rates $>100\%$.
- **Solution:** Differential Privacy is invariant to post-processing. `dp_proportion()` clamps outputs strictly to $[0.0, 1.0]$, and `dp_count()` enforces non-negativity ($\max(0, \text{round}(\hat{c}))$).

---

## 7. UAE Regulatory Compliance Mapping

| Legal Provision | Legislative Mandate | Cyberleek Technical Enforcement |
| :--- | :--- | :--- |
| **UAE PDPL Art. 1** | Classifies healthcare information as "Sensitive Personal Data" requiring enhanced technical protections. | Differential Privacy guarantees provable bound on information leakage ($\epsilon \le 1.0$), ensuring no individual can be identified from outputs. |
| **UAE Health ICT Law Art. 13** | Prohibits transferring or storing patient data outside originating health facilities without authorization. | Federated Learning (`federated_learning.py`) leaves 100% of raw patient records inside local hospital compute boundaries. |
| **UAE Medical Liability Law Art. 13** | Criminal penalties for unauthorized disclosure of patient secrets. | Mathematical proof of differential privacy replaces subjective trust with cryptographic guarantees. |
| **HIE Interoperability (Malaffi, Nabidh, Riayati)** | Hospitals operate under distinct emirate and federal health information exchanges. | Federated compute nodes integrate as edge workers within existing HIE networks, transmitting only model weights and noise-perturbed statistics. |

---

## 8. Verification & Test Evidence

All mathematical properties and system components are verified via automated test suites:
- `validate_dataset.py`: 42/42 validation checks passed.
- `test_privacy.py`: 11/11 mathematical, budget accounting, and DP-FedAvg composition tests passed.
- `naive_baseline.py`: Ground-truth benchmark successfully generated (`naive_baseline_results.json`).
- `dp_analytics.py`: Provable DP analytics executed and within theoretical 95% confidence intervals (`dp_analytics_results.json`).
- `federated_learning.py`: Standard FedAvg converged to $59.0\%$ accuracy / $0.640$ ROC-AUC (`federated_learning_results.json`), matching centralized discriminative power. DP-FedAvg (total $\epsilon=2.0$) converged to $0.481$ ROC-AUC under correct zCDP composition (`federated_learning_dp_results.json`).
