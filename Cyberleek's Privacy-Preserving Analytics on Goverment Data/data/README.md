# Dataset Documentation

**IMPORTANT DISCLAIMER:** This is entirely synthetic data generated for the purposes of this hackathon demonstrator. It contains no real patient information, and any resemblance to real individuals is purely coincidental.

## What the Data Represents
This directory contains simulated patient records from three distinct UAE hospitals. The data is designed to reflect demographic and clinical characteristics relevant to public health analytics, specifically focusing on diabetes prevalence and 30-day hospital readmission risk.

## How to Regenerate
To generate or regenerate the datasets from scratch, run the data generation script from the project root:
```bash
python generate_dataset.py [--seed SEED]
```

## Datasets
The script generates three hospital-specific CSV files:
- `hospital_a_abu_dhabi.csv`: Al Mafraq Hospital (Abu Dhabi) — 8,000 records, older/Emirati-skewed population, highest diabetes prevalence.
- `hospital_b_dubai.csv`: Rashid Hospital (Dubai) — 5,000 records, younger/more diverse population (incl. Filipino nationality group).
- `hospital_c_rak.csv`: Saqr Hospital (Ras Al Khaimah) — 2,000 records, mixed demographics.

Patient IDs are prefixed per hospital (`AMH-`, `RH-`, `SH-`) and are guaranteed unique both within and across files — no patient appears in more than one hospital's data, and no raw records are ever shared between hospitals.

## Schema Definition

### Core Columns (present in all 3 files)
| Column Name | Type | Description | Range / Values |
| :--- | :--- | :--- | :--- |
| `patient_id` | String | Unique synthetic identifier, hospital-prefixed | e.g. `AMH-00001`, `RH-00001`, `SH-00001` |
| `age` | Integer | Patient age in years | 18 - 95 |
| `gender` | String | Patient gender | `M`, `F` |
| `nationality` | String | Patient nationality group | `Emirati`, `South Asian`, `Filipino` (Hospital B only), `Other` |
| `bmi` | Float | Body Mass Index | 16.0 - 50.0 |
| `has_diabetes` | Boolean | Diabetes diagnosis indicator | `True` / `False` |
| `has_hypertension` | Boolean | Hypertension diagnosis indicator | `True` / `False` |
| `has_obesity` | Boolean | Derived: `True` iff `bmi >= 30.0` | `True` / `False` |
| `smoking_status` | String | Smoking status | `Current`, `Former`, `Never` |
| `blood_pressure_systolic` | Integer | Systolic blood pressure (mmHg) | 90 - 200, always >= diastolic + 25 |
| `blood_pressure_diastolic` | Integer | Diastolic blood pressure (mmHg) | 60 - 120 |
| `hba1c` | Float | Hemoglobin A1c level (%) | 4.0 - 14.0 (may be missing, see below) |
| `admission_date` | Date | Hospital admission date | 2024-01-01 to 2025-12-31, skewed toward weekdays |
| `discharge_date` | Date | Hospital discharge date | `admission_date + length_of_stay` days |
| `length_of_stay` | Integer | Length of stay in days | 1 - 35 |
| `readmitted_30_days` | Boolean | Readmitted within 30 days of discharge | `True` / `False` |

### Hospital-Specific Extra Columns
Different hospitals collect slightly different supplementary data — this is intentional, to reflect realistic non-identical schemas across institutions:
- **Hospital A (Abu Dhabi):** `insurance_type` (`Thiqa`, `Private`, `Self-pay`), `cholesterol_total` (Float, ~5% missing)
- **Hospital B (Dubai):** `insurance_type` (`DHA`, `Private`, `Self-pay`), `emergency_admission` (Boolean)
- **Hospital C (RAK):** `referral_source` (`Walk-in`, `GP`, `Emergency`)

## Missing Data Patterns
To simulate real-world electronic health records (EHR):
- `hba1c` is missing for ~8% of non-diabetic patients and ~3% of diabetic patients (diabetics are tested more consistently), for an overall missingness of roughly 3-12%.
- `cholesterol_total` (Hospital A only) is missing for ~5% of records.
- All other core fields (`age`, `gender`, `patient_id`, `has_diabetes`, `has_hypertension`, etc.) have no missing values.

## Ground Truth Statistics
Run the validation script to check schema/distribution/internal-consistency correctness and print the true, non-private statistics across the datasets (used as the baseline for measuring utility loss in the DP algorithms):
```bash
python validate_dataset.py
```
