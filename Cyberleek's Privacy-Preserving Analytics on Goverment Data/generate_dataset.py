"""
Dataset Generator for Privacy-Preserving Analytics on Sensitive Government Data
Generates synthetic, realistic healthcare records distributed across 3 hospitals in the UAE.

Requirements: numpy, pandas, scipy
Usage: python generate_dataset.py [--seed SEED]
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from datetime import datetime, timedelta

def generate_hospital_data(n, config, rng):
    """Generates synthetic patient data for a single hospital."""
    
    # 1. Age, Gender, Nationality
    # Age mixture
    weights = config['age_weights']
    components = rng.choice(3, size=n, p=weights)
    
    age = np.zeros(n)
    age[components == 0] = rng.normal(35, 10, size=(components == 0).sum())
    age[components == 1] = rng.normal(50, 12, size=(components == 1).sum())
    age[components == 2] = rng.normal(65, 10, size=(components == 2).sum())
    
    age = np.clip(np.round(age), 18, 95).astype(int)
    
    # Gender
    gender = rng.choice(['M', 'F'], size=n, p=[0.55, 0.45])
    
    # Nationality
    nat_keys = list(config['nationality_dist'].keys())
    nat_probs = list(config['nationality_dist'].values())
    nationality = rng.choice(nat_keys, size=n, p=nat_probs)
    
    # Patient ID
    patient_ids = [f"{config['prefix']}-{str(i).zfill(5)}" for i in range(1, n + 1)]
    
    # 2. BMI (conditional on age)
    # Target mean ~27.5, std ~5.5, correlated with age
    # Base it loosely on age: older -> slightly higher BMI up to a point
    bmi_base = 20.0 + 0.15 * age + rng.normal(0, 4.0, size=n)
    bmi = np.round(np.clip(bmi_base, 16.0, 50.0), 1)
    
    # 3. Diabetes
    is_emirati = (nationality == 'Emirati').astype(int)
    is_south_asian = (nationality == 'South Asian').astype(int)
    
    diab_logit = -3.8 + 0.04 * age + 0.07 * (bmi - 25) + 0.6 * is_emirati + 0.35 * is_south_asian
    diab_prob = expit(diab_logit)
    has_diabetes = rng.binomial(1, diab_prob).astype(bool)
    
    # 4. Hypertension
    hyp_logit = -3.8 + 0.05 * age + 0.06 * (bmi - 25) + 0.6 * has_diabetes.astype(int)
    hyp_prob = expit(hyp_logit)
    has_hypertension = rng.binomial(1, hyp_prob).astype(bool)
    
    # 5. Obesity
    has_obesity = (bmi >= 30.0)
    
    # 6. Smoking Status
    smoking_status = np.empty(n, dtype=object)
    
    m_mask = (gender == 'M')
    f_mask = (gender == 'F')
    
    smoking_status[m_mask] = rng.choice(['Current', 'Former', 'Never'], size=m_mask.sum(), p=[0.20, 0.12, 0.68])
    smoking_status[f_mask] = rng.choice(['Current', 'Former', 'Never'], size=f_mask.sum(), p=[0.05, 0.03, 0.92])
    
    # 7. Blood Pressure
    sys_bp = np.zeros(n)
    dia_bp = np.zeros(n)
    
    sys_bp[~has_hypertension] = rng.normal(120, 12, size=(~has_hypertension).sum())
    dia_bp[~has_hypertension] = rng.normal(76, 6, size=(~has_hypertension).sum())
    
    sys_bp[has_hypertension] = rng.normal(145, 14, size=has_hypertension.sum())
    dia_bp[has_hypertension] = rng.normal(92, 8, size=has_hypertension.sum())
    
    sys_bp = np.clip(np.round(sys_bp), 90, 200).astype(int)
    dia_bp = np.clip(np.round(dia_bp), 60, 120).astype(int)
    
    # Ensure sys - dia >= 25
    diff = sys_bp - dia_bp
    invalid = diff < 25
    sys_bp[invalid] = dia_bp[invalid] + rng.integers(25, 40, size=invalid.sum())
    sys_bp = np.clip(sys_bp, 90, 200)
    
    # 8. HbA1c
    hba1c = np.zeros(n)
    hba1c[has_diabetes] = rng.normal(7.8, 1.5, size=has_diabetes.sum())
    hba1c[~has_diabetes] = rng.normal(5.4, 0.4, size=(~has_diabetes).sum())
    hba1c = np.clip(hba1c, 4.0, 14.0)
    
    # 9. Dates and Length of Stay
    start_date = datetime(2024, 1, 1).timestamp()
    end_date = datetime(2025, 12, 31).timestamp()
    
    admit_timestamps = rng.uniform(start_date, end_date, size=n)
    admit_dates = [datetime.fromtimestamp(ts).date() for ts in admit_timestamps]
    
    # Skew towards weekdays (0-4 are Mon-Fri)
    range_start = datetime(2024, 1, 1).date()
    range_end = datetime(2025, 12, 31).date()
    for i in range(n):
        if admit_dates[i].weekday() >= 5:
            if rng.random() < 0.6:  # 60% chance to move a weekend admit to a weekday
                offset = int(rng.choice([-2, -1, 1, 2]))
                shifted = admit_dates[i] + timedelta(days=offset)
                admit_dates[i] = min(max(shifted, range_start), range_end)
                
    los = np.round(rng.gamma(shape=1.8, scale=2.2, size=n) + 1)
    los = np.clip(los, 1, 35).astype(int)
    
    discharge_dates = [admit_dates[i] + timedelta(days=int(los[i])) for i in range(n)]
    
    # 10. Readmission
    readm_logit = -2.6 + 0.015 * age + 0.45 * has_diabetes.astype(int) + 0.6 * (age > 65).astype(int) + 0.4 * (los > 7).astype(int)
    readm_prob = expit(readm_logit)
    readmitted_30_days = rng.binomial(1, readm_prob).astype(bool)
    
    # Compile base dataframe
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'age': age,
        'gender': gender,
        'nationality': nationality,
        'bmi': bmi,
        'has_diabetes': has_diabetes,
        'has_hypertension': has_hypertension,
        'has_obesity': has_obesity,
        'smoking_status': smoking_status,
        'blood_pressure_systolic': sys_bp,
        'blood_pressure_diastolic': dia_bp,
        'hba1c': np.round(hba1c, 1),
        'admission_date': admit_dates,
        'discharge_date': discharge_dates,
        'length_of_stay': los,
        'readmitted_30_days': readmitted_30_days
    })
    
    # 11. Extra columns
    if 'insurance_type' in config['extra_columns']:
        options = config['extra_columns']['insurance_type']
        df['insurance_type'] = rng.choice(options, size=n)
        
    if 'cholesterol_total' in config['extra_columns']:
        # Correlated with age and bmi
        chol_base = 150 + 1.0 * age + 1.5 * (bmi - 25) + rng.normal(0, 30, size=n)
        chol_base = np.clip(np.round(chol_base), 100, 350)
        df['cholesterol_total'] = chol_base
        
    if 'emergency_admission' in config['extra_columns']:
        df['emergency_admission'] = rng.binomial(1, 0.3, size=n).astype(bool)
        
    if 'referral_source' in config['extra_columns']:
        df['referral_source'] = rng.choice(['Walk-in', 'GP', 'Emergency'], size=n, p=[0.4, 0.4, 0.2])
        
    # 12. Missing Data Injection
    # ~8% missing hba1c for non-diabetics
    non_diab_idx = df[~df['has_diabetes']].index
    missing_non_diab = rng.choice(non_diab_idx, size=int(0.08 * len(non_diab_idx)), replace=False)
    df.loc[missing_non_diab, 'hba1c'] = np.nan
    
    # ~3% missing hba1c for diabetics
    diab_idx = df[df['has_diabetes']].index
    missing_diab = rng.choice(diab_idx, size=int(0.03 * len(diab_idx)), replace=False)
    df.loc[missing_diab, 'hba1c'] = np.nan
    
    # Hospital A cholesterol missingness
    if 'cholesterol_total' in df.columns:
        missing_chol = rng.choice(df.index, size=int(0.05 * n), replace=False)
        df.loc[missing_chol, 'cholesterol_total'] = np.nan
        
    return df


def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"seed must be non-negative, got {ivalue}")
    return ivalue


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic UAE hospital data.")
    parser.add_argument("--seed", type=non_negative_int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    # Hospital Configurations
    hospitals = {
        'hospital_a_abu_dhabi': {
            'name': 'Al Mafraq Hospital (Abu Dhabi)',
            'prefix': 'AMH',
            'n_records': 8000,
            'nationality_dist': {'Emirati': 0.60, 'South Asian': 0.25, 'Other': 0.15},
            'age_weights': [0.20, 0.30, 0.50],  # Skews older (mu=65 is 50%)
            'extra_columns': {
                'insurance_type': ['Thiqa', 'Private', 'Self-pay'],
                'cholesterol_total': True
            }
        },
        'hospital_b_dubai': {
            'name': 'Rashid Hospital (Dubai)',
            'prefix': 'RH',
            'n_records': 5000,
            'nationality_dist': {'Emirati': 0.30, 'South Asian': 0.45, 'Filipino': 0.10, 'Other': 0.15},
            'age_weights': [0.50, 0.30, 0.20],  # Skews younger (mu=35 is 50%)
            'extra_columns': {
                'insurance_type': ['DHA', 'Private', 'Self-pay'],
                'emergency_admission': True
            }
        },
        'hospital_c_rak': {
            'name': 'Saqr Hospital (Ras Al Khaimah)',
            'prefix': 'SH',
            'n_records': 2000,
            'nationality_dist': {'Emirati': 0.50, 'South Asian': 0.30, 'Other': 0.20},
            'age_weights': [0.30, 0.40, 0.30],  # Mixed demographics
            'extra_columns': {
                'referral_source': ['Walk-in', 'GP', 'Emergency']
            }
        }
    }
    
    # Directory setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    all_dfs = []
    
    for fname, config in hospitals.items():
        print(f"Generating data for {config['name']}...")
        df = generate_hospital_data(config['n_records'], config, rng)
        
        out_path = os.path.join(data_dir, f"{fname}.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> Saved {config['n_records']} records to {out_path}")
        
        df['hospital'] = config['name']
        all_dfs.append(df)
        
    # Global statistics
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print("\n" + "="*40)
    print("SUMMARY STATISTICS")
    print("="*40)
    print(f"Total Records: {len(combined)}")
    print(f"Overall Readmission Rate: {combined['readmitted_30_days'].mean():.2%}")
    print(f"Overall Diabetes Prevalence: {combined['has_diabetes'].mean():.2%}")
    print("\nDiabetes Prevalence by Hospital:")
    diab_by_hosp = combined.groupby('hospital')['has_diabetes'].mean().sort_values(ascending=False)
    for hosp, rate in diab_by_hosp.items():
        print(f"  {hosp}: {rate:.2%}")
        

if __name__ == "__main__":
    main()
