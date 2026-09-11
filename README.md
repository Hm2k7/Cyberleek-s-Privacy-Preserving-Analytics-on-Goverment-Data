# Cyberleek using differential privacy for the statistics and federated learning for the model, then proving the protection works by attacking it.

Everything runs locally on synthetic data. No external services, no GPU, no network access required.

## Requirements

- Python 3.10 or newer
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` (see `requirements.txt`)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running it

The dataset is already committed, so you can start at step 2. Each script prints its own summary and writes a JSON file you can inspect.

```bash
# 1. (Optional) Regenerate the synthetic hospital data expect "42/42 checks passed"
.venv/bin/python validate_dataset.py

# 3. Check the DP math and budget accounting membership inference + differencing    -> attack_results.json
.venv/bin/python attack_demo.py

# 8. Sweep the privacy budget (~45s)  -> tradeoff_curves.png, tradeoff_summary.json
.venv/bin/python tradeoff_analysis.py --trials 20
```

Steps 4 26.6% (true value 26.63%), error well under 0.1pp |
| `federated_learning.py` | FedAvg ROC-AUC with no raw data shared |
| `federated_learning.py --dp` | ROC-AUC drops to 0.05) |
| `tradeoff_analysis.py` | Two curves written to `tradeoff_curves.png` |

Full numbers are in [`docs/results.md`](docs/results.md).

## Useful options

```bash
# Tighter or looser privacy budget for the statistics
python dp_analytics.py --epsilon 0.5

# Longer federated training, custom DP parameters
python federated_learning.py --rounds 30 --local-epochs 5 --lr 0.05 \
                            --dp --epsilon 5.0 --delta 1e-5 --clip-norm 0.05

# Faster trade-off sweep
python tradeoff_analysis.py --trials 5
```

Most scripts also take `--data-dir` (default `data/`), `--output` (where to write the JSON), and `--seed`. For `federated_learning.py --dp`, `--epsilon` is the **total** budget for the whole run, not per round data/                                # 15,000 synthetic records across 3 hospitals
hospital_a_abu_dhabi.csv         # 8,000 records
hospital_b_dubai.csv             # 5,000 records
hospital_c_rak.csv               # 2,000 records
README.md                        # Schema documentation
            generate_dataset.py                  # Synthetic data generator
test_privacy.py                      # 11 tests on the DP math and budget ledger
dp_analytics.py                      # DP statistics engine
federated_learning.py                # FedAvg and DP-FedAvg
tradeoff_analysis.py                 # Privacy budget sweep and plots
Cyberleek_Report.pdf                 # Write-up (objective results)
the problem, who deploys this, and what it replaces
- [`docs/results.md`](docs/results.md) importing the modules instead of using the CLIs
- [`docs/verification.md`](docs/verification.md) why these mechanisms, and the legal mapping
