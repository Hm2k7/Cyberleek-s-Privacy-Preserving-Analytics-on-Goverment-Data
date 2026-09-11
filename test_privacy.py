"""
Unit & Mathematical Verification Test Suite for Differential Privacy Core
Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data

Person 2: Core Privacy Technique Engineer

Validates:
1. Laplace noise empirical distribution matches theoretical Var = 2*(Δ/ε)^2
2. Gaussian noise empirical distribution matches theoretical σ^2
3. Strict sensitivity clipping enforcement (preventing outlier privacy breaches)
4. Budget tracker sequential composition summation
5. Budget tracker parallel composition across disjoint partitions
6. Budget exhaustion exception handling
7. Bounded post-processing properties (non-negative counts, [0, 1] proportions)

Usage:
    python test_privacy.py
"""

import sys
import math
import numpy as np

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dp_core import (
    laplace_mechanism,
    laplace_confidence_interval,
    gaussian_mechanism,
    clip_values,
    dp_count,
    dp_proportion,
    dp_mean,
    PrivacyBudgetTracker,
    PrivacyBudgetExhaustedError,
)
from federated_learning import (
    sigma_for_composed_gaussian,
    zcdp_epsilon_from_rho,
)


def run_test(test_name: str, test_func):
    """Executes a test and prints formatted pass/fail status."""
    try:
        test_func()
        print(f"  [✅ PASS] {test_name}")
        return True
    except Exception as e:
        print(f"  [❌ FAIL] {test_name}: {e}")
        return False


def test_laplace_distribution_moments():
    """Verify empirical mean and variance of Laplace mechanism match analytical formulas."""
    rng = np.random.default_rng(12345)
    sensitivity = 2.0
    epsilon = 0.5
    scale = sensitivity / epsilon  # 4.0
    expected_var = 2.0 * (scale ** 2)  # 32.0

    n_samples = 100_000
    samples = np.array([
        laplace_mechanism(true_value=0.0, sensitivity=sensitivity, epsilon=epsilon, rng=rng)
        for _ in range(n_samples)
    ])

    emp_mean = np.mean(samples)
    emp_var = np.var(samples)

    # Mean should be close to 0 (within 3 standard errors = 3 * sqrt(32 / 100000) ≈ 0.05)
    assert abs(emp_mean) < 0.08, f"Empirical mean {emp_mean:.4f} too far from 0.0"

    # Variance should match theoretical within 3% relative error
    rel_var_err = abs(emp_var - expected_var) / expected_var
    assert rel_var_err < 0.03, f"Empirical var {emp_var:.4f} deviated {rel_var_err:.2%} from expected {expected_var:.4f}"


def test_laplace_confidence_interval():
    """Verify that 95% of Laplace noise samples lie inside the theoretical confidence radius."""
    rng = np.random.default_rng(54321)
    sensitivity = 1.0
    epsilon = 1.0
    radius_95 = laplace_confidence_interval(sensitivity=sensitivity, epsilon=epsilon, confidence=0.95)

    n_samples = 50_000
    noise_samples = np.array([
        laplace_mechanism(0.0, sensitivity, epsilon, rng=rng)
        for _ in range(n_samples)
    ])

    within_ci_rate = np.mean(np.abs(noise_samples) <= radius_95)
    assert 0.94 <= within_ci_rate <= 0.96, f"Expected ~95% within CI, got {within_ci_rate:.2%}"


def test_gaussian_distribution_variance():
    """Verify Gaussian mechanism produces standard deviation matching analytical sigma."""
    rng = np.random.default_rng(999)
    sensitivity = 1.0
    epsilon = 0.8
    delta = 1e-5
    theoretical_sigma = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon

    n_samples = 50_000
    samples = np.array([
        gaussian_mechanism(0.0, sensitivity, epsilon, delta, rng=rng)
        for _ in range(n_samples)
    ])

    emp_sigma = np.std(samples)
    rel_err = abs(emp_sigma - theoretical_sigma) / theoretical_sigma
    assert rel_err < 0.03, f"Empirical sigma {emp_sigma:.4f} deviated {rel_err:.2%} from expected {theoretical_sigma:.4f}"


def test_sensitivity_clipping():
    """Verify clipping functions strictly bound all extreme and adversarial outliers."""
    raw_vals = np.array([-100.0, 2.5, 4.0, 8.5, 14.0, 999.0])
    clipped = clip_values(raw_vals, lower_bound=4.0, upper_bound=14.0)

    assert np.all(clipped >= 4.0), "Clipped values dropped below lower bound"
    assert np.all(clipped <= 14.0), "Clipped values exceeded upper bound"
    assert clipped[0] == 4.0, "Outlier below bound not clamped to minimum"
    assert clipped[-1] == 14.0, "Outlier above bound not clamped to maximum"


def test_privacy_budget_sequential_composition():
    """Verify sequential composition properly accumulates privacy expenditures."""
    tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)
    assert tracker.remaining_epsilon == 1.0

    tracker.allocate("query_1", epsilon=0.3)
    assert math.isclose(tracker.consumed_epsilon, 0.3)
    assert math.isclose(tracker.remaining_epsilon, 0.7)

    tracker.allocate("query_2", epsilon=0.4)
    assert math.isclose(tracker.consumed_epsilon, 0.7)
    assert math.isclose(tracker.remaining_epsilon, 0.3)


def test_privacy_budget_parallel_composition():
    """Verify parallel composition on disjoint hospital partitions."""
    tracker = PrivacyBudgetTracker(total_epsilon=1.0, total_delta=1e-5)

    # Query Hospital A at eps=0.25
    tracker.allocate("query_hosp_a", epsilon=0.25, partition_id="Hospital_A")
    assert math.isclose(tracker.consumed_epsilon, 0.25)

    # Query Hospital B at eps=0.25 (Disjoint partition: global budget should remain 0.25!)
    tracker.allocate("query_hosp_b", epsilon=0.25, partition_id="Hospital_B")
    assert math.isclose(tracker.consumed_epsilon, 0.25), f"Expected 0.25, got {tracker.consumed_epsilon}"

    # Query Hospital C at eps=0.30 (Expands max partition cost to 0.30)
    tracker.allocate("query_hosp_c", epsilon=0.30, partition_id="Hospital_C")
    assert math.isclose(tracker.consumed_epsilon, 0.30), f"Expected 0.30, got {tracker.consumed_epsilon}"


def test_privacy_budget_exhaustion_exception():
    """Verify that exceeding privacy budget raises PrivacyBudgetExhaustedError."""
    tracker = PrivacyBudgetTracker(total_epsilon=0.5, total_delta=1e-5)
    tracker.allocate("query_safe", epsilon=0.4)

    try:
        tracker.allocate("query_excessive", epsilon=0.2)
        assert False, "Failed to raise PrivacyBudgetExhaustedError when budget exceeded"
    except PrivacyBudgetExhaustedError:
        pass  # Expected behavior


def test_dp_proportion_and_postprocessing():
    """Verify dp_proportion clamps values to [0.0, 1.0] and handles edge cases."""
    rng = np.random.default_rng(42)
    # Test near boundary 0
    p_low, _ = dp_proportion(raw_numerator_count=0, total_population=1000, epsilon=0.1, rng=rng)
    assert 0.0 <= p_low <= 1.0, f"Proportion {p_low} out of bounds [0, 1]"

    # Test near boundary 1
    p_high, _ = dp_proportion(raw_numerator_count=1000, total_population=1000, epsilon=0.1, rng=rng)
    assert 0.0 <= p_high <= 1.0, f"Proportion {p_high} out of bounds [0, 1]"


def test_dp_count_nonnegative():
    """Verify dp_count is always an integer >= 0."""
    rng = np.random.default_rng(777)
    # Count of 0 with strong noise could produce negative without post-processing
    for _ in range(50):
        c = dp_count(raw_count=0, epsilon=0.1, rng=rng)
        assert isinstance(c, int), f"Count must be integer, got {type(c)}"
        assert c >= 0, f"Count must be non-negative, got {c}"


def test_zcdp_composition_round_trip():
    """Verify the DP-FedAvg zCDP composition math is self-consistent: the per-round
    sigma it derives, when composed back up over `rounds` releases, reproduces the
    caller's requested TOTAL epsilon (not rounds*epsilon, and not just epsilon)."""
    total_epsilon = 2.0
    delta = 1e-5
    rounds = 15
    l2_sensitivity = 0.03333

    sigma, rho_total, rho_per_round = sigma_for_composed_gaussian(
        total_epsilon=total_epsilon, delta=delta, rounds=rounds, l2_sensitivity=l2_sensitivity
    )

    # rho composes additively and exactly under zCDP: rounds independent releases
    # of the same (sensitivity, sigma) sum to rounds * rho_per_round.
    assert math.isclose(rho_per_round * rounds, rho_total, rel_tol=1e-9)

    # Converting the composed rho_total back to (epsilon, delta)-DP must reproduce
    # the originally requested total_epsilon.
    recovered_epsilon = zcdp_epsilon_from_rho(rho_total, delta)
    assert math.isclose(recovered_epsilon, total_epsilon, rel_tol=1e-6), \
        f"Composed budget recovered eps={recovered_epsilon:.6f}, expected {total_epsilon}"

    # A single release at this sigma consumes exactly rho_per_round of zCDP.
    single_release_rho = (l2_sensitivity ** 2) / (2.0 * sigma ** 2)
    assert math.isclose(single_release_rho, rho_per_round, rel_tol=1e-9)


def test_zcdp_composition_more_rounds_needs_more_noise():
    """Verify that, for a fixed TOTAL epsilon budget, composing over more rounds
    requires a larger per-round sigma (since the same total privacy loss must be
    spread across more releases) -- this is the property whose absence caused
    DP-FedAvg to silently overspend privacy in the original implementation."""
    sigma_few, _, _ = sigma_for_composed_gaussian(total_epsilon=2.0, delta=1e-5, rounds=5, l2_sensitivity=0.0333)
    sigma_many, _, _ = sigma_for_composed_gaussian(total_epsilon=2.0, delta=1e-5, rounds=50, l2_sensitivity=0.0333)
    assert sigma_many > sigma_few, "More communication rounds at a fixed total epsilon must require more noise per round"

    # And, critically, sigma must scale sub-linearly (~sqrt(rounds)) under zCDP composition,
    # not linearly (~rounds) as it would under naive basic/sequential composition.
    ratio = sigma_many / sigma_few
    assert ratio < 10.0 - 1e-9, f"sigma ratio {ratio:.2f} grew ~linearly with rounds (10x for 10x rounds); expected sub-linear (~sqrt) growth"


def main():
    print("=" * 65)
    print("  CYBERLEEK — CORE PRIVACY TECHNIQUE UNIT TESTS")
    print("  Differential Privacy Mathematical & Budget Verification")
    print("=" * 65)

    tests = [
        ("Laplace Distribution Moments (Var = 2*(Δ/ε)^2)", test_laplace_distribution_moments),
        ("Laplace Theoretical 95% Confidence Radius", test_laplace_confidence_interval),
        ("Gaussian Distribution Variance (σ^2)", test_gaussian_distribution_variance),
        ("Sensitivity Clipping & Outlier Bounding", test_sensitivity_clipping),
        ("Sequential Budget Composition Summation", test_privacy_budget_sequential_composition),
        ("Parallel Budget Composition on Disjoint Silos", test_privacy_budget_parallel_composition),
        ("Privacy Budget Exhaustion Guardrail", test_privacy_budget_exhaustion_exception),
        ("DP Proportion Post-Processing Invariance [0, 1]", test_dp_proportion_and_postprocessing),
        ("DP Count Non-Negativity & Integer Enforcement", test_dp_count_nonnegative),
        ("DP-FedAvg zCDP Composition Round-Trip Consistency", test_zcdp_composition_round_trip),
        ("DP-FedAvg zCDP Sub-Linear Noise Growth vs. Rounds", test_zcdp_composition_more_rounds_needs_more_noise),
    ]

    passed = 0
    for name, func in tests:
        if run_test(name, func):
            passed += 1

    print("=" * 65)
    print(f"  TEST RESULT: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("  ✅ ALL DIFFERENTIAL PRIVACY TESTS PASSED")
    else:
        print(f"  ❌ {len(tests) - passed} TEST(S) FAILED")
    print("=" * 65)

    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
