"""
Differential Privacy Mathematical Core & Budget Accounting
Cyberleek — Privacy-Preserving Analytics on Sensitive Government Data

Person 2: Core Privacy Technique Engineer

This module provides verified mathematical primitives for Differential Privacy:
1. Laplace Mechanism (Pure ε-DP)
2. Gaussian Mechanism ((ε, δ)-DP)
3. Strict sensitivity clipping and validation (preventing unbounded outlier leaks)
4. DP Mean and Proportion estimators with budget splitting
5. PrivacyBudgetTracker with Sequential and Parallel Composition accounting
6. Post-processing invariant enforcement (non-negative counts, bounded proportions)
"""

import math
import sys
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class PrivacyBudgetExhaustedError(Exception):
    """Raised when an analytical query demands more privacy budget (ε or δ) than remaining."""
    pass


class PrivacyBudgetTracker:
    """
    Cryptographic Privacy Budget Ledger.
    
    Tracks cumulative ε and δ expenditures to prevent the classic DP vulnerability of
    infinite budget reuse across repeated queries.
    
    Supports:
    - Sequential Composition: Total ε consumed = sum(ε_i) across queries on identical data.
    - Parallel Composition: Cost across mutually disjoint partitions (e.g. Hospital A, B, C)
      is max(ε_k) rather than sum(ε_k).
    - Hard budget enforcement: Rejects queries exceeding total allocation.
    """

    def __init__(self, total_epsilon: float = 1.0, total_delta: float = 1e-5):
        if total_epsilon <= 0:
            raise ValueError(f"total_epsilon must be positive, got {total_epsilon}")
        if total_delta < 0:
            raise ValueError(f"total_delta cannot be negative, got {total_delta}")

        self.total_epsilon = float(total_epsilon)
        self.total_delta = float(total_delta)
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.query_log: List[Dict[str, Any]] = []
        
        # Partition-specific tracking for parallel composition
        # e.g., {'Hospital_A': 0.2, 'Hospital_B': 0.2}
        self.partition_budgets: Dict[str, float] = {}

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.consumed_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self.consumed_delta)

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Checks whether the requested budget can be allocated without exceeding limits."""
        return (self.consumed_epsilon + epsilon <= self.total_epsilon + 1e-9) and \
               (self.consumed_delta + delta <= self.total_delta + 1e-12)

    def allocate(self, query_name: str, epsilon: float, delta: float = 0.0, 
                 partition_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Allocates privacy budget for a query, verifying bounds and logging the transaction.
        
        If partition_id is provided, applies parallel composition rules: queries on disjoint
        datasets only increase the global budget by the marginal increase in max(partition_epsilons).
        """
        if epsilon <= 0:
            raise ValueError(f"Query epsilon must be positive, got {epsilon}")
        if delta < 0:
            raise ValueError(f"Query delta cannot be negative, got {delta}")

        effective_eps_increment = epsilon
        if partition_id is not None:
            # Parallel composition logic:
            old_part_eps = self.partition_budgets.get(partition_id, 0.0)
            new_part_eps = old_part_eps + epsilon
            self.partition_budgets[partition_id] = new_part_eps
            
            # The parallel cost across all partitions is max(partition_budgets)
            # Find the new max across partitions
            new_max = max(self.partition_budgets.values())
            # For simplicity in global sequential accounting, we track partition-specific
            # operations as bounded by new_max.
            effective_eps_increment = max(0.0, new_max - (self.consumed_epsilon - old_part_eps))

        if not self.can_spend(effective_eps_increment, delta):
            raise PrivacyBudgetExhaustedError(
                f"Privacy budget exhausted! Requested ε={effective_eps_increment:.4f}, δ={delta:.2e}, "
                f"but only ε={self.remaining_epsilon:.4f}, δ={self.remaining_delta:.2e} remains "
                f"(Total allocated: ε={self.total_epsilon}, δ={self.total_delta})."
            )

        self.consumed_epsilon += effective_eps_increment
        self.consumed_delta += delta

        entry = {
            "query_id": len(self.query_log) + 1,
            "query_name": query_name,
            "partition_id": partition_id or "global",
            "requested_epsilon": float(epsilon),
            "requested_delta": float(delta),
            "effective_epsilon_increment": float(effective_eps_increment),
            "cumulative_epsilon": float(self.consumed_epsilon),
            "cumulative_delta": float(self.consumed_delta),
            "remaining_epsilon": float(self.remaining_epsilon),
        }
        self.query_log.append(entry)
        return entry

    def summary(self) -> Dict[str, Any]:
        """Returns a snapshot of the privacy budget status."""
        return {
            "total_budget": {"epsilon": self.total_epsilon, "delta": self.total_delta},
            "consumed_budget": {"epsilon": self.consumed_epsilon, "delta": self.consumed_delta},
            "remaining_budget": {"epsilon": self.remaining_epsilon, "delta": self.remaining_delta},
            "total_queries_executed": len(self.query_log),
            "budget_utilization_pct": (self.consumed_epsilon / self.total_epsilon) * 100.0,
        }


# =====================================================================
# SENSITIVITY CLIPPING & BOUNDING
# =====================================================================

def clip_values(values: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    """
    Strictly clips values to [lower_bound, upper_bound] to guarantee sensitivity bounds.
    
    A critical failure in amateur DP implementations is calculating empirical sensitivity
    without clipping, allowing a single adversarial outlier to leak private information.
    """
    if lower_bound > upper_bound:
        raise ValueError(f"lower_bound ({lower_bound}) must be <= upper_bound ({upper_bound})")
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, lower_bound, upper_bound)


# =====================================================================
# CORE DIFFERENTIAL PRIVACY MECHANISMS
# =====================================================================

def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float, 
                      rng: Optional[np.random.Generator] = None) -> float:
    """
    Applies the Laplace Mechanism for pure ε-Differential Privacy.
    
    Noise ~ Laplace(0, scale = sensitivity / epsilon).
    
    Theoretical Properties:
    - Global L1 Sensitivity: Δf
    - Noise scale b = Δf / ε
    - Variance: Var[Y] = 2 * b^2 = 2 * (Δf / ε)^2
    - 95% Confidence Radius: b * ln(20) ≈ 2.9957 * b
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if sensitivity < 0:
        raise ValueError(f"sensitivity cannot be negative, got {sensitivity}")
    if sensitivity == 0:
        return float(true_value)

    if rng is None:
        rng = np.random.default_rng()

    scale = sensitivity / epsilon
    noise = rng.laplace(loc=0.0, scale=scale)
    return float(true_value + noise)


def laplace_confidence_interval(sensitivity: float, epsilon: float, confidence: float = 0.95) -> float:
    """
    Computes theoretical error radius ±r such that P(|Noise| <= r) = confidence.
    For Laplace distribution: r = (Δf / ε) * (-ln(1 - confidence)).
    For 95% confidence: r = (Δf / ε) * ln(20) ≈ 2.9957 * (Δf / ε).
    """
    if epsilon <= 0 or sensitivity < 0 or not (0 < confidence < 1):
        raise ValueError("Invalid parameters for Laplace confidence interval")
    scale = sensitivity / epsilon
    radius = -scale * math.log(1.0 - confidence)
    return float(radius)


def gaussian_mechanism(true_value: float, l2_sensitivity: float, epsilon: float, delta: float,
                       rng: Optional[np.random.Generator] = None) -> float:
    """
    Applies the Gaussian Mechanism for (ε, δ)-Differential Privacy.
    
    Noise ~ N(0, σ^2) where σ = (Δ2 f * sqrt(2 * ln(1.25 / δ))) / ε.
    Valid for ε in (0, 1].
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if l2_sensitivity < 0:
        raise ValueError(f"l2_sensitivity cannot be negative, got {l2_sensitivity}")
    if l2_sensitivity == 0:
        return float(true_value)

    if rng is None:
        rng = np.random.default_rng()

    sigma = (l2_sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon
    noise = rng.normal(loc=0.0, scale=sigma)
    return float(true_value + noise)


# =====================================================================
# DIFFERONENTIALLY PRIVATE STATISTICAL AGGREGATORS
# =====================================================================

def dp_count(raw_count: int, epsilon: float, tracker: Optional[PrivacyBudgetTracker] = None,
             query_name: str = "dp_count", partition_id: Optional[str] = None,
             rng: Optional[np.random.Generator] = None) -> int:
    """
    Computes a differentially private count using the Laplace mechanism.
    
    - Global L1 Sensitivity: Δf = 1 (Adding/removing one patient changes count by at most 1)
    - Post-Processing: Result is bounded to be >= 0 and rounded to integer.
    """
    if tracker is not None:
        tracker.allocate(query_name=query_name, epsilon=epsilon, partition_id=partition_id)

    sensitivity = 1.0
    noisy_val = laplace_mechanism(float(raw_count), sensitivity=sensitivity, epsilon=epsilon, rng=rng)
    
    # Post-processing: counts cannot be negative
    clamped_count = max(0, int(round(noisy_val)))
    return clamped_count


def dp_proportion(raw_numerator_count: int, total_population: int, epsilon: float,
                  tracker: Optional[PrivacyBudgetTracker] = None,
                  query_name: str = "dp_proportion", partition_id: Optional[str] = None,
                  rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    Computes a differentially private proportion (prevalence rate).
    
    Case A: If total_population N is public/known (e.g. registered hospital beds/sample count),
            sensitivity Δ = 1 / N, query costs full ε.
    Case B: If both numerator and denominator are private, budget is split (ε/2 each).
    
    Here we treat known denominator N (standard in public health census):
    Sensitivity Δf = 1.0 / total_population.
    
    Returns:
        (dp_prevalence, confidence_radius_95)
    """
    if total_population <= 0:
        raise ValueError("total_population must be positive")

    if tracker is not None:
        tracker.allocate(query_name=query_name, epsilon=epsilon, partition_id=partition_id)

    sensitivity = 1.0 / float(total_population)
    true_rate = float(raw_numerator_count) / float(total_population)
    
    noisy_rate = laplace_mechanism(true_rate, sensitivity=sensitivity, epsilon=epsilon, rng=rng)
    
    # Post-processing: proportions must strictly reside in [0.0, 1.0]
    clamped_rate = float(np.clip(noisy_rate, 0.0, 1.0))
    radius_95 = laplace_confidence_interval(sensitivity=sensitivity, epsilon=epsilon, confidence=0.95)
    
    return clamped_rate, radius_95


def dp_mean(values: np.ndarray, lower_bound: float, upper_bound: float, epsilon: float,
            known_n: Optional[int] = None, tracker: Optional[PrivacyBudgetTracker] = None,
            query_name: str = "dp_mean", partition_id: Optional[str] = None,
            rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    Computes a differentially private mean for bounded continuous attributes
    (e.g., HbA1c, Systolic BP, Age, Length of Stay).
    
    Prevents classic DP pitfall:
    1. First strictly clips values to [lower_bound, upper_bound].
    2. If N is known: Sensitivity Δ = (upper_bound - lower_bound) / N. Entire ε used for noisy sum/N.
    3. If N is private: Splits ε into ε_sum = 0.5*ε and ε_count = 0.5*ε (DP Sum / DP Count).
    
    Returns:
        (dp_mean_value, confidence_radius_95)
    """
    clipped = clip_values(values, lower_bound, upper_bound)
    n_records = len(clipped) if known_n is None else known_n
    if n_records <= 0:
        raise ValueError("Cannot compute DP mean on empty dataset")

    range_span = upper_bound - lower_bound

    if known_n is not None:
        # Known N scenario: Sensitivity = (U - L) / N
        sensitivity = range_span / float(n_records)
        true_mean = float(np.mean(clipped))
        
        if tracker is not None:
            tracker.allocate(query_name=query_name, epsilon=epsilon, partition_id=partition_id)
            
        noisy_mean = laplace_mechanism(true_mean, sensitivity=sensitivity, epsilon=epsilon, rng=rng)
        clamped_mean = float(np.clip(noisy_mean, lower_bound, upper_bound))
        radius_95 = laplace_confidence_interval(sensitivity=sensitivity, epsilon=epsilon, confidence=0.95)
        return clamped_mean, radius_95

    else:
        # Private N scenario: Split budget ε_sum = 0.5 * ε, ε_count = 0.5 * ε
        eps_sum = 0.5 * epsilon
        eps_count = 0.5 * epsilon
        
        if tracker is not None:
            tracker.allocate(query_name=f"{query_name}_sum", epsilon=eps_sum, partition_id=partition_id)
            tracker.allocate(query_name=f"{query_name}_count", epsilon=eps_count, partition_id=partition_id)

        true_sum = float(np.sum(clipped))
        sum_sensitivity = range_span
        noisy_sum = laplace_mechanism(true_sum, sensitivity=sum_sensitivity, epsilon=eps_sum, rng=rng)
        
        count_sensitivity = 1.0
        noisy_count = laplace_mechanism(float(len(clipped)), sensitivity=count_sensitivity, epsilon=eps_count, rng=rng)
        clamped_count = max(1.0, noisy_count)  # Avoid division by zero
        
        dp_est = noisy_sum / clamped_count
        clamped_mean = float(np.clip(dp_est, lower_bound, upper_bound))
        
        # Conservative 95% radius
        radius_95 = (sum_sensitivity / eps_sum * 2.9957) / clamped_count
        return clamped_mean, float(radius_95)
