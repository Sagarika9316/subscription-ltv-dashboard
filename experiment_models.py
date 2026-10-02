from math import ceil, isfinite
from statistics import NormalDist


def two_proportion_sample_size(
    baseline_rate,
    minimum_absolute_lift,
    alpha=0.05,
    power=0.80,
):
    """Estimate equal-arm sample size for a two-sided test of conversion rates."""
    values = (baseline_rate, minimum_absolute_lift, alpha, power)
    if not all(isfinite(value) for value in values):
        raise ValueError("Experiment inputs must be finite numbers.")
    if not 0 < baseline_rate < 1:
        raise ValueError("Baseline conversion rate must be between 0 and 1.")
    if not 0 < minimum_absolute_lift < 1 - baseline_rate:
        raise ValueError("Minimum lift must be positive and keep treatment conversion below 100%.")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1.")
    if not 0.5 < power < 1:
        raise ValueError("Power must be greater than 0.5 and less than 1.")

    treatment_rate = baseline_rate + minimum_absolute_lift
    pooled_rate = (baseline_rate + treatment_rate) / 2
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    pooled_variance = 2 * pooled_rate * (1 - pooled_rate)
    arm_variance = (
        baseline_rate * (1 - baseline_rate)
        + treatment_rate * (1 - treatment_rate)
    )
    numerator = (
        z_alpha * pooled_variance**0.5 + z_power * arm_variance**0.5
    ) ** 2
    return ceil(numerator / minimum_absolute_lift**2)
