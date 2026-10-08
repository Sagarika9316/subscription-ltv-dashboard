import numpy as np
import pandas as pd
import pytest

from app import build_eligibility
from cohort_models import cohort_churn_summary, cohort_retention_matrix
from experiment_models import (
    estimate_incremental_value_at_lift,
    two_proportion_sample_size,
)
from experiment_models import two_proportion_sample_size
from ltv_models import (
    add_ltv_cac_ratio,
    bootstrap_confidence_interval,
    estimate_observed_payback,
    rank_gross_ltv_cac_candidates,
    survival_ltv_forecast,
)
from survival_models import (
    aft_long_term_revenue_forecast,
    aft_survival_forecast,
    cox_survival_forecast,
    evaluate_survival_models,
    evaluate_survival_models_rolling,
    kaplan_meier_retention_curve,
)
from streamlit_app import build_marketing_action_table, compute_ltv_summary
from utils import (
    STATIC_CAC_BY_SOURCE,
    contribution_at,
    normalize_subscription_dataframe,
)


def test_experiment_sample_size_grows_for_smaller_minimum_lift():
    one_percentage_point = two_proportion_sample_size(0.05, 0.01)
    half_percentage_point = two_proportion_sample_size(0.05, 0.005)

    assert one_percentage_point == 8158
    assert half_percentage_point == 31234
    assert half_percentage_point > one_percentage_point


def test_experiment_sample_size_rejects_invalid_rates():
    with pytest.raises(ValueError, match="Baseline conversion rate"):
        two_proportion_sample_size(0, 0.01)
    with pytest.raises(ValueError, match="Minimum lift"):
        two_proportion_sample_size(0.99, 0.02)


def test_build_eligibility_works_without_read_only_errors():
    df = pd.DataFrame(
        {
            "plan": ["monthly", "annual"],
            "channel": ["direct", "paid_social"],
            "utm_campaign": [None, "spring_launch"],
            "created_at": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
            "ended_at": [pd.Timestamp("2023-02-15"), pd.NaT],
            "end_reason": ["voluntary", "payment_failed"],
        }
    )

    elig = build_eligibility(df)

    assert elig.shape == (2, 39)
    assert elig.dtype == bool
    assert elig.sum() >= 0


def test_compute_ltv_summary_uses_selected_statistic():
    df = pd.DataFrame(
        {
            "source": ["direct", "direct", "direct"],
            "plan": ["monthly", "monthly", "annual"],
            "step": [1, 1, 12],
            "price": [15, 15, 150],
            "created_at": [pd.Timestamp("2023-01-01")] * 3,
            "ended_at": [pd.NaT] * 3,
            "end_reason": ["voluntary"] * 3,
        }
    )

    mean_result, _, _ = compute_ltv_summary(
        df, asof=pd.Timestamp("2024-12-31"), selected_horizons=[1], statistic="mean"
    )
    median_result, _, _ = compute_ltv_summary(
        df, asof=pd.Timestamp("2024-12-31"), selected_horizons=[1], statistic="median"
    )

    mean_ltv = mean_result.query("plan == 'all'").iloc[0]["ltv"]
    median_ltv = median_result.query("plan == 'all'").iloc[0]["ltv"]
    assert mean_ltv == 60
    assert median_ltv == 15


def test_normalization_preserves_row_level_billing_values():
    df = pd.DataFrame(
        {
            "source": ["email", "paid"],
            "plan": ["monthly", "annual"],
            "billing_interval_months": [1, 12],
            "price": [19.5, 180],
            "currency": ["USD", "USD"],
            "net_contribution_per_charge": [8.5, 92],
            "created_at": ["2025-01-01", "2025-01-01"],
            "ended_at": [None, None],
            "end_reason": [None, None],
        }
    )

    normalized = normalize_subscription_dataframe(df)

    assert normalized["price"].tolist() == [19.5, 180.0]
    assert normalized["step"].tolist() == [1, 12]
    assert normalized["billing_interval_months"].tolist() == [1, 12]
    assert normalized["currency"].tolist() == ["USD", "USD"]
    assert normalized["net_contribution_per_charge"].tolist() == [8.5, 92.0]
    assert normalized.attrs["normalization_assumptions"] == []


def test_normalization_reports_plan_defaults_and_derives_source():
    df = pd.DataFrame(
        {
            "plan": ["monthly", "annual"],
            "channel": ["email", "paid_social"],
            "utm_campaign": [None, "spring"],
            "created_at": ["2025-01-01", "2025-01-01"],
            "ended_at": [None, None],
            "end_reason": [None, None],
        }
    )

    normalized = normalize_subscription_dataframe(df)

    assert normalized["source"].tolist() == ["email", "paid_social/spring"]
    assert normalized["step"].tolist() == [1, 12]
    assert normalized["billing_interval_months"].tolist() == [1, 12]
    assert normalized["price"].tolist() == [15.0, 150.0]
    assert normalized["currency"].tolist() == ["USD", "USD"]
    assert len(normalized.attrs["normalization_assumptions"]) == 4


def test_normalization_rejects_mixed_currencies():
    df = pd.DataFrame(
        {
            "source": ["email", "paid"],
            "plan": ["monthly", "monthly"],
            "price": [15, 15],
            "currency": ["USD", "GBP"],
            "created_at": ["2025-01-01", "2025-01-01"],
            "ended_at": [None, None],
            "end_reason": [None, None],
        }
    )

    with pytest.raises(ValueError, match="Only one currency is supported per upload"):
        normalize_subscription_dataframe(df)


def test_normalization_rejects_invalid_row_level_price():
    df = pd.DataFrame(
        {
            "source": ["email"],
            "plan": ["monthly"],
            "step": [1],
            "price": [-5],
            "created_at": ["2025-01-01"],
            "ended_at": [None],
            "end_reason": [None],
        }
    )

    with pytest.raises(ValueError, match="must not be negative"):
        normalize_subscription_dataframe(df)


def test_normalization_rejects_ended_subscription_without_reason():
    df = pd.DataFrame(
        {
            "source": ["email"],
            "plan": ["monthly"],
            "step": [1],
            "price": [15],
            "created_at": ["2025-01-01"],
            "ended_at": ["2025-02-01"],
            "end_reason": [None],
        }
    )

    with pytest.raises(ValueError, match="must have an 'end_reason'"):
        normalize_subscription_dataframe(df)


def test_contribution_at_uses_paid_charges_and_billing_schedule():
    df = pd.DataFrame(
        {
            "source": ["email", "paid"],
            "plan": ["monthly", "annual"],
            "step": [1, 12],
            "price": [15, 150],
            "net_contribution_per_charge": [8, 60],
        }
    )
    paid = np.zeros((2, 38), dtype=bool)
    paid[0, :3] = True
    paid[1, :2] = True

    result = contribution_at(df, paid, 3)

    assert result.tolist() == [24.0, 60.0]


def test_contribution_at_requires_explicit_cost_adjusted_values():
    df = pd.DataFrame({"step": [1], "price": [15]})
    paid = np.zeros((1, 38), dtype=bool)

    with pytest.raises(ValueError, match="requires a 'net_contribution_per_charge'"):
        contribution_at(df, paid, 1)


def test_bootstrap_confidence_interval_is_deterministic_and_handles_constant_values():
    interval = bootstrap_confidence_interval([7, 7, 7], n_bootstrap=100)

    assert interval == (7, 7)


def test_unit_economics_calculates_ltv_cac_and_observed_payback():
    df = pd.DataFrame(
        {
            "source": ["email"],
            "step": [1],
            "price": [15],
            "created_at": [pd.Timestamp("2020-01-01")],
        }
    )
    paid = np.ones((1, 38), dtype=bool)
    cac = pd.DataFrame({"source": ["email"], "cac": [30]})

    economics = add_ltv_cac_ratio(pd.DataFrame({"source": ["email"], "ltv": [45]}), cac)
    payback = estimate_observed_payback(
        df, paid, pd.Timestamp("2024-12-31"), [1, 3], cac
    )

    assert economics.loc[0, "ltv_cac"] == 1.5
    assert economics.loc[0, "ltv_minus_cac"] == 15
    assert payback.loc[0, "payback_months"] == 3


def test_gross_ltv_cac_candidates_rank_conservatively_and_exclude_missing_costs():
    summary = pd.DataFrame(
        {
            "source": ["high point", "strong lower bound", "unknown CAC"],
            "ltv": [180, 150, 300],
            "ci_low": [90, 120, 200],
            "cac": [40, 50, np.nan],
            "n": [1000, 200, 500],
        }
    )

    ranked = rank_gross_ltv_cac_candidates(summary)

    assert ranked["source"].tolist() == ["strong lower bound", "high point"]
    assert ranked.loc[0, "gross_ltv_cac"] == 3
    assert ranked.loc[0, "lower_bound_gross_ltv_cac"] == 2.4
    assert ranked.loc[0, "passes_gross_screen"]

    stricter = rank_gross_ltv_cac_candidates(summary, minimum_ratio=2.5)
    assert not stricter["passes_gross_screen"].any()


def test_marketing_action_table_separates_test_candidates_uncertainty_and_missing_costs():
    summary = pd.DataFrame(
        {
            "source": ["candidate", "uncertain", "below screen", "missing cost"],
            "ltv": [200, 200, 80, 100],
            "ci_low": [150, 90, 60, 80],
            "ci_high": [240, 260, 100, 120],
            "cac": [40, 40, 40, np.nan],
            "n": [100, 50, 20, 10],
        }
    )

    actions = build_marketing_action_table(
        summary, minimum_ratio=2.5, currency="USD"
    ).set_index("Source")

    assert "controlled test" in actions.loc["candidate", "Marketing next step"]
    assert "Uncertain" in actions.loc["uncertain", "Marketing next step"]
    assert "Below the current" in actions.loc["below screen", "Marketing next step"]
    assert "Add verified acquisition cost" in actions.loc[
        "missing cost", "Marketing next step"
    ]
    assert actions.loc["missing cost", "Acquisition cost"] == "Not entered"


def test_marketing_action_table_warns_when_cac_is_illustrative():
    summary = pd.DataFrame(
        {
            "source": ["email"],
            "ltv": [100],
            "ci_low": [80],
            "ci_high": [120],
            "cac": [20],
            "n": [100],
        }
    )

    actions = build_marketing_action_table(
        summary, minimum_ratio=1, currency="USD", illustrative_cac=True
    )

    assert "Replace example cost" in actions.loc[0, "Marketing next step"]


def test_static_cac_assumptions_are_available_by_acquisition_source():
    assert STATIC_CAC_BY_SOURCE["email"] == 20
    assert STATIC_CAC_BY_SOURCE["direct"] == 40
    assert STATIC_CAC_BY_SOURCE["organic_search"] == 30
    assert STATIC_CAC_BY_SOURCE["organic_social"] == 25
    assert STATIC_CAC_BY_SOURCE["paid_social/lookalike_subscribers"] == 60
    assert STATIC_CAC_BY_SOURCE["paid_social/prospecting_broad"] == 60


def test_experiment_sample_size_uses_two_arm_conversion_inputs():
    assert two_proportion_sample_size(0.05, 0.01) == 8158
    assert two_proportion_sample_size(0.05, 0.005) == 31234
    assert two_proportion_sample_size(0.05, 0.005) > two_proportion_sample_size(0.05, 0.01)


def test_experiment_sample_size_rejects_invalid_conversion_inputs():
    with pytest.raises(ValueError, match="Baseline conversion rate"):
        two_proportion_sample_size(0, 0.01)
    with pytest.raises(ValueError, match="Minimum lift"):
        two_proportion_sample_size(0.99, 0.02)


def test_experiment_economics_estimates_value_after_assumed_cac():
    result = estimate_incremental_value_at_lift(
        sample_per_arm=10000,
        absolute_lift=0.01,
        value_per_subscriber=200,
        cac_per_subscriber=30,
    )

    assert result["incremental_subscribers"] == 100
    assert result["incremental_value"] == 20000
    assert result["incremental_cac"] == 3000
    assert result["value_after_cac"] == 17000


def test_static_cac_assumptions_are_labeled_examples():
    assert STATIC_CAC_BY_SOURCE == {
        "email": 20,
        "direct": 40,
        "organic_search": 30,
        "organic_social": 25,
        "paid_social/lookalike_subscribers": 60,
        "paid_social/prospecting_broad": 60,
    }


def test_survival_forecast_uses_plan_billing_cadence_and_observed_support():
    asof = pd.Timestamp("2024-12-31")
    df = pd.DataFrame(
        {
            "source": ["email", "email"],
            "plan": ["monthly", "annual"],
            "step": [1, 12],
            "price": [15, 150],
            "created_at": [pd.Timestamp("2020-01-01")] * 2,
            "ended_at": [pd.NaT, pd.NaT],
        }
    )

    forecast = survival_ltv_forecast(df, asof, [1, 3, 12])
    three_month_ltv = forecast.loc[forecast["H"] == 3, "forecast_ltv"].iloc[0]

    assert three_month_ltv == 97.5

    recent = df.iloc[[0]].copy()
    recent["created_at"] = pd.Timestamp("2024-11-30")
    supported = survival_ltv_forecast(recent, asof, [1, 3])
    assert supported["H"].tolist() == [1]


def test_cox_survival_forecast_returns_horizon_ltv_by_source():
    rows = []
    created = pd.Timestamp("2020-01-01")
    for index in range(80):
        source = "email" if index % 2 == 0 else "paid"
        plan = "monthly" if index % 4 < 2 else "annual"
        ended = created if index == 0 else (
            created + pd.DateOffset(months=5 + index % 18)
            if index % 3 == 0
            else pd.NaT
        )
        rows.append(
            {
                "source": source,
                "plan": plan,
                "step": 1 if plan == "monthly" else 12,
                "price": 15 if plan == "monthly" else 150,
                "created_at": created,
                "ended_at": ended,
            }
        )

    result = cox_survival_forecast(
        pd.DataFrame(rows), pd.Timestamp("2024-12-31"), [1, 3, 12]
    )

    at_12_months = result[result["H"] == 12]
    assert set(at_12_months["source"]) == {"email", "paid"}
    assert (at_12_months["cox_ltv"] > 0).all()
    assert (at_12_months["n"] == 40).all()


def test_aft_survival_forecast_returns_horizon_ltv_by_source():
    rows = []
    created = pd.Timestamp("2020-01-01")
    for index in range(80):
        source = "email" if index % 2 == 0 else "paid"
        plan = "monthly" if index % 4 < 2 else "annual"
        ended = created + pd.DateOffset(months=5 + index % 18) if index % 3 == 0 else pd.NaT
        rows.append(
            {
                "source": source,
                "plan": plan,
                "step": 1 if plan == "monthly" else 12,
                "price": 15 if plan == "monthly" else 150,
                "created_at": created,
                "ended_at": ended,
            }
        )

    result = aft_survival_forecast(
        pd.DataFrame(rows), pd.Timestamp("2024-12-31"), [1, 3, 12]
    )

    at_12_months = result[result["H"] == 12]
    assert set(at_12_months["source"]) == {"email", "paid"}
    assert (at_12_months["aft_ltv"] > 0).all()
    assert (at_12_months["n"] == 40).all()


def test_long_term_aft_forecast_extrapolates_and_returns_parameter_interval():
    rows = []
    created = pd.Timestamp("2020-01-01")
    for index in range(120):
        source = "email" if index % 2 == 0 else "paid"
        plan = "monthly" if index % 4 < 2 else "annual"
        ended = (
            created + pd.DateOffset(months=6 + index % 30)
            if index % 3 == 0
            else pd.NaT
        )
        rows.append(
            {
                "source": source,
                "plan": plan,
                "step": 1 if plan == "monthly" else 12,
                "price": 15 if plan == "monthly" else 150,
                "created_at": created,
                "ended_at": ended,
            }
        )

    result = aft_long_term_revenue_forecast(
        pd.DataFrame(rows),
        asof=pd.Timestamp("2024-12-31"),
        horizon=60,
    )

    assert set(result["source"]) == {"email", "paid"}
    assert (result["forecast_ltv"] > 0).all()
    assert (result["ci_low"] <= result["forecast_ltv"]).all()
    assert (result["ci_high"] >= result["forecast_ltv"]).all()
    assert (result["n"] == 60).all()
    assert (result["extrapolation_months"] > 0).all()


def test_time_based_model_evaluation_scores_mature_holdout():
    rows = []
    train_start = pd.Timestamp("2024-01-01")
    holdout_dates = [pd.Timestamp("2024-08-01"), pd.Timestamp("2024-09-01")]
    for index in range(128):
        is_holdout = index >= 80
        source = "email" if index % 2 == 0 else "paid"
        plan = "monthly" if index % 4 < 2 else "annual"
        created = holdout_dates[(index // 2) % 2] if is_holdout else train_start
        has_ended = index % 3 == 0
        ended = created if index == 0 else (
            created + pd.DateOffset(months=2) if has_ended else pd.NaT
        )
        rows.append(
            {
                "source": source,
                "plan": plan,
                "step": 1 if plan == "monthly" else 12,
                "price": 15 if plan == "monthly" else 150,
                "created_at": created,
                "ended_at": ended,
                "end_reason": "voluntary" if has_ended else None,
            }
        )

    summary, by_source = evaluate_survival_models(
        pd.DataFrame(rows), pd.Timestamp("2024-12-31"), horizon=3
    )

    assert summary["train_n"] == 80
    assert summary["scored_n"] == 48
    assert summary["coverage"] == 1
    assert summary["cox_parameter_count"] > 0
    assert summary["cox_events_per_parameter"] > 0
    assert summary["ph_tests"]["p"].between(0, 1).all()
    assert summary["ph_tests"]["potential_violation"].equals(
        summary["ph_tests"]["p"] < 0.05
    )
    assert summary["km_mae"] >= 0
    assert summary["cox_mae"] >= 0
    assert summary["aft_mae"] >= 0
    assert summary["km_rmse"] >= 0
    assert summary["cox_rmse"] >= 0
    assert summary["aft_rmse"] >= 0
    assert len(by_source) == 2


def test_time_based_evaluation_rejects_uninformative_constant_holdout():
    rows = []
    train_start = pd.Timestamp("2022-01-01")
    test_start = pd.Timestamp("2023-08-01")
    for index in range(120):
        is_holdout = index >= 80
        source = "email" if index % 2 == 0 else "paid"
        if is_holdout:
            created = test_start + pd.DateOffset(days=index - 80)
            plan = "annual"
            ended = created + pd.DateOffset(months=6) if index % 3 == 0 else pd.NaT
        else:
            created = train_start
            plan = "monthly" if index % 4 < 2 else "annual"
            ended = created + pd.DateOffset(months=3 + index % 5) if index % 3 == 0 else pd.NaT
        rows.append(
            {
                "source": source,
                "plan": plan,
                "step": 1 if plan == "monthly" else 12,
                "price": 15 if plan == "monthly" else 150,
                "created_at": created,
                "ended_at": ended,
                "end_reason": "voluntary" if pd.notna(ended) else None,
            }
        )

    with pytest.raises(ValueError, match="zero MAE/RMSE is not informative"):
        evaluate_survival_models(
            pd.DataFrame(rows), pd.Timestamp("2025-01-03"), horizon=12
        )


def test_rolling_survival_evaluation_returns_non_overlapping_folds():
    rows = []
    for index in range(80):
        source = "email" if index % 2 == 0 else "paid"
        plan = "monthly" if index % 4 < 2 else "annual"
        ended = pd.Timestamp("2023-03-01") if index % 3 == 0 else pd.NaT
        rows.append(
            {
                "source": source,
                "plan": plan,
                "step": 1 if plan == "monthly" else 12,
                "price": 15 if plan == "monthly" else 150,
                "created_at": pd.Timestamp("2023-01-01"),
                "ended_at": ended,
                "end_reason": "voluntary" if pd.notna(ended) else None,
            }
        )
    for cohort_date in ["2024-01-15", "2024-04-15", "2024-07-15"]:
        created = pd.Timestamp(cohort_date)
        for index in range(48):
            source = "email" if index % 2 == 0 else "paid"
            plan = "monthly" if index % 4 < 2 else "annual"
            ended = created + pd.DateOffset(months=2) if index % 3 == 0 else pd.NaT
            rows.append(
                {
                    "source": source,
                    "plan": plan,
                    "step": 1 if plan == "monthly" else 12,
                    "price": 15 if plan == "monthly" else 150,
                    "created_at": created,
                    "ended_at": ended,
                    "end_reason": "voluntary" if pd.notna(ended) else None,
                }
            )

    folds = evaluate_survival_models_rolling(
        pd.DataFrame(rows), pd.Timestamp("2024-12-31"), horizon=3, n_folds=3
    )

    assert len(folds) == 3
    assert folds["status"].eq("complete").all()
    assert folds["holdout_start"].is_monotonic_increasing
    assert all(
        (pd.Timestamp(next_start) - pd.Timestamp(previous_end)).days == 1
        for previous_end, next_start in zip(
            folds["holdout_end"], folds["holdout_start"].iloc[1:]
        )
    )
    assert folds["scored_n"].eq(48).all()
    assert folds["aft_mae"].notna().all()
    assert folds["aft_rmse"].notna().all()


def test_cohort_retention_matrix_omits_immature_ages():
    df = pd.DataFrame(
        {
            "source": ["email"] * 4,
            "plan": ["monthly"] * 4,
            "created_at": [pd.Timestamp("2024-01-01")] * 3
            + [pd.Timestamp("2024-02-15")],
            "ended_at": [pd.Timestamp("2024-01-30"), pd.NaT, pd.NaT, pd.NaT],
        }
    )

    result = cohort_retention_matrix(df, pd.Timestamp("2024-03-01"), max_age=2)
    january = result[result["cohort_month"] == pd.Timestamp("2024-01-01")]
    february = result[result["cohort_month"] == pd.Timestamp("2024-02-01")]

    assert january.loc[january["age_month"] == 1, "retention"].iloc[0] == 2 / 3
    assert february["age_month"].tolist() == [0]


def test_cohort_matrix_splits_mature_churn_reasons_and_censors_active_users():
    created = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            "source": ["email"] * 4,
            "plan": ["monthly"] * 4,
            "created_at": [created] * 4,
            "ended_at": [
                pd.Timestamp("2024-01-30"),
                pd.Timestamp("2024-02-01"),
                pd.Timestamp("2024-02-15"),
                pd.NaT,
            ],
            "end_reason": ["voluntary", "payment_failed", "other", None],
        }
    )

    result = cohort_retention_matrix(df, pd.Timestamp("2024-03-01"), max_age=2)
    month_one = result[result["age_month"] == 1].iloc[0]
    month_two = result[result["age_month"] == 2].iloc[0]

    assert month_one["matured"] == 4
    assert month_one["retained"] == 3
    assert month_one["voluntary_churned"] == 1
    assert month_one["payment_failure_churned"] == 0
    assert month_two["churned"] == 3
    assert month_two["voluntary_churned"] == 1
    assert month_two["payment_failure_churned"] == 1
    assert month_two["other_churned"] == 1
    assert month_two["retained"] == 1


def test_standalone_cohort_churn_summary_has_stable_reason_columns():
    created = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            "source": ["email"] * 4,
            "plan": ["monthly"] * 4,
            "created_at": [created] * 4,
            "ended_at": [
                pd.Timestamp("2024-01-30"),
                pd.Timestamp("2024-02-01"),
                pd.Timestamp("2024-02-15"),
                pd.NaT,
            ],
            "end_reason": ["voluntary", "payment_failed", "other", None],
        }
    )

    result = cohort_churn_summary(df, pd.Timestamp("2024-03-01"), [1, 2])
    month_one = result[result["age_month"] == 1].iloc[0]
    month_two = result[result["age_month"] == 2].iloc[0]

    assert month_one["matured"] == 4
    assert month_one["churned"] == 1
    assert month_one["payment_failure_churned"] == 0
    assert month_two["churned"] == 3
    assert month_two["voluntary_churned"] == 1
    assert month_two["payment_failure_churned"] == 1
    assert month_two["other_churned"] == 1
    assert month_two["retention"] == 0.25


def test_kaplan_meier_retention_curve_censors_active_users_and_limits_followup():
    created = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            "source": ["email"] * 4,
            "plan": ["monthly"] * 4,
            "created_at": [created] * 4,
            "ended_at": [
                pd.Timestamp("2024-01-20"),
                pd.Timestamp("2024-02-10"),
                pd.NaT,
                pd.NaT,
            ],
        }
    )

    curve = kaplan_meier_retention_curve(
        df,
        pd.Timestamp("2024-03-10"),
        max_age=6,
        min_at_risk=2,
    )

    assert curve["age_month"].tolist() == [0, 1, 2]
    assert curve.loc[curve["age_month"] == 0, "retention"].iloc[0] == 1
    assert curve["retention"].between(0, 1).all()
    assert curve["ci_low"].le(curve["retention"]).all()
    assert curve["ci_high"].ge(curve["retention"]).all()
