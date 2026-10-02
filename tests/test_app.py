import numpy as np
import pandas as pd
import pytest

from app import build_eligibility
from cohort_models import cohort_retention_matrix
from ltv_models import (
    add_ltv_cac_ratio,
    bootstrap_confidence_interval,
    estimate_observed_payback,
    survival_ltv_forecast,
)
from survival_models import (
    aft_survival_forecast,
    cox_survival_forecast,
    evaluate_survival_models,
    evaluate_survival_models_rolling,
)
from streamlit_app import compute_ltv_summary
from utils import contribution_at, normalize_subscription_dataframe


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
            "step": [1, 12],
            "price": [19.5, 180],
            "net_contribution_per_charge": [8.5, 92],
            "created_at": ["2025-01-01", "2025-01-01"],
            "ended_at": [None, None],
            "end_reason": [None, None],
        }
    )

    normalized = normalize_subscription_dataframe(df)

    assert normalized["price"].tolist() == [19.5, 180.0]
    assert normalized["step"].tolist() == [1, 12]
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
    assert normalized["price"].tolist() == [15.0, 150.0]
    assert len(normalized.attrs["normalization_assumptions"]) == 3


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
