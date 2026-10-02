import numpy as np
import pandas as pd
import pytest

from app import build_eligibility
from ltv_models import (
    add_ltv_cac_ratio,
    bootstrap_confidence_interval,
    estimate_observed_payback,
    survival_ltv_forecast,
)
from survival_models import cox_survival_forecast
from survival_models import evaluate_survival_models
from streamlit_app import compute_ltv_summary


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
        ended = created + pd.DateOffset(months=2) if has_ended else pd.NaT
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
    assert summary["km_mae"] >= 0
    assert summary["cox_mae"] >= 0
    assert summary["km_rmse"] >= 0
    assert summary["cox_rmse"] >= 0
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
