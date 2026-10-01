import numpy as np
import pandas as pd

from app import build_eligibility
from ltv_models import (
    add_ltv_cac_ratio,
    bootstrap_confidence_interval,
    estimate_observed_payback,
    survival_ltv_forecast,
)
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
