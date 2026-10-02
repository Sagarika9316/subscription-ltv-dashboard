import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from utils import build_eligibility, mature_at, revenue_at


def cox_survival_forecast(df, asof, horizons, penalizer=0.1):
    columns = ["source", "H", "cox_ltv", "n"]
    asof = pd.Timestamp(asof)
    required = ["source", "plan", "step", "price", "created_at", "ended_at"]
    observed = df.dropna(subset=[column for column in required if column != "ended_at"]).copy()
    observed = observed[observed["created_at"] <= asof]
    if observed.empty:
        raise ValueError("No complete customer records are available for the Cox model.")

    event = observed["ended_at"].notna() & (observed["ended_at"] <= asof)
    observed_end = observed["ended_at"].where(event, asof)
    observed["duration"] = (
        (observed_end - observed["created_at"]).dt.days / 30.44
    ).clip(lower=0)
    observed["event"] = event.astype(int)

    if observed["event"].sum() < 2:
        raise ValueError("The Cox model requires at least two observed cancellations.")
    if observed["source"].nunique() < 2 and observed["plan"].nunique() < 2:
        raise ValueError("The Cox model requires variation in source or plan.")

    model = CoxPHFitter(penalizer=penalizer)
    model.fit(
        observed[["duration", "event", "source", "plan"]],
        duration_col="duration",
        event_col="event",
        formula="C(source) + C(plan)",
    )

    segment_counts = observed.groupby("source")["plan"].nunique()
    estimates = {}
    for (source, plan), group in observed.groupby(["source", "plan"]):
        max_followup = group["duration"].max()
        step = int(group["step"].mode().iloc[0])
        price = float(group["price"].mean())
        profile = pd.DataFrame({"source": [source], "plan": [plan]})

        for horizon in sorted(set(horizons)):
            if horizon > max_followup:
                continue
            billing_months = list(range(0, int(horizon), step))
            renewal_months = [month for month in billing_months if month > 0]
            expected_revenue = price
            if renewal_months:
                survival = model.predict_survival_function(
                    profile,
                    times=renewal_months,
                ).iloc[:, 0]
                expected_revenue += price * float(survival.sum())
            estimates.setdefault((source, horizon), []).append(
                (expected_revenue, len(group))
            )

    rows = []
    for (source, horizon), segments in estimates.items():
        if len(segments) != segment_counts.loc[source]:
            continue
        sample_size = sum(count for _, count in segments)
        forecast = sum(value * count for value, count in segments) / sample_size
        rows.append(
            {
                "source": source,
                "H": horizon,
                "cox_ltv": forecast,
                "n": sample_size,
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(["source", "H"])


def evaluate_survival_models(df, asof, horizon, n_bootstrap=500):
    asof = pd.Timestamp(asof)
    test_end = asof - pd.DateOffset(months=horizon) + pd.DateOffset(days=1)
    test_start = test_end - pd.DateOffset(months=horizon) + pd.DateOffset(days=1)
    train_cutoff = test_start - pd.DateOffset(days=1)

    train = df[df["created_at"] <= train_cutoff].copy()
    test = df[
        (df["created_at"] >= test_start)
        & (df["created_at"] <= test_end)
    ].copy()
    test = test.loc[mature_at(test, horizon, asof=asof)].copy()
    if train.empty or test.empty:
        raise ValueError("Not enough history for a mature time-based holdout at this horizon.")

    train_event = train["ended_at"].notna() & (train["ended_at"] <= train_cutoff)
    train_end = train["ended_at"].where(train_event, train_cutoff)
    train["duration"] = (
        (train_end - train["created_at"]).dt.days / 30.44
    ).clip(lower=0)
    train["event"] = train_event.astype(int)

    formula_terms = []
    if train["source"].nunique() > 1:
        formula_terms.append("C(source)")
    if train["plan"].nunique() > 1:
        formula_terms.append("C(plan)")
    if not formula_terms:
        raise ValueError("Cox comparison requires variation in source or plan.")

    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(
        train[["duration", "event", "source", "plan"]],
        duration_col="duration",
        event_col="event",
        formula=" + ".join(formula_terms),
    )
    cox_parameter_count = len(cox.params_)
    cox_events_per_parameter = int(train["event"].sum()) / max(cox_parameter_count, 1)

    eligibility = build_eligibility(test, asof=asof)
    paid = eligibility[:, :38].copy()
    failed = test["end_reason"].eq("payment_failed").to_numpy()
    paid[failed] = eligibility[failed, :38] & eligibility[failed, 1:39]
    test["actual"] = revenue_at(test, paid, horizon)
    test["km_prediction"] = np.nan
    test["cox_prediction"] = np.nan

    for (source, plan), holdout in test.groupby(["source", "plan"]):
        history = train[(train["source"] == source) & (train["plan"] == plan)]
        if history.empty or history["duration"].max() < horizon:
            continue

        km = KaplanMeierFitter().fit(
            history["duration"],
            event_observed=history["event"],
        )
        step = int(holdout["step"].mode().iloc[0])
        renewal_months = list(range(step, int(horizon), step))
        km_survival = km.predict(renewal_months).sum() if renewal_months else 0.0
        profile = pd.DataFrame({"source": [source], "plan": [plan]})
        cox_survival = (
            cox.predict_survival_function(profile, times=renewal_months)
            .iloc[:, 0]
            .sum()
            if renewal_months
            else 0.0
        )
        test.loc[holdout.index, "km_prediction"] = holdout["price"].to_numpy() * (
            1 + km_survival
        )
        test.loc[holdout.index, "cox_prediction"] = holdout["price"].to_numpy() * (
            1 + cox_survival
        )

    scored = test.dropna(subset=["km_prediction", "cox_prediction"]).copy()
    if scored.empty:
        raise ValueError("No holdout source-plan groups have sufficient training follow-up.")

    actual = scored["actual"].to_numpy()
    km_error = scored["km_prediction"].to_numpy() - actual
    cox_error = scored["cox_prediction"].to_numpy() - actual
    if (
        np.ptp(actual) <= 1e-9
        and np.allclose(km_error, 0)
        and np.allclose(cox_error, 0)
    ):
        raise ValueError(
            "Holdout revenue is constant and both models predict it exactly, so zero MAE/RMSE is not informative. "
            "Choose a plan and horizon that include renewal revenue or customer-level variation."
        )
    delta_abs_error = np.abs(cox_error) - np.abs(km_error)

    monthly = scored.groupby(scored["created_at"].dt.to_period("M"))["actual"].agg(
        count="size"
    )
    delta_by_month = pd.Series(delta_abs_error, index=scored.index).groupby(
        scored["created_at"].dt.to_period("M")
    ).sum()
    rng = np.random.default_rng(42)
    month_count = len(monthly)
    month_indices = rng.integers(0, month_count, size=(n_bootstrap, month_count))
    month_counts = monthly["count"].to_numpy()
    month_deltas = delta_by_month.reindex(monthly.index).to_numpy()
    bootstrap_delta = (
        month_deltas[month_indices].sum(axis=1)
        / month_counts[month_indices].sum(axis=1)
    )
    delta_ci_low, delta_ci_high = np.quantile(bootstrap_delta, [0.025, 0.975])

    summary = {
        "horizon": horizon,
        "train_cutoff": train_cutoff,
        "test_start": test["created_at"].min(),
        "test_end": test["created_at"].max(),
        "train_n": len(train),
        "train_events": int(train["event"].sum()),
        "cox_parameter_count": cox_parameter_count,
        "cox_events_per_parameter": cox_events_per_parameter,
        "holdout_n": len(test),
        "scored_n": len(scored),
        "coverage": len(scored) / len(test),
        "actual_mean": float(actual.mean()),
        "km_mae": float(np.abs(km_error).mean()),
        "cox_mae": float(np.abs(cox_error).mean()),
        "km_rmse": float(np.sqrt(np.mean(km_error**2))),
        "cox_rmse": float(np.sqrt(np.mean(cox_error**2))),
        "km_bias": float(km_error.mean()),
        "cox_bias": float(cox_error.mean()),
        "cox_minus_km_mae": float(delta_abs_error.mean()),
        "delta_mae_ci_low": float(delta_ci_low),
        "delta_mae_ci_high": float(delta_ci_high),
    }

    source_rows = []
    for source, group in scored.groupby("source"):
        source_rows.append(
            {
                "source": source,
                "customers": len(group),
                "actual_ltv": group["actual"].mean(),
                "km_ltv": group["km_prediction"].mean(),
                "cox_ltv": group["cox_prediction"].mean(),
                "km_mae": np.abs(group["km_prediction"] - group["actual"]).mean(),
                "cox_mae": np.abs(group["cox_prediction"] - group["actual"]).mean(),
            }
        )
    by_source = pd.DataFrame(source_rows).sort_values("source")
    return summary, by_source
