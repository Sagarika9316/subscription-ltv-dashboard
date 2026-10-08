import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import proportional_hazard_test

from utils import build_eligibility, mature_at, revenue_at


def kaplan_meier_retention_curve(
    df,
    asof,
    max_age=36,
    min_at_risk=1,
    confidence=0.95,
):
    required = {"source", "plan", "created_at", "ended_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if max_age < 0:
        raise ValueError("max_age must be non-negative.")
    if min_at_risk < 1:
        raise ValueError("min_at_risk must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    asof = pd.Timestamp(asof)
    observed = df[
        df["created_at"].notna() & (df["created_at"] <= asof)
    ].copy()
    columns = ["source", "plan", "age_month", "at_risk", "retention", "ci_low", "ci_high"]
    if observed.empty:
        return pd.DataFrame(columns=columns)

    event = observed["ended_at"].notna() & (observed["ended_at"] <= asof)
    observed_end = observed["ended_at"].where(event, asof)
    observed["duration"] = (
        (observed_end - observed["created_at"]).dt.days / 30.44
    ).clip(lower=0)
    observed["event"] = event.astype(int)
    ages = np.arange(max_age + 1)

    rows = []
    for (source, plan), group in observed.groupby(["source", "plan"], sort=True):
        at_risk_counts = (
            group["duration"].to_numpy()[:, None] >= ages[None, :]
        ).sum(axis=0)
        if at_risk_counts.max(initial=0) < min_at_risk:
            continue

        fitter = KaplanMeierFitter(alpha=1 - confidence).fit(
            group["duration"],
            event_observed=group["event"],
            timeline=ages,
            ci_labels=["ci_low", "ci_high"],
        )
        confidence_intervals = fitter.confidence_interval_survival_function_
        for age, at_risk in zip(ages, at_risk_counts):
            if at_risk < min_at_risk:
                continue
            rows.append(
                {
                    "source": source,
                    "plan": plan,
                    "age_month": int(age),
                    "at_risk": int(at_risk),
                    "retention": float(fitter.predict(age)),
                    "ci_low": float(confidence_intervals.loc[age, "ci_low"]),
                    "ci_high": float(confidence_intervals.loc[age, "ci_high"]),
                }
            )

    return pd.DataFrame(rows, columns=columns)


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


def aft_survival_forecast(df, asof, horizons, penalizer=0.1):
    columns = ["source", "H", "aft_ltv", "n"]
    asof = pd.Timestamp(asof)
    required = ["source", "plan", "step", "price", "created_at", "ended_at"]
    observed = df.dropna(
        subset=[column for column in required if column != "ended_at"]
    ).copy()
    observed = observed[observed["created_at"] <= asof]
    if observed.empty:
        raise ValueError("No complete customer records are available for the AFT model.")

    event = observed["ended_at"].notna() & (observed["ended_at"] <= asof)
    observed_end = observed["ended_at"].where(event, asof)
    observed["duration"] = (
        (observed_end - observed["created_at"]).dt.days / 30.44
    ).clip(lower=1 / 30.44)
    observed["event"] = event.astype(int)

    if observed["event"].sum() < 2:
        raise ValueError("The AFT model requires at least two observed cancellations.")

    formula_terms = []
    if observed["source"].nunique() > 1:
        formula_terms.append("C(source)")
    if observed["plan"].nunique() > 1:
        formula_terms.append("C(plan)")
    if not formula_terms:
        raise ValueError("AFT comparison requires variation in source or plan.")

    model = WeibullAFTFitter(penalizer=penalizer)
    model.fit(
        observed[["duration", "event", "source", "plan"]],
        duration_col="duration",
        event_col="event",
        formula=" + ".join(formula_terms),
    )

    segment_counts = observed.groupby("source")["plan"].nunique()
    estimates = {}
    for (source, plan), group in observed.groupby(["source", "plan"]):
        if group["duration"].max() < min(horizons):
            continue
        step = int(group["step"].mode().iloc[0])
        price = float(group["price"].mean())
        profile = pd.DataFrame({"source": [source], "plan": [plan]})

        for horizon in sorted(set(horizons)):
            if horizon > group["duration"].max():
                continue
            renewal_months = list(range(step, int(horizon), step))
            survival = (
                model.predict_survival_function(profile, times=renewal_months)
                .iloc[:, 0]
                .sum()
                if renewal_months
                else 0.0
            )
            expected_revenue = price * (1 + survival)
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
            {"source": source, "H": horizon, "aft_ltv": forecast, "n": sample_size}
        )

    return pd.DataFrame(rows, columns=columns).sort_values(["source", "H"])


def aft_long_term_revenue_forecast(
    df,
    asof,
    horizon,
    confidence=0.95,
    penalizer=0.1,
    n_parameter_draws=1000,
    random_state=42,
):
    """Estimate source-level gross revenue through a finite Weibull AFT horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if n_parameter_draws < 2:
        raise ValueError("n_parameter_draws must be at least 2.")

    required = {"source", "plan", "step", "price", "created_at", "ended_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    asof = pd.Timestamp(asof)
    observed = df[
        df["created_at"].notna() & (df["created_at"] <= asof)
    ].copy()
    columns = [
        "source",
        "forecast_ltv",
        "ci_low",
        "ci_high",
        "n",
        "max_observed_followup_months",
        "extrapolation_months",
    ]
    if observed.empty:
        return pd.DataFrame(columns=columns)

    event = observed["ended_at"].notna() & (observed["ended_at"] <= asof)
    observed_end = observed["ended_at"].where(event, asof)
    observed["duration"] = (
        (observed_end - observed["created_at"]).dt.days / 30.44
    ).clip(lower=1 / 30.44)
    observed["event"] = event.astype(int)

    if (observed["step"] <= 0).any() or (observed["price"] < 0).any():
        raise ValueError("Billing intervals must be positive and prices non-negative.")
    if observed["event"].sum() < 2:
        raise ValueError("The Weibull model requires at least two observed cancellations.")

    formula_terms = []
    if observed["source"].nunique() > 1:
        formula_terms.append("C(source)")
    if observed["plan"].nunique() > 1:
        formula_terms.append("C(plan)")
    if not formula_terms:
        raise ValueError("The Weibull forecast requires variation in source or plan.")

    model = WeibullAFTFitter(penalizer=penalizer)
    model.fit(
        observed[["duration", "event", "source", "plan"]],
        duration_col="duration",
        event_col="event",
        formula=" + ".join(formula_terms),
    )
    parameter_names = list(model.params_.index)
    parameters = model.params_.to_numpy(dtype=float)
    covariance = model.variance_matrix_.loc[
        parameter_names, parameter_names
    ].to_numpy(dtype=float)
    if not np.isfinite(parameters).all() or not np.isfinite(covariance).all():
        raise ValueError("Weibull parameter uncertainty is unavailable.")
    covariance = (covariance + covariance.T) / 2
    rng = np.random.default_rng(random_state)
    parameter_draws = rng.multivariate_normal(
        parameters,
        covariance,
        size=n_parameter_draws,
    )
    lambda_names = [name for name in parameter_names if name[0] == "lambda_"]
    rho_names = [name for name in parameter_names if name[0] == "rho_"]
    lambda_positions = [parameter_names.index(name) for name in lambda_names]
    rho_positions = [parameter_names.index(name) for name in rho_names]
    source_draws = {}
    source_points = {}
    source_counts = {}
    source_followup = {}

    for (source, plan), group in observed.groupby(["source", "plan"], sort=True):
        profile = pd.DataFrame({"source": [source], "plan": [plan]})
        design = model.regressors.transform_df(profile).iloc[0]
        lambda_design = design.loc["lambda_"].reindex(
            [name[1] for name in lambda_names]
        ).to_numpy(dtype=float)
        rho_design = design.loc["rho_"].reindex(
            [name[1] for name in rho_names]
        ).to_numpy(dtype=float)
        step = int(group["step"].mode().iloc[0])
        price = float(group["price"].mean())
        renewal_months = np.arange(step, int(horizon), step, dtype=float)

        def expected_revenue(draws):
            if renewal_months.size == 0:
                return np.full(len(draws), price)
            log_scale = draws[:, lambda_positions] @ lambda_design
            log_shape = draws[:, rho_positions] @ rho_design
            shape = np.exp(np.clip(log_shape, -50, 50))
            log_hazard = shape[:, None] * (
                np.log(renewal_months)[None, :] - log_scale[:, None]
            )
            survival = np.exp(-np.exp(np.clip(log_hazard, -745, 709)))
            return price * (1 + survival.sum(axis=1))

        point = expected_revenue(parameters[None, :])[0]
        draws = expected_revenue(parameter_draws)
        count = len(group)
        source_draws[source] = source_draws.get(
            source, np.zeros(n_parameter_draws)
        ) + count * draws
        source_points[source] = source_points.get(source, 0.0) + count * point
        source_counts[source] = source_counts.get(source, 0) + count
        source_followup[source] = min(
            source_followup.get(source, float("inf")),
            float(group["duration"].max()),
        )

    rows = []
    for source in sorted(source_draws):
        forecast = source_points[source] / source_counts[source]
        combined_draws = source_draws[source] / source_counts[source]
        ci_low, ci_high = np.quantile(
            combined_draws,
            [(1 - confidence) / 2, 1 - (1 - confidence) / 2],
        )
        max_observed = source_followup[source]
        rows.append(
            {
                "source": source,
                "forecast_ltv": forecast,
                "ci_low": max(0.0, float(ci_low)),
                "ci_high": float(ci_high),
                "n": source_counts[source],
                "max_observed_followup_months": max_observed,
                "extrapolation_months": max(0.0, horizon - max_observed),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def evaluate_survival_models(
    df,
    asof,
    horizon,
    n_bootstrap=500,
    holdout_start=None,
    holdout_end=None,
):
    asof = pd.Timestamp(asof)
    if (holdout_start is None) != (holdout_end is None):
        raise ValueError("Provide both holdout_start and holdout_end, or neither.")
    if holdout_start is None:
        test_end = asof - pd.DateOffset(months=horizon) + pd.DateOffset(days=1)
        test_start = test_end - pd.DateOffset(months=horizon) + pd.DateOffset(days=1)
    else:
        test_start = pd.Timestamp(holdout_start)
        test_end = pd.Timestamp(holdout_end)
        if test_start > test_end:
            raise ValueError("holdout_start must be on or before holdout_end.")
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
    aft = WeibullAFTFitter(penalizer=0.1)
    aft_train = train.copy()
    aft_train["duration"] = aft_train["duration"].clip(lower=1 / 30.44)
    aft.fit(
        aft_train[["duration", "event", "source", "plan"]],
        duration_col="duration",
        event_col="event",
        formula=" + ".join(formula_terms),
    )
    cox_parameter_count = len(cox.params_)
    cox_events_per_parameter = int(train["event"].sum()) / max(cox_parameter_count, 1)
    ph_test = proportional_hazard_test(
        cox,
        train[["duration", "event", "source", "plan"]],
        time_transform="rank",
    )
    ph_tests = ph_test.summary.reset_index()
    ph_tests = ph_tests.rename(columns={ph_tests.columns[0]: "predictor"})
    ph_tests["potential_violation"] = ph_tests["p"] < 0.05

    eligibility = build_eligibility(test, asof=asof)
    paid = eligibility[:, :38].copy()
    failed = test["end_reason"].eq("payment_failed").to_numpy()
    paid[failed] = eligibility[failed, :38] & eligibility[failed, 1:39]
    test["actual"] = revenue_at(test, paid, horizon)
    test["km_prediction"] = np.nan
    test["cox_prediction"] = np.nan
    test["aft_prediction"] = np.nan

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
        aft_survival = (
            aft.predict_survival_function(profile, times=renewal_months)
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
        test.loc[holdout.index, "aft_prediction"] = holdout["price"].to_numpy() * (
            1 + aft_survival
        )

    scored = test.dropna(
        subset=["km_prediction", "cox_prediction", "aft_prediction"]
    ).copy()
    if scored.empty:
        raise ValueError("No holdout source-plan groups have sufficient training follow-up.")

    actual = scored["actual"].to_numpy()
    km_error = scored["km_prediction"].to_numpy() - actual
    cox_error = scored["cox_prediction"].to_numpy() - actual
    aft_error = scored["aft_prediction"].to_numpy() - actual
    if (
        np.ptp(actual) <= 1e-9
        and np.allclose(km_error, 0)
        and np.allclose(cox_error, 0)
        and np.allclose(aft_error, 0)
    ):
        raise ValueError(
            "Holdout revenue is constant and all models predict it exactly, so zero MAE/RMSE is not informative. "
            "Choose a plan and horizon that include renewal revenue or customer-level variation."
        )
    cox_delta_abs_error = np.abs(cox_error) - np.abs(km_error)
    aft_delta_abs_error = np.abs(aft_error) - np.abs(km_error)

    monthly = scored.groupby(scored["created_at"].dt.to_period("M"))["actual"].agg(
        count="size"
    )
    delta_by_month = pd.DataFrame(
        {
            "cox": cox_delta_abs_error,
            "aft": aft_delta_abs_error,
            "month": scored["created_at"].dt.to_period("M").to_numpy(),
        },
        index=scored.index,
    ).groupby("month")[["cox", "aft"]].sum()
    rng = np.random.default_rng(42)
    month_count = len(monthly)
    month_indices = rng.integers(0, month_count, size=(n_bootstrap, month_count))
    month_counts = monthly["count"].to_numpy()
    month_deltas = delta_by_month.reindex(monthly.index).to_numpy()
    bootstrap_deltas = month_deltas[month_indices].sum(axis=1) / month_counts[
        month_indices
    ].sum(axis=1)[:, None]
    cox_delta_ci_low, cox_delta_ci_high = np.quantile(
        bootstrap_deltas[:, 0], [0.025, 0.975]
    )
    aft_delta_ci_low, aft_delta_ci_high = np.quantile(
        bootstrap_deltas[:, 1], [0.025, 0.975]
    )

    summary = {
        "horizon": horizon,
        "train_cutoff": train_cutoff,
        "test_start": test["created_at"].min(),
        "test_end": test["created_at"].max(),
        "train_n": len(train),
        "train_events": int(train["event"].sum()),
        "cox_parameter_count": cox_parameter_count,
        "cox_events_per_parameter": cox_events_per_parameter,
        "ph_tests": ph_tests,
        "ph_violation_count": int(ph_tests["potential_violation"].sum()),
        "holdout_n": len(test),
        "scored_n": len(scored),
        "coverage": len(scored) / len(test),
        "actual_mean": float(actual.mean()),
        "km_mae": float(np.abs(km_error).mean()),
        "cox_mae": float(np.abs(cox_error).mean()),
        "aft_mae": float(np.abs(aft_error).mean()),
        "km_rmse": float(np.sqrt(np.mean(km_error**2))),
        "cox_rmse": float(np.sqrt(np.mean(cox_error**2))),
        "aft_rmse": float(np.sqrt(np.mean(aft_error**2))),
        "km_bias": float(km_error.mean()),
        "cox_bias": float(cox_error.mean()),
        "aft_bias": float(aft_error.mean()),
        "cox_minus_km_mae": float(cox_delta_abs_error.mean()),
        "cox_delta_mae_ci_low": float(cox_delta_ci_low),
        "cox_delta_mae_ci_high": float(cox_delta_ci_high),
        "aft_minus_km_mae": float(aft_delta_abs_error.mean()),
        "aft_delta_mae_ci_low": float(aft_delta_ci_low),
        "aft_delta_mae_ci_high": float(aft_delta_ci_high),
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
                "aft_ltv": group["aft_prediction"].mean(),
                "km_mae": np.abs(group["km_prediction"] - group["actual"]).mean(),
                "cox_mae": np.abs(group["cox_prediction"] - group["actual"]).mean(),
                "aft_mae": np.abs(group["aft_prediction"] - group["actual"]).mean(),
            }
        )
    by_source = pd.DataFrame(source_rows).sort_values("source")
    return summary, by_source


def evaluate_survival_models_rolling(df, asof, horizon, n_folds=3):
    if n_folds < 1:
        raise ValueError("n_folds must be positive.")

    asof = pd.Timestamp(asof)
    latest_end = asof - pd.DateOffset(months=horizon) + pd.DateOffset(days=1)
    rows = []
    for fold_index in range(n_folds):
        periods_back = n_folds - fold_index - 1
        holdout_end = latest_end - pd.DateOffset(months=periods_back * horizon)
        holdout_start = (
            holdout_end - pd.DateOffset(months=horizon) + pd.DateOffset(days=1)
        )
        row = {
            "fold": fold_index + 1,
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "status": "complete",
            "error": "",
        }
        try:
            summary, _ = evaluate_survival_models(
                df,
                asof,
                horizon,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
            )
            row.update(
                {
                    "train_cutoff": summary["train_cutoff"],
                    "train_n": summary["train_n"],
                    "train_events": summary["train_events"],
                    "holdout_n": summary["holdout_n"],
                    "scored_n": summary["scored_n"],
                    "coverage": summary["coverage"],
                    "km_mae": summary["km_mae"],
                    "cox_mae": summary["cox_mae"],
                    "aft_mae": summary["aft_mae"],
                    "km_rmse": summary["km_rmse"],
                    "cox_rmse": summary["cox_rmse"],
                    "aft_rmse": summary["aft_rmse"],
                    "cox_minus_km_mae": summary["cox_minus_km_mae"],
                    "cox_delta_mae_ci_low": summary["cox_delta_mae_ci_low"],
                    "cox_delta_mae_ci_high": summary["cox_delta_mae_ci_high"],
                    "aft_minus_km_mae": summary["aft_minus_km_mae"],
                    "aft_delta_mae_ci_low": summary["aft_delta_mae_ci_low"],
                    "aft_delta_mae_ci_high": summary["aft_delta_mae_ci_high"],
                    "cox_events_per_parameter": summary[
                        "cox_events_per_parameter"
                    ],
                }
            )
        except (ConvergenceError, ValueError) as error:
            row["status"] = "unavailable"
            row["error"] = str(error)
        rows.append(row)

    return pd.DataFrame(rows)
