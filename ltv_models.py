import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from utils import mature_at, revenue_at


def bootstrap_confidence_interval(
    values,
    statistic="mean",
    confidence=0.95,
    n_bootstrap=500,
    random_state=42,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if values.size == 0:
        return np.nan, np.nan

    reducer = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(random_state)
    estimates = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[index] = reducer(sample)

    tail = (1 - confidence) / 2
    lower, upper = np.quantile(estimates, [tail, 1 - tail])
    return float(lower), float(upper)


def source_ltv_confidence_intervals(
    df,
    paid,
    horizon,
    asof,
    statistic="mean",
    confidence=0.95,
    n_bootstrap=500,
    random_state=42,
):
    mature = mature_at(df, horizon, asof=asof)
    mature_df = df.loc[mature, ["source"]].copy()
    mature_df["ltv"] = revenue_at(df, paid, horizon)[mature]

    rows = []
    for source, group in mature_df.groupby("source"):
        lower, upper = bootstrap_confidence_interval(
            group["ltv"].to_numpy(),
            statistic=statistic,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        rows.append({"source": source, "ci_low": lower, "ci_high": upper})

    return pd.DataFrame(rows, columns=["source", "ci_low", "ci_high"])


def add_ltv_cac_ratio(ltv_by_source, cac_by_source):
    costs = cac_by_source[["source", "cac"]].copy()
    costs["cac"] = pd.to_numeric(costs["cac"], errors="coerce")
    costs = costs[costs["cac"] > 0].drop_duplicates("source", keep="last")

    result = ltv_by_source.merge(costs, on="source", how="left")
    result["ltv_cac"] = result["ltv"] / result["cac"]
    result["ltv_minus_cac"] = result["ltv"] - result["cac"]
    return result


def rank_gross_ltv_cac_candidates(source_summary, minimum_ratio=1.0):
    if not np.isfinite(minimum_ratio) or minimum_ratio <= 0:
        raise ValueError("minimum_ratio must be a positive finite number.")
    required = {"source", "ltv", "ci_low", "cac", "n"}
    missing = required - set(source_summary.columns)
    if missing:
        raise ValueError(f"Missing recommendation columns: {', '.join(sorted(missing))}")

    candidates = source_summary.copy()
    for column in ("ltv", "ci_low", "cac", "n"):
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
        candidates = candidates[np.isfinite(candidates[column])]
    candidates = candidates[
        (candidates["cac"] > 0)
        & candidates["ltv"].notna()
        & candidates["ci_low"].notna()
        & (candidates["n"] > 0)
    ].copy()
    candidates["gross_ltv_cac"] = candidates["ltv"] / candidates["cac"]
    candidates["lower_bound_gross_ltv_cac"] = candidates["ci_low"] / candidates["cac"]
    candidates["gross_ltv_minus_cac"] = candidates["ltv"] - candidates["cac"]
    candidates["passes_gross_screen"] = (
        candidates["lower_bound_gross_ltv_cac"] >= minimum_ratio
    )
    return candidates.sort_values(
        ["lower_bound_gross_ltv_cac", "n"],
        ascending=[False, False],
    ).reset_index(drop=True)


def estimate_observed_payback(df, paid, asof, horizons, cac_by_source):
    costs = cac_by_source[["source", "cac"]].copy()
    costs["cac"] = pd.to_numeric(costs["cac"], errors="coerce")
    costs = costs[costs["cac"] > 0].drop_duplicates("source", keep="last")

    revenue_rows = []
    for horizon in sorted(set(horizons)):
        mature = mature_at(df, horizon, asof=asof)
        if not mature.any():
            continue
        cohort = df.loc[mature, ["source"]].copy()
        cohort["revenue"] = revenue_at(df, paid, horizon)[mature]
        means = cohort.groupby("source")["revenue"].mean()
        revenue_rows.extend(
            {"source": source, "horizon": horizon, "mean_revenue": value}
            for source, value in means.items()
        )

    revenue = pd.DataFrame(revenue_rows, columns=["source", "horizon", "mean_revenue"])
    rows = []
    for source, cac in costs.itertuples(index=False, name=None):
        reached = revenue[
            (revenue["source"] == source) & (revenue["mean_revenue"] >= cac)
        ]
        payback = reached["horizon"].min() if not reached.empty else np.nan
        rows.append({"source": source, "payback_months": payback})

    return pd.DataFrame(rows, columns=["source", "payback_months"])


def survival_ltv_forecast(df, asof, horizons):
    columns = ["source", "H", "forecast_ltv", "n"]
    asof = pd.Timestamp(asof)
    observed = df[
        df["created_at"].notna() & (df["created_at"] <= asof)
    ].copy()
    if observed.empty:
        return pd.DataFrame(columns=columns)

    event = observed["ended_at"].notna() & (observed["ended_at"] <= asof)
    observed_end = observed["ended_at"].where(event, asof)
    observed["duration"] = (
        (observed_end - observed["created_at"]).dt.days / 30.44
    ).clip(lower=0)
    observed["event"] = event.astype(int)

    segment_counts = observed.groupby("source")["plan"].nunique()
    estimates = {}
    for (source, plan), group in observed.groupby(["source", "plan"]):
        max_followup = group["duration"].max()
        if pd.isna(max_followup):
            continue

        fitter = KaplanMeierFitter().fit(
            group["duration"],
            event_observed=group["event"],
        )
        step = int(group["step"].mode().iloc[0])
        price = float(group["price"].mean())
        for horizon in sorted(set(horizons)):
            if horizon > max_followup:
                continue
            billing_months = range(0, int(horizon), step)
            expected_revenue = sum(
                price if month == 0 else price * float(fitter.predict(month))
                for month in billing_months
            )
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
                "forecast_ltv": forecast,
                "n": sample_size,
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(["source", "H"])
