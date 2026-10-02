import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


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
