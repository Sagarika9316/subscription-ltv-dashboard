import pandas as pd


def cohort_retention_matrix(df, asof, max_age=12, min_customers=1):
    required = {"source", "plan", "created_at", "ended_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if max_age < 0:
        raise ValueError("max_age must be non-negative.")
    if min_customers < 1:
        raise ValueError("min_customers must be positive.")

    asof = pd.Timestamp(asof)
    data = df[
        df["created_at"].notna()
        & (df["created_at"] <= asof)
        & df["source"].notna()
        & df["plan"].notna()
    ].copy()
    data["cohort_month"] = data["created_at"].dt.to_period("M").dt.to_timestamp()

    rows = []
    for (cohort_month, source, plan), cohort in data.groupby(
        ["cohort_month", "source", "plan"], sort=True
    ):
        cohort_size = len(cohort)
        if cohort_size < min_customers:
            continue

        for age in range(max_age + 1):
            checkpoint = cohort["created_at"] + pd.DateOffset(months=age)
            mature = checkpoint <= asof
            at_risk = int(mature.sum())
            if at_risk < min_customers:
                continue

            ended_at = cohort["ended_at"]
            retained = ended_at.isna() | (ended_at >= checkpoint)
            retained_count = int((retained & mature).sum())
            rows.append(
                {
                    "cohort_month": cohort_month,
                    "source": source,
                    "plan": plan,
                    "age_month": age,
                    "cohort_size": cohort_size,
                    "at_risk": at_risk,
                    "retained": retained_count,
                    "retention": retained_count / at_risk,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "cohort_month",
            "source",
            "plan",
            "age_month",
            "cohort_size",
            "at_risk",
            "retained",
            "retention",
        ],
    )
