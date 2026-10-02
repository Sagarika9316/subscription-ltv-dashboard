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
            matured_count = int(mature.sum())
            if matured_count < min_customers:
                continue

            ended_at = cohort["ended_at"]
            retained = ended_at.isna() | (ended_at >= checkpoint)
            retained_count = int((retained & mature).sum())
            churned_mask = ended_at.notna() & (ended_at < checkpoint) & mature
            churned_count = int(churned_mask.sum())
            if "end_reason" in cohort:
                end_reason = cohort["end_reason"].fillna("").astype(str).str.strip().str.lower()
            else:
                end_reason = pd.Series("", index=cohort.index)
            voluntary_count = int((churned_mask & end_reason.eq("voluntary")).sum())
            payment_failure_count = int(
                (churned_mask & end_reason.eq("payment_failed")).sum()
            )
            other_count = churned_count - voluntary_count - payment_failure_count
            rows.append(
                {
                    "cohort_month": cohort_month,
                    "source": source,
                    "plan": plan,
                    "age_month": age,
                    "cohort_size": cohort_size,
                    "matured": matured_count,
                    "retained": retained_count,
                    "churned": churned_count,
                    "voluntary_churned": voluntary_count,
                    "payment_failure_churned": payment_failure_count,
                    "other_churned": other_count,
                    "retention": retained_count / matured_count,
                    "churn_rate": churned_count / matured_count,
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
            "matured",
            "retained",
            "churned",
            "voluntary_churned",
            "payment_failure_churned",
            "other_churned",
            "retention",
            "churn_rate",
        ],
    )


def cohort_churn_summary(df, asof, horizons=(1, 3, 6, 12), min_customers=1):
    required = {"source", "plan", "created_at", "ended_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if min_customers < 1:
        raise ValueError("min_customers must be positive.")

    columns = [
        "source",
        "plan",
        "age_month",
        "matured",
        "retained",
        "churned",
        "voluntary_churned",
        "payment_failure_churned",
        "other_churned",
        "retention",
        "churn_rate",
    ]
    asof = pd.Timestamp(asof)
    data = df[
        df["created_at"].notna()
        & (df["created_at"] <= asof)
        & df["source"].notna()
        & df["plan"].notna()
    ].copy()
    if "end_reason" in data:
        data["end_reason"] = (
            data["end_reason"].fillna("").astype(str).str.strip().str.lower()
        )
    else:
        data["end_reason"] = ""

    rows = []
    for (source, plan), group in data.groupby(["source", "plan"], sort=True):
        for age in sorted(set(horizons)):
            if age < 0:
                raise ValueError("horizons must be non-negative.")
            checkpoint = group["created_at"] + pd.DateOffset(months=age)
            mature = checkpoint <= asof
            matured = int(mature.sum())
            if matured < min_customers:
                continue

            ended = group["ended_at"]
            churned_mask = ended.notna() & (ended < checkpoint) & mature
            churned = int(churned_mask.sum())
            voluntary = int(
                (churned_mask & group["end_reason"].eq("voluntary")).sum()
            )
            payment_failure = int(
                (churned_mask & group["end_reason"].eq("payment_failed")).sum()
            )
            retained = matured - churned
            rows.append(
                {
                    "source": source,
                    "plan": plan,
                    "age_month": age,
                    "matured": matured,
                    "retained": retained,
                    "churned": churned,
                    "voluntary_churned": voluntary,
                    "payment_failure_churned": payment_failure,
                    "other_churned": churned - voluntary - payment_failure,
                    "retention": retained / matured,
                    "churn_rate": churned / matured,
                }
            )

    return pd.DataFrame(rows, columns=columns)
