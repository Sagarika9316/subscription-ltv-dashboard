import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-06-30")
K = 38  # anniversaries 0..37, enough for 36 months of monthly billing
HORIZONS = [1, 3, 6, 12, 24, 36]

DEFAULT_BILLING = {"monthly": (1, 15.0), "annual": (12, 150.0)}


def normalize_subscription_dataframe(df):
    df = df.copy()
    required = {"plan", "created_at", "ended_at", "end_reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required subscription columns: {', '.join(sorted(missing))}")

    assumptions = []
    df["plan"] = df["plan"].astype("string").str.strip().str.lower()
    if df["plan"].isna().any() or df["plan"].eq("").any():
        raise ValueError("The 'plan' column must not contain blank values.")

    for column in ("created_at", "canceled_at", "ended_at"):
        if column not in df:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        invalid = df[column].notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"The '{column}' column contains invalid dates.")
        df[column] = parsed
    if df["created_at"].isna().any():
        raise ValueError("The 'created_at' column must not contain blank values.")
    df["end_reason"] = df["end_reason"].map(
        lambda value: value.strip().lower() if isinstance(value, str) else value
    )
    ended = df["ended_at"].notna()
    missing_reason = ended & df["end_reason"].isna()
    if missing_reason.any():
        raise ValueError("Ended subscriptions must have an 'end_reason'.")
    allowed_reasons = {"voluntary", "payment_failed"}
    unknown_reasons = set(df.loc[ended, "end_reason"].dropna().unique()) - allowed_reasons
    if unknown_reasons:
        labels = ", ".join(sorted(unknown_reasons))
        raise ValueError(
            f"Unsupported end_reason values: {labels}. Expected 'voluntary' or 'payment_failed'."
        )

    if "source" not in df:
        if "channel" not in df:
            raise ValueError("Provide either a 'source' column or a 'channel' column.")
        campaign = df.get("utm_campaign", pd.Series("", index=df.index)).fillna("").astype(str)
        df["source"] = np.where(
            df["channel"].eq("paid_social"),
            "paid_social/" + campaign,
            df["channel"],
        )
        assumptions.append("Acquisition source was derived from channel and campaign.")
    df["source"] = df["source"].astype("string").str.strip()
    if df["source"].isna().any() or df["source"].eq("").any():
        raise ValueError("The 'source' column must not contain blank values.")

    for column, default_index, label in (
        ("step", 0, "billing interval"),
        ("price", 1, "subscription price"),
    ):
        if column not in df:
            defaults = {
                plan: billing[default_index]
                for plan, billing in DEFAULT_BILLING.items()
            }
            values = df["plan"].map(defaults)
            if values.isna().any():
                unknown = sorted(df.loc[values.isna(), "plan"].unique().tolist())
                raise ValueError(
                    f"Provide '{column}' for plans without a built-in {label}: {', '.join(unknown)}."
                )
            df[column] = values
            if column == "step":
                assumptions.append(
                    "Billing intervals defaulted by plan (monthly=1, annual=12 months)."
                )
            else:
                assumptions.append(
                    "Prices defaulted by plan (monthly=$15, annual=$150)."
                )

        original = df[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any() or numeric.isna().any():
            raise ValueError(f"The '{column}' column must contain numeric values for every subscription.")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"The '{column}' column must contain finite values.")
        if column == "step":
            if (numeric <= 0).any() or (numeric % 1 != 0).any():
                raise ValueError("Billing intervals in 'step' must be positive whole months.")
            df[column] = numeric.astype(int)
        else:
            if (numeric < 0).any():
                raise ValueError("Subscription prices in 'price' must not be negative.")
            df[column] = numeric.astype(float)

    contribution_column = "net_contribution_per_charge"
    if contribution_column in df:
        original = df[contribution_column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any() or numeric.isna().any():
            raise ValueError(
                f"The '{contribution_column}' column must contain numeric values for every subscription."
            )
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"The '{contribution_column}' column must contain finite values.")
        df[contribution_column] = numeric.astype(float)

    df.attrs["normalization_assumptions"] = assumptions
    return df


def prepare_dataframe(path="subscriptions.csv"):
    return normalize_subscription_dataframe(pd.read_csv(path))


def build_eligibility(df, asof=ASOF):
    """Return the renewal eligibility matrix for each customer and anniversary."""
    df = df.copy()
    if "step" not in df.columns:
        df["step"] = np.where(df["plan"] == "annual", 12, 1)
    if "source" not in df.columns and "channel" in df.columns:
        df["source"] = np.where(
            df["channel"] == "paid_social",
            "paid_social/" + df["utm_campaign"].fillna(""),
            df["channel"],
        )

    elig = np.zeros((len(df), K + 1), dtype=bool)
    for step in (1, 12):
        idx = np.where(df["step"].to_numpy() == step)[0]
        sub = df.iloc[idx]
        for k in range(K):
            d = sub["created_at"] + pd.DateOffset(months=k * step)
            ok = np.array(d <= asof, dtype=bool, copy=True)
            e = sub["ended_at"].to_numpy()
            ok &= np.where(
                sub["end_reason"].eq("payment_failed").to_numpy(),
                d.to_numpy() <= e,
                sub["ended_at"].isna().to_numpy() | (d.to_numpy() < e),
            )
            elig[idx, k] = ok
    return elig


def revenue_at(df, paid, H):
    in_window = (np.arange(K)[None, :] * df["step"].to_numpy()[:, None]) < H
    return (paid & in_window).sum(axis=1) * df["price"].to_numpy()


def contribution_at(df, paid, H):
    column = "net_contribution_per_charge"
    if column not in df:
        raise ValueError(f"Contribution LTV requires a '{column}' column.")
    in_window = (np.arange(K)[None, :] * df["step"].to_numpy()[:, None]) < H
    return (paid & in_window).sum(axis=1) * df[column].to_numpy()


def mature_at(df, H, asof=ASOF):
    return ((df["created_at"] + pd.DateOffset(months=H)) <= asof).to_numpy()
