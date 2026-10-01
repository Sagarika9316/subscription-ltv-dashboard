import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-06-30")
K = 38  # anniversaries 0..37, enough for 36 months of monthly billing
HORIZONS = [1, 3, 6, 12, 24, 36]


def prepare_dataframe(path="subscriptions.csv"):
    df = pd.read_csv(path, parse_dates=["created_at", "canceled_at", "ended_at"])
    df["step"] = np.where(df["plan"] == "annual", 12, 1)
    df["price"] = np.where(df["plan"] == "annual", 150, 15)
    df["source"] = np.where(
        df["channel"] == "paid_social",
        "paid_social/" + df["utm_campaign"].fillna(""),
        df["channel"],
    )
    return df


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


def mature_at(df, H, asof=ASOF):
    return ((df["created_at"] + pd.DateOffset(months=H)) <= asof).to_numpy()
