import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from utils import (
    ASOF,
    HORIZONS,
    K,
    STATIC_CAC_BY_SOURCE,
    build_eligibility,
    mature_at,
    prepare_dataframe,
    revenue_at,
)


def main():
    df = prepare_dataframe()
    print(df.head())

    failed = (df["end_reason"] == "payment_failed").to_numpy()
    elig = build_eligibility(df)

    paid = elig[:, :K].copy()
    paid[failed] = elig[failed, :K] & elig[failed, 1 : K + 1]

    print("charges per sub (mean):", paid.sum(axis=1).mean())

    rows = []
    for H in HORIZONS:
        m = mature_at(df, H)
        t = pd.DataFrame({"source": df["source"][m], "plan": df["plan"][m], "rev": revenue_at(df, paid, H)[m]})
        allp = t.groupby("source").rev.agg(ltv="mean", n="size").assign(plan="all")
        byp = t.groupby(["source", "plan"]).rev.agg(ltv="mean", n="size").reset_index().set_index("source")
        out = pd.concat([allp, byp]).reset_index().assign(H=H)
        rows.append(out)

    res = pd.concat(rows)
    ltv = res[res["plan"] == "all"].pivot(index="source", columns="H", values="ltv").round(1)
    print(res[(res["H"] == 12)].pivot(index="source", columns="plan", values="ltv").round(1))

    rng = np.random.default_rng(42)
    H = 12
    m = mature_at(df, H)
    t = pd.DataFrame({"source": df["source"][m], "rev": revenue_at(df, paid, H)[m]})

    ci = []
    for s, g in t.groupby("source"):
        a = g["rev"].to_numpy()
        means = [rng.choice(a, len(a)).mean() for _ in range(300)]
        ci.append((s, len(a), a.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)))
    print(pd.DataFrame(ci, columns=["source", "n", "ltv", "lo", "hi"]).round(1))

    naive = revenue_at(df, paid, 12).mean()
    mature = revenue_at(df, paid, 12)[mature_at(df, 12)].mean()
    print(f"naive {naive:.2f} vs mature-only {mature:.2f}")

    ltv.to_csv("ltv_by_horizon.csv")
    res.to_csv("ltv_all_cuts.csv", index=False)

    d = df.copy()
    d["end"] = d["ended_at"].fillna(ASOF)
    d["months"] = (d["end"] - d["created_at"]).dt.days / 30.44
    d["event"] = d["ended_at"].notna().astype(int)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for i, plan in enumerate(["monthly", "annual"]):
        for s, g in d[d["plan"] == plan].groupby("source"):
            km = KaplanMeierFitter().fit(g["months"], g["event"], label=s)
            km.plot_survival_function(ax=ax[i], ci_show=False)
        ax[i].set_title(f"{plan} plan: share still subscribed")
        ax[i].set_xlabel("months since start")
    plt.tight_layout()
    fig.savefig("subscription_survival.png", dpi=150)
    plt.close(fig)

    d["vol"] = (d["end_reason"] == "voluntary").astype(int)
    d["pf"] = (d["end_reason"] == "payment_failed").astype(int)
    print(
        d[d["created_at"] + pd.DateOffset(months=12) <= ASOF]
        .groupby("source")[["vol", "pf"]]
        .mean()
        .round(3)
    )

    t12 = ltv[12].to_frame("ltv12")
    t12["cac"] = pd.Series(STATIC_CAC_BY_SOURCE)
    t12["ltv_cac"] = (t12["ltv12"] / t12["cac"]).round(2)
    print(t12)


if __name__ == "__main__":
    main()
