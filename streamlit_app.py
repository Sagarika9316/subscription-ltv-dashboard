import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

from ltv_models import (
    add_ltv_cac_ratio,
    estimate_observed_payback,
    source_ltv_confidence_intervals,
    survival_ltv_forecast,
)
from utils import ASOF, HORIZONS, build_eligibility, mature_at, revenue_at


st.set_page_config(
    page_title="LTV Source Comparison",
    page_icon="📈",
    layout="wide",
)


def normalize_dataframe(df):
    df = df.copy()
    if "plan" not in df.columns:
        raise ValueError("The file must contain a 'plan' column.")
    if "source" not in df.columns:
        df["step"] = np.where(df["plan"] == "annual", 12, 1)
        df["price"] = np.where(df["plan"] == "annual", 150, 15)
        df["source"] = np.where(
            df["channel"] == "paid_social",
            "paid_social/" + df["utm_campaign"].fillna(""),
            df["channel"],
        )
    return df


def compute_ltv_summary(df, asof=ASOF, selected_horizons=None, statistic="mean"):
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'.")

    failed = (df["end_reason"] == "payment_failed").to_numpy()
    elig = build_eligibility(df, asof=asof)
    paid = elig[:, :38].copy()
    paid[failed] = elig[failed, :38] & elig[failed, 1:39]

    if selected_horizons is None:
        selected_horizons = HORIZONS

    rows = []
    for H in selected_horizons:
        m = mature_at(df, H, asof=asof)
        t = pd.DataFrame({
            "source": df["source"][m],
            "plan": df["plan"][m],
            "rev": revenue_at(df, paid, H)[m],
        })
        allp = t.groupby("source").rev.agg(ltv=statistic, n="size").assign(plan="all")
        byp = t.groupby(["source", "plan"]).rev.agg(ltv=statistic, n="size").reset_index().set_index("source")
        out = pd.concat([allp, byp]).reset_index().assign(H=H)
        rows.append(out)

    res = pd.concat(rows, ignore_index=True)
    ltv = res[res["plan"] == "all"].pivot(index="source", columns="H", values="ltv").round(1)
    return res, ltv, paid


st.title("Lifetime Value by Acquisition Source")
st.caption("Compare LTV across acquisition sources, plan types, and time horizons for a subscription business.")

with st.sidebar:
    st.header("Controls")
    st.markdown(
        "**Observed LTV + interval**  \n"
        "How much revenue did mature customers generate by this horizon, and how uncertain is the estimate?\n\n"
        "**CAC + payback**  \n"
        "How much LTV per acquisition dollar, and when does observed revenue cover CAC?\n\n"
        "**Retention forecast**  \n"
        "What cumulative revenue might historical retention imply through the selected horizon?"
    )
    uploaded_file = st.file_uploader("Upload subscriptions CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, parse_dates=["created_at", "canceled_at", "ended_at"])
    else:
        try:
            data = pd.read_csv("subscriptions.csv", parse_dates=["created_at", "canceled_at", "ended_at"])
        except FileNotFoundError:
            data = None

    if data is not None:
        df = normalize_dataframe(data)
        asof = st.date_input("As-of date", value=pd.Timestamp(ASOF).date())
        asof_ts = pd.Timestamp(asof)
        plan_filter = st.multiselect("Plan", sorted(df["plan"].dropna().unique().tolist()), default=sorted(df["plan"].dropna().unique().tolist()))
        source_filter = st.multiselect("Acquisition source", sorted(df["source"].dropna().unique().tolist()), default=sorted(df["source"].dropna().unique().tolist()))
        cohort_start_default = df["created_at"].min().date()
        cohort_end_default = df["created_at"].max().date()
        cohort_range = st.date_input(
            "Signup cohort dates",
            value=(cohort_start_default, cohort_end_default),
            min_value=cohort_start_default,
            max_value=cohort_end_default,
        )
        horizon_filter = st.multiselect("Time horizon (months)", options=HORIZONS, default=HORIZONS)
        metric_view = st.radio("Metric view", ["mean LTV", "median LTV"])
        confidence_level = st.select_slider(
            "Confidence interval",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda value: f"{value:.0%}",
        )
        with st.expander("Acquisition costs (CAC)"):
            st.caption("Enter CAC per customer in the same currency as revenue. Leave unknown sources blank.")
            cac_sources = sorted(df["source"].dropna().unique().tolist())
            cac_defaults = pd.DataFrame(
                {
                    "source": cac_sources,
                    "cac": pd.Series(index=range(len(cac_sources)), dtype="float64"),
                }
            )
            cac_inputs = st.data_editor(
                cac_defaults,
                hide_index=True,
                num_rows="fixed",
                key="cac_by_source_editor",
                column_config={
                    "source": st.column_config.TextColumn("Source", disabled=True),
                    "cac": st.column_config.NumberColumn(
                        "CAC per customer",
                        min_value=0.0,
                        step=1.0,
                        format="%.2f",
                    ),
                },
            )

if data is not None:
    cohort_start, cohort_end = cohort_range
    filtered = df[
        (df["plan"].isin(plan_filter))
        & (df["source"].isin(source_filter))
        & (df["created_at"] >= pd.Timestamp(cohort_start))
        & (df["created_at"] < pd.Timestamp(cohort_end) + pd.DateOffset(days=1))
    ].copy()
    if filtered.empty:
        st.warning("No rows match the current filters.")
        st.stop()
    if not horizon_filter:
        st.warning("Select at least one time horizon.")
        st.stop()

    statistic = "median" if metric_view == "median LTV" else "mean"
    res, ltv, paid = compute_ltv_summary(
        filtered,
        asof=asof_ts,
        selected_horizons=horizon_filter,
        statistic=statistic,
    )

    available_horizons = [
        horizon
        for horizon in horizon_filter
        if ((res["H"] == horizon) & (res["plan"] == "all")).any()
    ]
    if not available_horizons:
        st.warning("No subscribers in this signup cohort have reached the selected time horizons yet.")
        st.stop()

    st.subheader("Filter summary")
    st.json({
        "rows": len(filtered),
        "plans": sorted(filtered["plan"].unique().tolist()),
        "sources": sorted(filtered["source"].unique().tolist()),
        "signup_cohort": [str(cohort_start), str(cohort_end)],
        "horizons_with_mature_data": available_horizons,
        "as_of": str(asof_ts.date()),
    })

    h = st.selectbox("Selected comparison horizon", options=available_horizons)
    horizon_unit = "month" if h == 1 else "months"

    ltv_view = res[(res["H"] == h) & (res["plan"] == "all")].set_index("source").sort_values("ltv", ascending=False)
    intervals = source_ltv_confidence_intervals(
        filtered,
        paid,
        h,
        asof=asof_ts,
        statistic=statistic,
        confidence=confidence_level,
    ).set_index("source")
    ltv_view = ltv_view.join(intervals)
    has_cac = pd.to_numeric(cac_inputs["cac"], errors="coerce").gt(0).any()
    if has_cac:
        economics = add_ltv_cac_ratio(ltv_view.reset_index(), cac_inputs).set_index("source")
        payback = estimate_observed_payback(
            filtered,
            paid,
            asof=asof_ts,
            horizons=available_horizons,
            cac_by_source=cac_inputs,
        ).set_index("source")
        ltv_view = ltv_view.join(economics[["cac", "ltv_cac"]]).join(payback)
        payback_display = pd.Series("", index=ltv_view.index, dtype="object")
        costed = ltv_view["cac"].notna()
        payback_display.loc[costed] = "Not reached"
        reached = costed & ltv_view["payback_months"].notna()
        payback_display.loc[reached] = ltv_view.loc[reached, "payback_months"].map(
            lambda months: f"{int(months)} mo"
        )
        ltv_view["payback"] = payback_display
    else:
        ltv_view["cac"] = np.nan
        ltv_view["ltv_cac"] = np.nan
        ltv_view["payback"] = ""

    forecast = survival_ltv_forecast(filtered, asof_ts, available_horizons)
    forecast_view = forecast[forecast["H"] == h].set_index("source")
    if forecast_view.empty:
        ltv_view["forecast_ltv"] = np.nan
        ltv_view["forecast_n"] = np.nan
    else:
        ltv_view = ltv_view.join(
            forecast_view[["forecast_ltv", "n"]].rename(columns={"n": "forecast_n"})
        )

    st.subheader(f"Model comparison at {h} {horizon_unit}")
    comparison = ltv_view[
        [
            "ltv",
            "ci_low",
            "ci_high",
            "n",
            "cac",
            "ltv_cac",
            "payback",
            "forecast_ltv",
            "forecast_n",
        ]
    ].rename(
        columns={
            "ltv": "Observed LTV",
            "ci_low": f"{confidence_level:.0%} CI lower",
            "ci_high": f"{confidence_level:.0%} CI upper",
            "n": "Observed N",
            "cac": "CAC",
            "ltv_cac": "LTV:CAC",
            "payback": "Observed payback",
            "forecast_ltv": "Forecast LTV",
            "forecast_n": "Forecast N",
        }
    )
    st.dataframe(comparison, width="stretch")
    st.caption("Blank CAC columns mean no positive CAC has been entered. Blank forecast cells mean one or more plan segments lack sufficient follow-up.")

    st.markdown("**Model assumptions and sample coverage**")
    observed_notes, economics_notes, forecast_notes = st.columns(3)
    observed_notes.caption(
        f"Observed LTV uses fully mature customers at {h} months; interval is bootstrap-based. Observed N is shown per source."
    )
    economics_notes.caption(
        "LTV:CAC uses entered source-level costs. Payback is the first selected mature horizon where mean observed revenue covers CAC."
    )
    forecast_notes.caption(
        "Forecast uses Kaplan–Meier retention by source and plan; Forecast N counts customers in supported segments and is not realized LTV."
    )

    bar_data = ltv_view["ltv"].sort_values(ascending=False)
    st.bar_chart(bar_data)

    st.subheader("Detailed LTV table")
    detail = res[(res["H"] == h)].pivot(index="source", columns="plan", values="ltv").round(2)
    st.dataframe(detail, width="stretch")

    st.subheader("Summary metrics")
    col1, col2, col3 = st.columns(3)
    mature_mask = mature_at(filtered, h, asof=asof_ts)
    mature_ltv = revenue_at(filtered, paid, h)[mature_mask]
    overall_ltv = mature_ltv.mean() if statistic == "mean" else np.median(mature_ltv)
    summary_stat_label = "Mean" if statistic == "mean" else "Median"
    col1.metric("Mature subscribers", f"{len(mature_ltv):,}")
    col2.metric(
        f"{h}-month {summary_stat_label.lower()} LTV",
        round(float(overall_ltv), 2),
    )
    col3.metric("Largest source", ltv_view.index[0] if not ltv_view.empty else "N/A")

    st.markdown(
        "This dashboard compares expected customer lifetime value by acquisition source under the current monthly/annual subscription assumptions."
    )

    st.subheader("Observed LTV trajectory")
    horizon_table = res[
        (res["plan"] == "all") & res["H"].isin(available_horizons)
    ].pivot(index="source", columns="H", values="ltv").round(1)
    st.line_chart(horizon_table)

    st.download_button(
        label="Download current LTV table",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name="ltv_compare.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Forecast by source")
    st.caption(
        "Modelled expected revenue uses Kaplan–Meier retention and observed billing prices. "
        "Forecasts are limited to horizons within observed follow-up and are not realized LTV."
    )
    forecast_view = forecast_view.sort_values("forecast_ltv", ascending=False)
    if forecast_view.empty:
        st.info("There is not enough observed follow-up for this source and horizon.")
    else:
        st.bar_chart(forecast_view["forecast_ltv"])
else:
    st.info("The dashboard is ready. Choose a CSV in the sidebar to compare LTV by acquisition source.")
