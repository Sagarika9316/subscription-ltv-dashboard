import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st
from lifelines.exceptions import ConvergenceError

from ltv_models import (
    add_ltv_cac_ratio,
    estimate_observed_payback,
    source_ltv_confidence_intervals,
    survival_ltv_forecast,
)
from utils import ASOF, HORIZONS, build_eligibility, mature_at, revenue_at
from survival_models import cox_survival_forecast, evaluate_survival_models


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
        plan_options = sorted(df["plan"].dropna().unique().tolist())
        plan_choice = st.selectbox(
            "Plan type",
            ["All"] + [plan.title() for plan in plan_options],
        )
        plan_filter = (
            plan_options
            if plan_choice == "All"
            else [plan for plan in plan_options if plan.title() == plan_choice]
        )
        source_filter = st.multiselect("Acquisition source", sorted(df["source"].dropna().unique().tolist()), default=sorted(df["source"].dropna().unique().tolist()))
        cohort_start_default = df["created_at"].min().date()
        cohort_end_default = df["created_at"].max().date()
        cohort_range = st.date_input(
            "Signup cohort dates",
            value=(cohort_start_default, cohort_end_default),
            min_value=cohort_start_default,
            max_value=cohort_end_default,
        )
        cohort_start, cohort_end = cohort_range
        horizon_candidates = df[
            df["plan"].isin(plan_filter)
            & df["source"].isin(source_filter)
            & (df["created_at"] >= pd.Timestamp(cohort_start))
            & (df["created_at"] < pd.Timestamp(cohort_end) + pd.DateOffset(days=1))
        ]
        horizon_filter = [
            horizon
            for horizon in HORIZONS
            if mature_at(horizon_candidates, horizon, asof=asof_ts).any()
        ]
        if horizon_filter:
            selected_horizon = 12 if 12 in horizon_filter else horizon_filter[-1]
            h = st.select_slider(
                "Time horizon (months)",
                options=horizon_filter,
                value=selected_horizon,
            )
        else:
            h = None
        min_users = st.slider(
            "Minimum mature customers per source",
            min_value=1,
            max_value=1000,
            value=1,
        )
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
        st.warning("No customers in this selection have reached a supported time horizon.")
        st.stop()
    if h is None:
        st.warning("Choose filters that include customers mature at a supported horizon.")
        st.stop()

    mature_counts = filtered.loc[mature_at(filtered, h, asof=asof_ts)].groupby("source").size()
    qualified_sources = mature_counts[mature_counts >= min_users].index
    filtered = filtered[filtered["source"].isin(qualified_sources)].copy()
    if filtered.empty:
        st.warning(f"No sources have at least {min_users} mature customers at {h} months.")
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

    if h not in available_horizons:
        st.warning("The selected horizon is not supported by the filtered customer data.")
        st.stop()
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
        ltv_view = ltv_view.join(
            economics[["cac", "ltv_cac", "ltv_minus_cac"]]
        ).join(payback)
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
        ltv_view["ltv_minus_cac"] = np.nan
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

    cox_error = None
    try:
        cox_forecast = cox_survival_forecast(filtered, asof_ts, available_horizons)
    except (ConvergenceError, ValueError) as error:
        cox_forecast = pd.DataFrame(columns=["source", "H", "cox_ltv", "n"])
        cox_error = str(error)
    cox_view = cox_forecast[cox_forecast["H"] == h].set_index("source")
    if cox_view.empty:
        ltv_view["cox_ltv"] = np.nan
        ltv_view["cox_n"] = np.nan
    else:
        ltv_view = ltv_view.join(
            cox_view[["cox_ltv", "n"]].rename(columns={"n": "cox_n"})
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
            "ltv_minus_cac",
            "payback",
            "forecast_ltv",
            "forecast_n",
            "cox_ltv",
            "cox_n",
        ]
    ].rename(
        columns={
            "ltv": "Observed LTV",
            "ci_low": f"{confidence_level:.0%} CI lower",
            "ci_high": f"{confidence_level:.0%} CI upper",
            "n": "Observed N",
            "cac": "CAC",
            "ltv_cac": "LTV:CAC",
            "ltv_minus_cac": "LTV - CAC",
            "payback": "Observed payback",
            "forecast_ltv": "Kaplan–Meier LTV",
            "forecast_n": "KM N",
            "cox_ltv": "Cox PH LTV",
            "cox_n": "Cox N",
        }
    )
    st.dataframe(comparison, width="stretch")
    st.caption("Blank CAC columns mean no positive CAC has been entered. Blank forecast cells mean one or more plan segments lack sufficient follow-up.")

    st.markdown("**Model assumptions and sample coverage**")
    observed_notes, economics_notes, km_notes, cox_notes = st.columns(4)
    observed_notes.caption(
        f"Observed LTV uses fully mature customers at {h} months; interval is bootstrap-based. Observed N is shown per source."
    )
    economics_notes.caption(
        "LTV:CAC uses entered source-level costs. Payback is the first selected mature horizon where mean observed revenue covers CAC."
    )
    km_notes.caption(
        "Kaplan–Meier is the non-parametric retention baseline. Its forecast is not realized LTV."
    )
    cox_notes.caption(
        "Cox PH adjusts for source and plan; assumes proportional hazards. Associations are not causal effects."
    )

    ltv_chart, net_chart = st.columns(2)
    with ltv_chart:
        st.subheader("Observed LTV by source")
        st.bar_chart(ltv_view["ltv"].sort_values(ascending=False))
    with net_chart:
        st.subheader("LTV - CAC by source")
        net_ltv = ltv_view["ltv_minus_cac"].dropna().sort_values(ascending=False)
        if net_ltv.empty:
            st.info("Enter CAC values to compare LTV less acquisition cost.")
        else:
            st.bar_chart(net_ltv)
        st.caption("Revenue less acquisition cost; excludes servicing and other costs.")

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
    st.subheader("Survival model forecasts")
    st.caption(
        "Both models use right-censored subscription durations and observed billing prices. Forecasts are not realized LTV."
    )
    forecast_view = forecast_view.sort_values("forecast_ltv", ascending=False)
    if forecast_view.empty:
        st.info("There is not enough observed follow-up for this source and horizon.")
    else:
        km_chart, cox_chart = st.columns(2)
        with km_chart:
            st.markdown("**Kaplan–Meier baseline**")
            st.bar_chart(forecast_view["forecast_ltv"])
        with cox_chart:
            st.markdown("**Cox proportional hazards**")
            if cox_error:
                st.warning(f"Cox model unavailable: {cox_error}")
            elif cox_view.empty:
                st.info("There is not enough supported follow-up for this Cox estimate.")
            else:
                st.bar_chart(cox_view["cox_ltv"].sort_values(ascending=False))

    with st.expander("Out-of-time model validation"):
        st.caption(
            "Trains on earlier signup cohorts and scores a later, fully mature holdout. "
            "Uses selected sources and plans, but the cohort-date filter is intentionally not applied. "
            "MAE and RMSE are in revenue units; lower is better. Negative Cox-minus-KM MAE favors Cox."
        )
        validation_horizon = st.select_slider(
            "Validation horizon (months)",
            options=HORIZONS,
            value=6 if 6 in HORIZONS else HORIZONS[0],
        )
        validation_data = df[
            df["source"].isin(source_filter) & df["plan"].isin(plan_filter)
        ].copy()
        validation_signature = (
            validation_horizon,
            str(asof_ts),
            tuple(sorted(source_filter)),
            tuple(sorted(plan_filter)),
            len(validation_data),
            str(validation_data["created_at"].min()),
            str(validation_data["created_at"].max()),
        )

        if st.button("Run validation", key="run_survival_validation"):
            try:
                with st.spinner("Fitting models and scoring the time-based holdout..."):
                    validation_summary, validation_by_source = evaluate_survival_models(
                        validation_data,
                        asof_ts,
                        validation_horizon,
                    )
                st.session_state["survival_validation_result"] = {
                    "signature": validation_signature,
                    "summary": validation_summary,
                    "by_source": validation_by_source,
                    "error": None,
                }
            except (ConvergenceError, ValueError) as error:
                st.session_state["survival_validation_result"] = {
                    "signature": validation_signature,
                    "summary": None,
                    "by_source": None,
                    "error": str(error),
                }

        validation_result = st.session_state.get("survival_validation_result")
        if validation_result and validation_result["signature"] == validation_signature:
            if validation_result["error"]:
                st.warning(f"Validation unavailable: {validation_result['error']}")
            else:
                validation_summary = validation_result["summary"]
                st.caption(
                    f"Training through {validation_summary['train_cutoff']:%Y-%m-%d} "
                    f"({validation_summary['train_n']:,} customers, "
                    f"{validation_summary['train_events']:,} cancellations); holdout "
                    f"{validation_summary['test_start']:%Y-%m-%d} to "
                    f"{validation_summary['test_end']:%Y-%m-%d}. "
                    f"Scored {validation_summary['scored_n']:,} of "
                    f"{validation_summary['holdout_n']:,} mature customers."
                )
                st.caption(
                    "Billing assumptions: uses row-level step/price when present; the built-in defaults are "
                    "$15 charged monthly and $150 charged annually. Revenue at horizon H includes charges "
                    "scheduled before H, not at H; taxes, refunds, and servicing costs are excluded unless "
                    "already reflected in price. Subscription termination is based on ended_at (canceled_at "
                    "is not used); payment-failure renewals follow a separate eligibility rule."
                )
                if validation_summary["cox_events_per_parameter"] < 10:
                    st.warning(
                        f"Cox PH has {validation_summary['train_events']:,} training cancellations for "
                        f"{validation_summary['cox_parameter_count']} fitted coefficients "
                        f"({validation_summary['cox_events_per_parameter']:.1f} events per coefficient). "
                        "This is a low-event rule-of-thumb warning; treat Cox scores as exploratory."
                    )
                mae_km, mae_cox, rmse_km, rmse_cox = st.columns(4)
                mae_km.metric("Kaplan–Meier MAE", f"{validation_summary['km_mae']:,.2f}")
                mae_cox.metric("Cox PH MAE", f"{validation_summary['cox_mae']:,.2f}")
                rmse_km.metric("Kaplan–Meier RMSE", f"{validation_summary['km_rmse']:,.2f}")
                rmse_cox.metric("Cox PH RMSE", f"{validation_summary['cox_rmse']:,.2f}")
                st.metric(
                    "Cox − Kaplan–Meier MAE",
                    f"{validation_summary['cox_minus_km_mae']:+,.2f}",
                    delta=(
                        "95% month-block CI "
                        f"[{validation_summary['delta_mae_ci_low']:+,.2f}, "
                        f"{validation_summary['delta_mae_ci_high']:+,.2f}]"
                    ),
                    delta_color="off",
                )
                if validation_summary["cox_mae"] < validation_summary["km_mae"]:
                    result_note = "Cox has lower MAE on this holdout."
                else:
                    result_note = "Kaplan–Meier has lower MAE on this holdout."
                if validation_summary["cox_rmse"] > validation_summary["km_rmse"]:
                    result_note += " Cox has higher RMSE, so the result is mixed."
                st.info(result_note)
                st.dataframe(
                    validation_result["by_source"].rename(
                        columns={
                            "source": "Source",
                            "customers": "Customers",
                            "actual_ltv": "Actual LTV",
                            "km_ltv": "KM prediction",
                            "cox_ltv": "Cox prediction",
                            "km_mae": "KM MAE",
                            "cox_mae": "Cox MAE",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        else:
            st.info("Choose a validation horizon and run the holdout comparison.")
else:
    st.info("The dashboard is ready. Choose a CSV in the sidebar to compare LTV by acquisition source.")
