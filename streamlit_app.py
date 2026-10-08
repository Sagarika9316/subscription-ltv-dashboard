import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st
from lifelines.exceptions import ConvergenceError

from cohort_models import cohort_churn_summary, cohort_retention_matrix
from experiment_models import (
    estimate_incremental_value_at_lift,
    two_proportion_sample_size,
)
from ltv_models import (
    add_ltv_cac_ratio,
    estimate_observed_payback,
    rank_gross_ltv_cac_candidates,
    source_ltv_confidence_intervals,
    survival_ltv_forecast,
)
from utils import (
    ASOF,
    HORIZONS,
    LONG_TERM_FORECAST_HORIZON,
    STATIC_CAC_BY_SOURCE,
    build_eligibility,
    mature_at,
    normalize_subscription_dataframe,
    contribution_at,
    revenue_at,
)
from survival_models import (
    aft_survival_forecast,
    cox_survival_forecast,
    evaluate_survival_models,
    evaluate_survival_models_rolling,
    aft_long_term_revenue_forecast,
    kaplan_meier_retention_curve,
)


st.set_page_config(
    page_title="LTV Source Comparison",
    page_icon="📈",
    layout="wide",
)


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


st.title("Subscription value by acquisition source")
st.caption(
    "Compare the revenue customers have generated so far, see how many customers "
    "support each comparison, and explore clearly labeled retention forecasts."
)
with st.expander("A quick guide: what these numbers mean"):
    st.markdown(
        "- **Observed revenue** is what customers generated within the selected "
        "time window. It is not unlimited lifetime value.\n"
        "- **Mature customers** have had enough time to reach that window. Newer "
        "customers are excluded from the observed comparison.\n"
        "- **Forecast revenue** estimates future scheduled charges from historical "
        "cancellation patterns; it is not money already received.\n"
        "- **Gross revenue is not profit.** It does not subtract marketing or "
        "servicing costs. CAC must be entered separately for a cost comparison.\n"
        "- Results use the prices shown in the billing assumptions. Missing billing "
        "prices default to $15 monthly and $150 annually.\n"
        "- A confidence interval summarizes uncertainty in a source-level estimate; "
        "it is not a guarantee about an individual customer."
    )

with st.sidebar:
    st.header("Controls")
    st.markdown(
        "Start by uploading the subscription CSV. Then choose a plan, source, signup "
        "period, and measurement horizon. The main comparison uses only customers "
        "who have reached that horizon."
    )
    uploaded_file = st.file_uploader("Upload subscriptions CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        try:
            data = pd.read_csv("subscriptions.csv")
        except FileNotFoundError:
            data = None

    if data is not None:
        try:
            df = normalize_subscription_dataframe(data)
        except ValueError as error:
            st.error(f"Invalid subscription data: {error}")
            st.stop()
        assumptions = df.attrs.get("normalization_assumptions", [])
        if assumptions:
            st.info("Input assumptions: " + " ".join(assumptions))
        st.caption(
            f"Billing currency: {df['currency'].iloc[0]}. Price is per scheduled charge; "
            "billing_interval_months is the number of months between charges, with the first charge at signup. "
            "The selected H-month revenue window includes charges scheduled before month H, not at H."
        )
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
            st.caption("Enter CAC per customer in the same currency as revenue. Unknown values stay blank.")
            static_cac_available = df["currency"].iloc[0] == "USD"
            use_static_cac = st.checkbox(
                "Use illustrative static CAC assumptions",
                value=False,
                key="use_static_cac_assumptions",
                disabled=not static_cac_available,
                help="USD-only example values from the batch analysis; not measured CAC from the subscription CSV.",
            )
            if not static_cac_available:
                st.warning(
                    f"Static CAC assumptions are in USD and disabled for {df['currency'].iloc[0]} data. Enter CAC values in {df['currency'].iloc[0]} instead."
                )
            if use_static_cac:
                st.warning(
                    "Illustrative assumptions are active. Replace them with verified source-level CAC before making business decisions."
                )
            cac_sources = sorted(df["source"].dropna().unique().tolist())
            cac_defaults = pd.DataFrame(
                {
                    "source": cac_sources,
                    "cac": [
                        STATIC_CAC_BY_SOURCE.get(source, np.nan)
                        if use_static_cac
                        else np.nan
                        for source in cac_sources
                    ],
                }
            )
            cac_inputs = st.data_editor(
                cac_defaults,
                hide_index=True,
                num_rows="fixed",
                key=(
                    "cac_by_source_static_editor"
                    if use_static_cac
                    else "cac_by_source_manual_editor"
                ),
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

if data is None:
    st.info(
        "To begin, upload your subscriptions CSV in the sidebar. It needs a plan, "
        "signup date, end date and end reason, plus an acquisition source or channel. "
        "For public deployments, upload only data you are permitted to share."
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

    st.subheader("Your current comparison")
    scope_cols = st.columns(4)
    scope_cols[0].metric("Subscriptions in scope", f"{len(filtered):,}")
    scope_cols[1].metric("Sources shown", f"{filtered['source'].nunique():,}")
    scope_cols[2].metric("Plan", plan_choice)
    scope_cols[3].metric("Data considered through", str(asof_ts.date()))
    st.caption(
        f"Signup dates: {cohort_start} to {cohort_end}. Mature-data horizons "
        f"available for this selection: {', '.join(map(str, horizon_filter))} months."
    )

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

    has_contribution = "net_contribution_per_charge" in filtered.columns
    if has_contribution:
        mature_mask = mature_at(filtered, h, asof=asof_ts)
        contribution_values = contribution_at(filtered, paid, h)
        contribution_data = pd.DataFrame(
            {
                "source": filtered.loc[mature_mask, "source"],
                "contribution_ltv": contribution_values[mature_mask],
            }
        )
        contribution_by_source = contribution_data.groupby("source")[
            "contribution_ltv"
        ].agg(statistic)
        ltv_view = ltv_view.join(contribution_by_source)
        ltv_view["contribution_minus_cac"] = (
            ltv_view["contribution_ltv"] - ltv_view["cac"]
        )
        ltv_view["contribution_ltv_cac"] = (
            ltv_view["contribution_ltv"] / ltv_view["cac"]
        )
    else:
        ltv_view["contribution_ltv"] = np.nan
        ltv_view["contribution_minus_cac"] = np.nan
        ltv_view["contribution_ltv_cac"] = np.nan

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

    aft_error = None
    try:
        aft_forecast = aft_survival_forecast(filtered, asof_ts, available_horizons)
    except (ConvergenceError, ValueError) as error:
        aft_forecast = pd.DataFrame(columns=["source", "H", "aft_ltv", "n"])
        aft_error = str(error)
    aft_view = aft_forecast[aft_forecast["H"] == h].set_index("source")
    if aft_view.empty:
        ltv_view["aft_ltv"] = np.nan
        ltv_view["aft_n"] = np.nan
    else:
        ltv_view = ltv_view.join(
            aft_view[["aft_ltv", "n"]].rename(columns={"n": "aft_n"})
        )

    st.header("What the data says")
    st.caption(
        f"Observed gross revenue per mature subscription through {h} "
        f"{horizon_unit}; values are in {df['currency'].iloc[0]}."
    )
    leader = ltv_view.iloc[0]
    leader_source = ltv_view.index[0]
    mature_total = int(ltv_view["n"].sum())
    headline_cols = st.columns(3)
    headline_cols[0].metric(
        f"Highest observed {h}-month revenue",
        leader_source,
        help="This is the source with the highest selected mean or median among mature customers.",
    )
    statistic_label = "Mean" if statistic == "mean" else "Median"
    headline_cols[1].metric(
        f"{statistic_label} per mature customer",
        f"{leader['ltv']:,.2f} {df['currency'].iloc[0]}",
        help=f"{confidence_level:.0%} bootstrap interval: "
        f"{leader['ci_low']:,.2f}–{leader['ci_high']:,.2f} {df['currency'].iloc[0]}.",
    )
    headline_cols[2].metric("Mature subscriptions compared", f"{mature_total:,}")
    st.markdown(
        f"**How to read this:** {leader_source} ranks highest for this filter and "
        f"horizon. The estimate uses {int(leader['n']):,} mature subscriptions; its "
        f"{confidence_level:.0%} bootstrap interval is "
        f"{leader['ci_low']:,.2f}–{leader['ci_high']:,.2f} "
        f"{df['currency'].iloc[0]}. This is an association in historical data, not "
        "proof that the source caused higher value or that increasing its budget will work."
    )
    if len(ltv_view) > 1:
        runner_up = ltv_view.iloc[1]
        if leader["ci_low"] <= runner_up["ci_high"]:
            st.info(
                f"The top source's interval overlaps the next-ranked source "
                f"({ltv_view.index[1]}). Treat the ranking as uncertain rather than "
                "assuming the leader is definitively better."
            )
    st.caption(
        "Observed revenue uses only subscriptions old enough to reach the selected "
        "horizon. It excludes costs and future charges."
    )

    st.subheader(f"Detailed results at {h} {horizon_unit}")
    st.caption(
        "Observed revenue uses mature customers. Forecast columns estimate future "
        "gross revenue from retention patterns. All money values are in "
        f"{df['currency'].iloc[0]}."
    )
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
            "contribution_ltv",
            "contribution_minus_cac",
            "contribution_ltv_cac",
            "forecast_ltv",
            "forecast_n",
            "cox_ltv",
            "cox_n",
            "aft_ltv",
            "aft_n",
        ]
    ].rename(
        columns={
            "ltv": "Observed gross revenue",
            "ci_low": f"{confidence_level:.0%} CI lower",
            "ci_high": f"{confidence_level:.0%} CI upper",
            "n": "Observed N",
            "cac": "CAC",
            "ltv_cac": "LTV:CAC",
            "ltv_minus_cac": "LTV - CAC",
            "payback": "Mean-revenue payback",
            "contribution_ltv": "Contribution LTV",
            "contribution_minus_cac": "Contribution - CAC",
            "contribution_ltv_cac": "Contribution:CAC",
            "forecast_ltv": "Kaplan–Meier forecast",
            "forecast_n": "KM N",
            "cox_ltv": "Cox PH forecast",
            "cox_n": "Cox N",
            "aft_ltv": "Weibull AFT forecast",
            "aft_n": "AFT N",
        }
    )
    comparison_config = {
        column: st.column_config.NumberColumn(format="%.2f")
        for column in (
            "Observed gross revenue",
            f"{confidence_level:.0%} CI lower",
            f"{confidence_level:.0%} CI upper",
            "CAC",
            "LTV:CAC",
            "LTV - CAC",
            "Contribution LTV",
            "Contribution - CAC",
            "Contribution:CAC",
            "Kaplan–Meier forecast",
            "Cox PH forecast",
            "Weibull AFT forecast",
        )
    }
    comparison_config.update(
        {
            "Observed N": st.column_config.NumberColumn(format="%.0f"),
            "KM N": st.column_config.NumberColumn(format="%.0f"),
            "Cox N": st.column_config.NumberColumn(format="%.0f"),
            "AFT N": st.column_config.NumberColumn(format="%.0f"),
            "LTV:CAC": st.column_config.NumberColumn(format="%.2fx"),
            "Contribution:CAC": st.column_config.NumberColumn(format="%.2fx"),
        }
    )
    st.dataframe(comparison, width="stretch", column_config=comparison_config)
    st.caption(
        "LTV:CAC is a ratio: 1.00x means selected-horizon gross revenue equals "
        "entered customer acquisition cost. It does not mean the customer is profitable."
    )
    st.caption(
        "Money values use the selected billing currency. Blank CAC values mean no "
        "positive CAC was entered. A blank forecast means a plan segment lacks "
        "required follow-up. Payback is based on mean observed revenue."
    )

    st.subheader("Marketing check: which source merits a test?")
    minimum_gross_ltv_cac = st.number_input(
        "Minimum lower-bound gross LTV:CAC for test screen",
        min_value=0.1,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="1.0x means the lower confidence bound of gross LTV only equals CAC; it does not cover other costs.",
    )
    candidates = rank_gross_ltv_cac_candidates(
        ltv_view.reset_index(), minimum_ratio=minimum_gross_ltv_cac
    )
    if candidates.empty:
        st.info(
            "Enter a positive CAC for at least one selected source to rank investment-test candidates."
        )
    else:
        candidate = candidates.iloc[0]
        passing_candidates = candidates[candidates["passes_gross_screen"]]
        candidate_metrics = st.columns(5)
        candidate_metrics[0].metric("Top-ranked source to investigate", candidate["source"])
        candidate_metrics[1].metric(
            "Entered CAC per customer",
            f"{candidate['cac']:,.2f} {df['currency'].iloc[0]}",
        )
        candidate_metrics[2].metric(
            "Observed gross LTV:CAC", f"{candidate['gross_ltv_cac']:.2f}x"
        )
        candidate_metrics[3].metric(
            f"Lower-bound gross LTV:CAC ({confidence_level:.0%})",
            f"{candidate['lower_bound_gross_ltv_cac']:.2f}x",
        )
        candidate_metrics[4].metric("Mature customers", f"{int(candidate['n']):,}")
        st.caption(
            f"Observed payback at selected mature horizons: {candidate['payback'] or 'not reached'}. "
            "Ranking uses the lower confidence bound of gross LTV:CAC, with mature sample size as a tie-breaker."
        )
        if not passing_candidates.empty:
            passing_source = passing_candidates.iloc[0]["source"]
            st.success(
                f"{passing_source} clears the {minimum_gross_ltv_cac:.1f}x gross LTV:CAC screen: "
                "candidate for a controlled incrementality test, not an instruction to scale spend."
            )
        else:
            st.warning(
                f"No source clears the {minimum_gross_ltv_cac:.1f}x conservative gross LTV:CAC screen. "
                "Do not recommend increased spend from these results. The top-ranked source is a hypothesis only."
            )
        st.caption(
            "This screen assumes entered CAC is accurate and uses gross revenue, not contribution profit. "
            "The default 1.0x threshold only means gross LTV covers CAC; it excludes service costs, refunds, and other costs."
        )

    st.subheader("Does observed revenue cover acquisition cost?")
    if not has_cac:
        cac_answer = "CAC comparison is unavailable because no positive CAC values are entered."
    elif candidates.empty:
        cac_answer = "No source has both an entered positive CAC and a usable LTV confidence interval."
    elif passing_candidates.empty:
        cac_answer = (
            f"No source clears the {minimum_gross_ltv_cac:.1f}x lower-bound gross LTV:CAC screen."
        )
    else:
        cac_candidate = passing_candidates.iloc[0]
        cac_basis = (
            "illustrative static CAC assumptions"
            if use_static_cac
            else "manually entered CAC values"
        )
        cac_answer = (
            f"{cac_candidate['source']} is the leading controlled-test candidate using {cac_basis}; "
            f"its lower-bound gross LTV:CAC is {cac_candidate['lower_bound_gross_ltv_cac']:.2f}x."
        )
    st.markdown(f"**Does any source clear the CAC screen?** {cac_answer}")
    st.info(
        "A revenue-to-CAC screen is not a budget recommendation. This subscription "
        "file does not contain randomized treatment/control outcomes or verified "
        "contribution costs. Use a promising source as a test hypothesis, not proof "
        "that increasing spend will be profitable."
    )

    with st.expander("Plan an incrementality experiment"):
        st.caption(
            "This plans a randomized test of paid-subscription conversion. Your subscription CSV has no eligible "
            "prospect counts or control assignment, so enter the baseline rate and audience from campaign data. "
            "Sample size assumes two equally sized, independently randomized arms and a two-sided test."
        )
        planner_sources = sorted(filtered["source"].dropna().unique().tolist())
        initial_source = candidate["source"] if has_cac and not candidates.empty else planner_sources[0]
        test_source = st.selectbox(
            "Source/campaign to test",
            planner_sources,
            index=planner_sources.index(initial_source),
            key="experiment_source",
        )
        control_condition = st.text_input(
            "Control condition",
            placeholder="e.g. current campaign strategy",
            key="experiment_control_condition",
        )
        treatment_condition = st.text_input(
            "Treatment condition",
            placeholder="e.g. increased spend or a new offer",
            key="experiment_treatment_condition",
        )
        baseline_rate_pct = st.number_input(
            "Historical control paid-conversion rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=0.1,
            format="%.2f",
            help="Use the rate among eligible prospects from campaign or experiment logs; it is not in the subscriptions CSV.",
        )
        minimum_lift_pp = st.number_input(
            "Minimum detectable absolute lift (percentage points)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=0.1,
            format="%.2f",
            help="Choose the smallest conversion increase worth acting on economically.",
        )
        design_alpha, design_power = st.columns(2)
        alpha_pct = design_alpha.selectbox(
            "False-positive rate (alpha)", [1, 5, 10], index=1, format_func=lambda value: f"{value}%"
        )
        power_pct = design_power.selectbox(
            "Statistical power", [80, 90, 95], index=0, format_func=lambda value: f"{value}%"
        )
        available_per_arm = st.number_input(
            "Eligible prospects available per arm (optional)",
            min_value=0,
            value=0,
            step=100,
            help="Enter the reachable audience for the chosen randomization unit. Leave at 0 if unknown.",
        )
        if not has_contribution:
            assumed_contribution_margin_pct = st.slider(
                "Illustrative contribution margin assumption (%)",
                min_value=0,
                max_value=100,
                value=60,
                step=5,
                help="Hypothetical only; the subscription CSV contains no variable-cost data.",
            )
            st.caption(
                "This margin is an example assumption, not measured cost data. Replace it with finance-validated contribution margin."
            )

        if baseline_rate_pct is None or minimum_lift_pp is None or minimum_lift_pp == 0:
            st.info("Enter a historical conversion baseline and a positive minimum lift to calculate sample size.")
        else:
            try:
                required_per_arm = two_proportion_sample_size(
                    baseline_rate_pct / 100,
                    minimum_lift_pp / 100,
                    alpha=alpha_pct / 100,
                    power=power_pct / 100,
                )
            except ValueError as error:
                st.error(f"Experiment design inputs are invalid: {error}")
            else:
                if control_condition.strip() and treatment_condition.strip():
                    st.caption(
                        f"Proposed comparison for {test_source}: {treatment_condition.strip()} "
                        f"versus {control_condition.strip()}. Primary outcome: new paid subscriptions per randomized eligible prospect."
                    )
                else:
                    st.caption("Describe the control and treatment before launching; keep all other conditions comparable.")
                sample_col, total_col, lift_col = st.columns(3)
                sample_col.metric("Required per arm", f"{required_per_arm:,}")
                total_col.metric("Required total audience", f"{2 * required_per_arm:,}")
                lift_col.metric(
                    "Incremental conversions at target lift",
                    f"{required_per_arm * minimum_lift_pp / 100:,.1f}",
                )
                if available_per_arm > 0:
                    if available_per_arm < required_per_arm:
                        st.warning(
                            f"The entered audience ({available_per_arm:,} per arm) is below the estimated "
                            f"requirement ({required_per_arm:,} per arm); the test may be underpowered."
                        )
                    else:
                        st.success("The entered audience meets the approximate sample-size target.")
                source_cac = pd.to_numeric(cac_inputs.loc[cac_inputs["source"] == test_source, "cac"], errors="coerce")
                if not source_cac.empty and source_cac.iloc[0] > 0:
                    scenario_per_arm = (
                        int(available_per_arm)
                        if available_per_arm > 0
                        else required_per_arm
                    )
                    source_result = ltv_view.loc[test_source]
                    has_source_contribution = (
                        has_contribution
                        and pd.notna(source_result["contribution_ltv"])
                    )
                    value_per_subscriber = (
                        float(source_result["contribution_ltv"])
                        if has_source_contribution
                        else float(source_result["ltv"])
                        * assumed_contribution_margin_pct
                        / 100
                    )
                    economics = estimate_incremental_value_at_lift(
                        sample_per_arm=scenario_per_arm,
                        absolute_lift=minimum_lift_pp / 100,
                        value_per_subscriber=value_per_subscriber,
                        cac_per_subscriber=float(source_cac.iloc[0]),
                    )
                    value_label = (
                        "Illustrative contribution after CAC"
                        if has_source_contribution
                        else "Illustrative value after CAC (assumed margin)"
                    )
                    economics_metrics = st.columns(3)
                    economics_metrics[0].metric(
                        "Additional subscribers at target lift",
                        f"{economics['incremental_subscribers']:,.1f}",
                    )
                    economics_metrics[1].metric(
                        "Incremental CAC estimate",
                        f"{economics['incremental_cac']:,.2f} {df['currency'].iloc[0]}",
                    )
                    economics_metrics[2].metric(
                        value_label,
                        f"{economics['value_after_cac']:,.2f} {df['currency'].iloc[0]}",
                    )
                    st.caption(
                        f"Scenario at the {minimum_lift_pp:.2f}-percentage-point target lift and "
                        f"{scenario_per_arm:,} prospects per arm. It uses "
                        f"{('observed contribution LTV' if has_source_contribution else f'gross LTV × {assumed_contribution_margin_pct}% assumed margin')} "
                        f"at the selected horizon and CAC {source_cac.iloc[0]:,.2f} per incremental subscriber, "
                        "held constant as spend scales. This is not an observed experiment result or proven profit; "
                        "use actual arm-level spend and contribution outcomes before deciding."
                    )
                else:
                    st.info(
                        f"No positive CAC is entered for {test_source}; the planner can size the conversion test, "
                        "but cannot estimate value after acquisition cost."
                    )
                st.caption(
                    "Approximate normal-theory sample size; assumes independent users, no cluster randomization, "
                    "stable conversion rates, and one primary outcome. Pre-register the analysis and evaluate "
                    "incremental contribution after costs before changing budget."
                )

    if has_contribution:
        st.caption(
            "Contribution LTV uses net_contribution_per_charge for each eligible billing event. "
            "Survival forecasts remain gross-revenue forecasts."
        )
    else:
        st.info(
            "Contribution LTV is unavailable because the CSV has no "
            "'net_contribution_per_charge' column. Supply per-charge revenue after variable costs and allocated refunds; no zero-cost assumption is made."
        )

    with st.expander("Model notes and limitations"):
        st.markdown(
            "- **Observed comparison:** includes only customers old enough to reach "
            f"the selected {h}-month checkpoint. The interval is bootstrap-based.\n"
            "- **Kaplan–Meier:** estimates retention over time; subscriptions still "
            "active at the as-of date count as active through their last observed date.\n"
            "- **Cox proportional hazards:** compares cancellation patterns by "
            "source and plan; assumes relative risks are proportional over time.\n"
            "- **Weibull AFT:** extrapolates subscription duration using a Weibull "
            "distribution. Long-range estimates depend on that assumption.\n"
            "- **All source comparisons are observational.** They do not show that "
            "the acquisition source caused a difference in customer value.\n"
            "- Gross revenue excludes acquisition costs, servicing costs, refunds, "
            "and taxes unless already represented in the input prices."
        )

    ltv_chart, net_chart, contribution_chart = st.columns(3)
    with ltv_chart:
        st.subheader("Observed gross revenue by source")
        st.bar_chart(
            ltv_view["ltv"].sort_values(ascending=False),
            y_label=f"Gross revenue ({df['currency'].iloc[0]})",
        )
    with net_chart:
        st.subheader("Observed revenue minus CAC")
        net_ltv = ltv_view["ltv_minus_cac"].dropna().sort_values(ascending=False)
        if net_ltv.empty:
            st.info("Enter CAC values to compare LTV less acquisition cost.")
        else:
            st.bar_chart(
                net_ltv,
                y_label=f"Revenue less CAC ({df['currency'].iloc[0]})",
            )
        st.caption(
            "Observed gross revenue less entered CAC. This is not profit; servicing "
            "and other costs are excluded."
        )
    with contribution_chart:
        st.subheader("Contribution value by source")
        if has_contribution:
            st.bar_chart(
                ltv_view["contribution_ltv"].dropna().sort_values(ascending=False),
                y_label=f"Contribution ({df['currency'].iloc[0]})",
            )
        else:
            st.info(
                "Contribution data is not in this CSV, so this chart is unavailable. "
                "Gross revenue should not be treated as profit."
            )

    st.subheader("Observed revenue by source and plan")
    detail = res[(res["H"] == h)].pivot(index="source", columns="plan", values="ltv").round(2)
    st.dataframe(detail, width="stretch")
    st.caption(
        f"Each value is {statistic_label.lower()} gross revenue per mature subscription "
        f"through month {h}, in {df['currency'].iloc[0]}."
    )

    st.subheader("All-source snapshot")
    col1, col2, col3 = st.columns(3)
    mature_mask = mature_at(filtered, h, asof=asof_ts)
    mature_ltv = revenue_at(filtered, paid, h)[mature_mask]
    overall_ltv = mature_ltv.mean() if statistic == "mean" else np.median(mature_ltv)
    summary_stat_label = "Mean" if statistic == "mean" else "Median"
    col1.metric("Mature subscriptions", f"{len(mature_ltv):,}")
    col2.metric(
        f"{h}-month {summary_stat_label.lower()} gross revenue",
        f"{overall_ltv:,.2f} {df['currency'].iloc[0]}",
    )
    col3.metric("Highest observed source", ltv_view.index[0] if not ltv_view.empty else "N/A")

    st.subheader("How observed revenue changes with time")
    horizon_table = res[
        (res["plan"] == "all") & res["H"].isin(available_horizons)
    ].pivot(index="source", columns="H", values="ltv").round(1)
    st.line_chart(
        horizon_table,
        y_label=f"Gross revenue ({df['currency'].iloc[0]})",
        x_label="Months after signup",
    )

    with st.expander("Retention and churn curves by source and plan"):
        st.caption(
            "The curve estimates the share still subscribed as months pass after signup. "
            "Customers still active are counted as active through the last date observed, "
            "not treated as cancellations. Curves stop when too few customers remain under observation."
        )
        curve_metric = st.radio(
            "Curve metric",
            ["Retention", "Cumulative churn"],
            horizontal=True,
            key="retention_curve_metric",
        )
        curve_data = kaplan_meier_retention_curve(
            filtered,
            asof=asof_ts,
            max_age=max(HORIZONS),
            min_at_risk=min_users,
            confidence=confidence_level,
        )
        if curve_data.empty:
            st.info("No source/plan groups have enough customers at risk to draw a curve.")
        else:
            curve_plans = sorted(curve_data["plan"].unique().tolist())
            curve_tabs = st.tabs([plan.title() for plan in curve_plans])
            for plan, tab in zip(curve_plans, curve_tabs):
                with tab:
                    plan_curve = curve_data[curve_data["plan"] == plan].copy()
                    if curve_metric == "Cumulative churn":
                        plan_curve["curve_value"] = 1 - plan_curve["retention"]
                        plan_curve["curve_low"] = 1 - plan_curve["ci_high"]
                        plan_curve["curve_high"] = 1 - plan_curve["ci_low"]
                        value_label = "Cumulative churn"
                    else:
                        plan_curve["curve_value"] = plan_curve["retention"]
                        plan_curve["curve_low"] = plan_curve["ci_low"]
                        plan_curve["curve_high"] = plan_curve["ci_high"]
                        value_label = "Retention"

                    chart = plan_curve.pivot(
                        index="age_month", columns="source", values="curve_value"
                    )
                    st.line_chart(chart, y_label=value_label, x_label="Months since signup")
                    curve_table = plan_curve.rename(
                        columns={
                            "source": "Source",
                            "age_month": "Months since signup",
                            "at_risk": "Customers at risk",
                            "curve_value": value_label,
                            "curve_low": f"{confidence_level:.0%} CI lower",
                            "curve_high": f"{confidence_level:.0%} CI upper",
                        }
                    )[
                        [
                            "Source",
                            "Months since signup",
                            "Customers at risk",
                            value_label,
                            f"{confidence_level:.0%} CI lower",
                            f"{confidence_level:.0%} CI upper",
                        ]
                    ]
                    st.dataframe(
                        curve_table,
                        hide_index=True,
                        column_config={
                            value_label: st.column_config.NumberColumn(format="percent"),
                            f"{confidence_level:.0%} CI lower": st.column_config.NumberColumn(format="percent"),
                            f"{confidence_level:.0%} CI upper": st.column_config.NumberColumn(format="percent"),
                        },
                        width="stretch",
                    )

    with st.expander("Cohort retention matrix"):
        st.caption(
            "Each row groups customers who signed up in the same month, source, and plan. "
            "Each cell shows the share still subscribed at that many months after signup. "
            "Blank cells mean not enough customers have been observed for that long."
        )
        cohort_data = df[
            df["source"].isin(source_filter)
            & df["plan"].isin(plan_filter)
            & (df["created_at"] >= pd.Timestamp(cohort_start))
            & (df["created_at"] < pd.Timestamp(cohort_end) + pd.DateOffset(days=1))
        ].copy()
        retention = cohort_retention_matrix(
            cohort_data,
            asof=asof_ts,
            max_age=max(HORIZONS),
            min_customers=min_users,
        )
        if retention.empty:
            st.info("No cohort ages meet the minimum customer threshold.")
        else:
            retention_table = retention.pivot(
                index=["source", "plan", "cohort_month"],
                columns="age_month",
                values="retention",
            )
            retention_table.columns = [f"Month {age}" for age in retention_table.columns]
            st.dataframe(
                retention_table,
                column_config={
                    column: st.column_config.NumberColumn(format="percent")
                    for column in retention_table.columns
                },
                width="stretch",
            )

        st.markdown("**Cumulative churn by source and plan**")
        st.caption(
            "Each row is measured at customer age 1, 3, 6, or 12 months. The denominator includes only "
            "customers with enough follow-up to reach that age. Active subscriptions are right-censored "
            "and counted as retained through their observable age, not as churn. Terminations on the exact "
            "checkpoint date count as retained at that checkpoint."
        )
        churn_summary = cohort_churn_summary(
            cohort_data,
            asof=asof_ts,
            horizons=[1, 3, 6, 12],
            min_customers=min_users,
        )
        if churn_summary.empty:
            st.info("No selected source/plan cohorts have mature observations at months 1, 3, 6, or 12.")
        else:
            churn_summary = churn_summary.rename(
                columns={
                    "source": "Source",
                    "plan": "Plan",
                    "age_month": "Months since signup",
                    "matured": "Matured customers",
                    "retained": "Retained customers",
                    "churned": "Churned customers",
                    "voluntary_churned": "Voluntary churned",
                    "payment_failure_churned": "Payment-failure churned",
                    "other_churned": "Other/unknown churned",
                    "retention": "Retention %",
                    "churn_rate": "Total churn %",
                }
            )
            percentage_columns = [
                "Retention %",
                "Total churn %",
            ]
            st.dataframe(
                churn_summary,
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(format="percent")
                    for column in percentage_columns
                },
                width="stretch",
            )

    st.download_button(
        label="Download current LTV table",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name="ltv_compare.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Survival model forecasts")
    st.caption(
        "These are model-based gross-revenue estimates, not realized revenue and "
        "not unlimited lifetime value. Compare them with the out-of-time validation below."
    )
    forecast_view = forecast_view.sort_values("forecast_ltv", ascending=False)
    if forecast_view.empty:
        st.info("There is not enough observed follow-up for this source and horizon.")
    else:
        km_chart, cox_chart, aft_chart = st.columns(3)
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
        with aft_chart:
            st.markdown("**Weibull AFT**")
            if aft_error:
                st.warning(f"AFT model unavailable: {aft_error}")
            elif aft_view.empty:
                st.info("There is not enough supported follow-up for this AFT estimate.")
            else:
                st.bar_chart(aft_view["aft_ltv"].sort_values(ascending=False))

    st.subheader(
        f"Long-term gross revenue forecast ({LONG_TERM_FORECAST_HORIZON} months)"
    )
    st.caption(
        "This is a source-level expected revenue estimate from one Weibull AFT model "
        "using source and plan as predictors, applied to each plan's billing schedule. "
        "It is not an individual customer's prediction or an unlimited lifetime value. "
        "The interval reflects model parameter "
        "uncertainty; it does not include price, model-choice, or future-business uncertainty."
    )
    try:
        long_term_forecast = aft_long_term_revenue_forecast(
            filtered,
            asof=asof_ts,
            horizon=LONG_TERM_FORECAST_HORIZON,
            confidence=confidence_level,
        )
    except (ConvergenceError, ValueError) as error:
        st.warning(f"Long-term Weibull forecast unavailable: {error}")
    else:
        if long_term_forecast.empty:
            st.info(
                "There are not enough observed cancellations to estimate the "
                "long-term Weibull forecast."
            )
        else:
            display_forecast = long_term_forecast.rename(
                columns={
                    "source": "Source",
                    "forecast_ltv": "60-month expected gross revenue",
                    "ci_low": f"{confidence_level:.0%} model interval lower",
                    "ci_high": f"{confidence_level:.0%} model interval upper",
                    "n": "Customers",
                    "max_observed_followup_months": "Shortest source/plan follow-up (months)",
                    "extrapolation_months": "Months extrapolated",
                }
            ).set_index("Source")
            st.dataframe(display_forecast.round(2), width="stretch")
            extrapolated = long_term_forecast[
                long_term_forecast["extrapolation_months"] > 0
            ]
            if not extrapolated.empty:
                st.warning(
                    "These 60-month estimates extrapolate beyond observed "
                    "subscription follow-up. The farther beyond observed history, "
                    "the more the result depends on the Weibull assumption."
                )
            st.caption(
                f"Prices use the selected billing assumptions ({df['currency'].iloc[0]}). "
                "The model interval is an approximate confidence interval for fitted "
                "Weibull parameters, not a range guaranteed to contain an individual "
                "customer's future revenue. Revenue excludes costs and discounts."
            )

    with st.expander("Can the forecast predict later customers? (model validation)"):
        st.caption(
            "Trains on earlier signups, then checks its predictions against a later group "
            "whose selected measurement period has fully elapsed. "
            "Uses selected sources and plans, but the cohort-date filter is intentionally not applied. "
            "Compares Kaplan–Meier, Cox PH, and Weibull AFT on the same customers. "
            "Cox assumes proportional hazards; Weibull AFT assumes Weibull event times. "
            "MAE is the average absolute prediction error. RMSE penalizes large misses more. "
            "Both are in gross-revenue units; lower is better. A short-horizon test does not "
            "validate a 60-month forecast."
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
                    f"Billing assumptions: currency is {df['currency'].iloc[0]}; uses row-level price and "
                    "billing_interval_months when present. Built-in defaults are "
                    "$15 charged monthly and $150 charged annually in USD. Revenue at horizon H includes charges "
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
                km_metrics, cox_metrics, aft_metrics = st.columns(3)
                with km_metrics:
                    st.markdown("**Kaplan–Meier baseline**")
                    st.metric("Typical error (MAE)", f"{validation_summary['km_mae']:,.2f}")
                    st.metric("Large-error-sensitive score (RMSE)", f"{validation_summary['km_rmse']:,.2f}")
                with cox_metrics:
                    st.markdown("**Cox proportional hazards**")
                    st.metric("Typical error (MAE)", f"{validation_summary['cox_mae']:,.2f}")
                    st.metric("Large-error-sensitive score (RMSE)", f"{validation_summary['cox_rmse']:,.2f}")
                with aft_metrics:
                    st.markdown("**Weibull AFT**")
                    st.metric("Typical error (MAE)", f"{validation_summary['aft_mae']:,.2f}")
                    st.metric("Large-error-sensitive score (RMSE)", f"{validation_summary['aft_rmse']:,.2f}")

                cox_delta, aft_delta = st.columns(2)
                cox_delta.metric(
                    "Cox − KM MAE",
                    f"{validation_summary['cox_minus_km_mae']:+,.2f}",
                    delta=(
                        "95% month-block CI "
                        f"[{validation_summary['cox_delta_mae_ci_low']:+,.2f}, "
                        f"{validation_summary['cox_delta_mae_ci_high']:+,.2f}]"
                    ),
                    delta_color="off",
                )
                aft_delta.metric(
                    "AFT − KM MAE",
                    f"{validation_summary['aft_minus_km_mae']:+,.2f}",
                    delta=(
                        "95% month-block CI "
                        f"[{validation_summary['aft_delta_mae_ci_low']:+,.2f}, "
                        f"{validation_summary['aft_delta_mae_ci_high']:+,.2f}]"
                    ),
                    delta_color="off",
                )
                mae_scores = {
                    "Kaplan–Meier": validation_summary["km_mae"],
                    "Cox PH": validation_summary["cox_mae"],
                    "Weibull AFT": validation_summary["aft_mae"],
                }
                rmse_scores = {
                    "Kaplan–Meier": validation_summary["km_rmse"],
                    "Cox PH": validation_summary["cox_rmse"],
                    "Weibull AFT": validation_summary["aft_rmse"],
                }
                mae_winner = min(mae_scores, key=mae_scores.get)
                rmse_winner = min(rmse_scores, key=rmse_scores.get)
                st.info(
                    f"Lowest MAE: {mae_winner}; lowest RMSE: {rmse_winner}. "
                    "A difference in winners indicates a trade-off, not one overall winner."
                )
                ph_tests = validation_summary["ph_tests"]
                ph_violations = ph_tests[ph_tests["potential_violation"]]
                if ph_violations.empty:
                    st.info(
                        "The Schoenfeld test found no statistically significant evidence against proportional hazards at the 5% level. This does not prove the assumption holds."
                    )
                else:
                    predictors = ", ".join(ph_violations["predictor"].tolist())
                    st.warning(
                        "The Schoenfeld test found potential proportional-hazards violations for: "
                        f"{predictors}. Treat Cox estimates cautiously; inspect time-varying effects."
                    )
                with st.expander("Cox proportional-hazards diagnostic details"):
                    st.caption(
                        "Schoenfeld-residual tests are fit on the training split only. "
                        "Small p-values flag possible time-varying effects; large samples can detect small departures."
                    )
                    ph_display = ph_tests[
                        ["predictor", "test_statistic", "p", "potential_violation"]
                    ].rename(
                        columns={
                            "predictor": "Predictor",
                            "test_statistic": "Test statistic",
                            "p": "p-value",
                            "potential_violation": "Potential violation",
                        }
                    )
                    st.dataframe(
                        ph_display,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "p-value": st.column_config.NumberColumn(format="%.4g"),
                            "Test statistic": st.column_config.NumberColumn(format="%.4g"),
                        },
                    )
                st.dataframe(
                    validation_result["by_source"].rename(
                        columns={
                            "source": "Source",
                            "customers": "Customers",
                            "actual_ltv": "Actual LTV",
                            "km_ltv": "KM prediction",
                            "cox_ltv": "Cox prediction",
                            "aft_ltv": "AFT prediction",
                            "km_mae": "KM MAE",
                            "cox_mae": "Cox MAE",
                            "aft_mae": "AFT MAE",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        else:
            st.info("Choose a validation horizon and run the holdout comparison.")

        st.divider()
        rolling_folds = st.select_slider(
            "Rolling holdout folds",
            options=[2, 3, 4],
            value=3,
        )
        rolling_signature = (
            validation_horizon,
            rolling_folds,
            str(asof_ts),
            tuple(sorted(source_filter)),
            tuple(sorted(plan_filter)),
            len(validation_data),
            str(validation_data["created_at"].min()),
            str(validation_data["created_at"].max()),
        )
        if st.button("Run rolling validation", key="run_rolling_validation"):
            try:
                with st.spinner("Evaluating consecutive mature signup-cohort windows..."):
                    rolling_results = evaluate_survival_models_rolling(
                        validation_data,
                        asof_ts,
                        validation_horizon,
                        n_folds=rolling_folds,
                    )
                st.session_state["rolling_validation_result"] = {
                    "signature": rolling_signature,
                    "results": rolling_results,
                    "error": None,
                }
            except (ConvergenceError, ValueError) as error:
                st.session_state["rolling_validation_result"] = {
                    "signature": rolling_signature,
                    "results": None,
                    "error": str(error),
                }

        rolling_result = st.session_state.get("rolling_validation_result")
        if rolling_result and rolling_result["signature"] == rolling_signature:
            if rolling_result["error"]:
                st.warning(f"Rolling validation unavailable: {rolling_result['error']}")
            else:
                fold_results = rolling_result["results"]
                completed_folds = fold_results[fold_results["status"] == "complete"]
                if completed_folds.empty:
                    st.warning("No rolling fold had enough mature observations and training follow-up.")
                else:
                    st.caption(
                        "Each fold trains before its signup window and tests on the next non-overlapping, "
                        "fully mature window. Fold metrics are comparable within this dataset; means below "
                        "are unweighted averages across completed folds."
                    )
                    fold_km, fold_cox, fold_aft = st.columns(3)
                    fold_km.metric(
                        "Average fold MAE / RMSE",
                        f"{completed_folds['km_mae'].mean():,.2f} / "
                        f"{completed_folds['km_rmse'].mean():,.2f}",
                        help="Kaplan–Meier baseline",
                    )
                    fold_cox.metric(
                        "Average fold MAE / RMSE",
                        f"{completed_folds['cox_mae'].mean():,.2f} / "
                        f"{completed_folds['cox_rmse'].mean():,.2f}",
                        help="Cox proportional hazards",
                    )
                    fold_aft.metric(
                        "Average fold MAE / RMSE",
                        f"{completed_folds['aft_mae'].mean():,.2f} / "
                        f"{completed_folds['aft_rmse'].mean():,.2f}",
                        help="Weibull accelerated failure time",
                    )
                    display_folds = fold_results.rename(
                        columns={
                            "fold": "Fold",
                            "status": "Status",
                            "holdout_start": "Signup window start",
                            "holdout_end": "Signup window end",
                            "train_cutoff": "Training cutoff",
                            "train_n": "Training N",
                            "train_events": "Training cancellations",
                            "holdout_n": "Mature holdout N",
                            "scored_n": "Scored N",
                            "coverage": "Scored coverage",
                            "km_mae": "KM MAE",
                            "cox_mae": "Cox MAE",
                            "aft_mae": "AFT MAE",
                            "km_rmse": "KM RMSE",
                            "cox_rmse": "Cox RMSE",
                            "aft_rmse": "AFT RMSE",
                            "cox_minus_km_mae": "Cox - KM MAE",
                            "aft_minus_km_mae": "AFT - KM MAE",
                            "cox_delta_mae_ci_low": "Cox delta CI low",
                            "cox_delta_mae_ci_high": "Cox delta CI high",
                            "aft_delta_mae_ci_low": "AFT delta CI low",
                            "aft_delta_mae_ci_high": "AFT delta CI high",
                            "cox_events_per_parameter": "Cox events / coefficient",
                            "error": "Note",
                        }
                    )
                    st.dataframe(display_folds, hide_index=True, width="stretch")
                    if (completed_folds["cox_events_per_parameter"] < 10).any():
                        st.warning(
                            "At least one fold has fewer than 10 Cox cancellation events per coefficient; "
                            "interpret its Cox scores as exploratory."
                        )
        else:
            st.info("Run rolling validation to see how model performance changes across signup cohorts.")
else:
    st.info("The dashboard is ready. Choose a CSV in the sidebar to compare LTV by acquisition source.")
