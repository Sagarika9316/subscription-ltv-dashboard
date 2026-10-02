# LTV Project

This project estimates lifetime value (LTV) by acquisition channel and plan using subscription data in `subscriptions.csv`.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

## Outputs

Running the script produces:

- `ltv_by_horizon.csv`
- `ltv_all_cuts.csv`
- `subscription_survival.png`

The script calculates renewal eligibility and cohort-level LTV for several time horizons.

## Subscription billing fields

For data-specific billing, include these columns in the CSV:

- `price`: numeric amount for one scheduled subscription charge.
- `billing_interval_months`: positive whole number of months between charges (for example, `1` monthly or `12` annual).
- `currency`: one three-letter currency code for the entire upload (for example, `USD`). The app does not convert currencies.

The first charge is assumed to occur at signup, with subsequent charges at each billing interval. If these fields are absent, the app uses a hypothetical USD schedule: monthly at $15 every 1 month and annual at $150 every 12 months. At horizon H, revenue includes scheduled charges strictly before month H, not a charge exactly at H. Make sure these assumptions match the actual billing contract before using the results for business decisions.

## Optional illustrative CAC assumptions

The Streamlit CAC editor is blank by default. Selecting **Use illustrative static CAC assumptions** pre-fills these example USD-per-subscriber values:

| Source | Illustrative CAC |
| --- | ---: |
| email | $20 |
| direct | $40 |
| organic_search | $30 |
| organic_social | $25 |
| paid_social/lookalike_subscribers | $60 |
| paid_social/prospecting_broad | $60 |

These values are hypothetical examples, not measured acquisition costs. Edit or replace them with verified spend before interpreting the CAC ranking. The default 1.0x gross LTV:CAC screen means gross revenue only matches CAC; it does not establish profit because variable costs are not deducted. A passing source is a candidate for a controlled test, not an instruction to increase spend.

## Optional contribution-margin input

The Streamlit dashboard can also show contribution LTV when the CSV includes
`net_contribution_per_charge`. This is the net amount per successful scheduled
charge after variable service/payment costs and refunds allocated to that
charge. Use the same currency as `price`; do not subtract CAC in this field,
because CAC is compared separately. If the column is absent, contribution LTV
is marked unavailable rather than assuming zero costs. Kaplan-Meier and Cox
forecasts remain gross-revenue forecasts.

## Incrementality experiment planner

The Streamlit planner estimates the eligible audience required **per arm** for
a randomized two-arm test of paid-subscription conversion. Supply a historical
control conversion rate and the smallest absolute lift worth detecting. Alpha
and power are design choices (defaults: 5% and 80%); the planner uses an
approximate two-sided normal test with equal-sized, independently randomized
arms. The subscription dataset has no exposed-prospect denominator, treatment
assignment, or experiment outcome, so these values must come from campaign or
experiment records. The planner sizes a test; it does not estimate causal lift
or recommend increasing spend.

When a positive CAC is available for the selected source, the planner also
shows an illustrative value-after-CAC scenario at the target lift. It assumes
the lift is achieved and CAC per acquired subscriber stays constant. If
`net_contribution_per_charge` is present it uses contribution LTV; otherwise it
uses gross LTV and labels the result **not profit**. This scenario is not an
observed experiment outcome and does not include unmodeled changes in media
spend, costs, or conversion behavior.
