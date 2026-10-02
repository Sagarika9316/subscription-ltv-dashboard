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

## Optional contribution-margin input

The Streamlit dashboard can also show contribution LTV when the CSV includes
`net_contribution_per_charge`. This is the net amount per successful scheduled
charge after variable service/payment costs and refunds allocated to that
charge. Use the same currency as `price`; do not subtract CAC in this field,
because CAC is compared separately. If the column is absent, contribution LTV
is marked unavailable rather than assuming zero costs. Kaplan-Meier and Cox
forecasts remain gross-revenue forecasts.
