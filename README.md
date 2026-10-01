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
