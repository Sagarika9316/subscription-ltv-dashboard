# Streamlit deployment notes

## Local run

```bash
python -m streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open https://share.streamlit.io/
3. Connect the repo.
4. Set the main file to `streamlit_app.py`.
5. Deploy.

## Suggested public app title

LTV Source Comparison

## Notes

- This app compares expected lifetime value by acquisition source.
- It works with subscription data containing at least: `plan`, `channel`, `utm_campaign`, `created_at`, `ended_at`, `end_reason`.
- It assumes two subscription plans: monthly and annual, with the business rules encoded in the LTV model.
