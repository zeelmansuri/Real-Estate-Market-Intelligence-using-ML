# Machine Learning Based Buyer Segmentation and Investment Profiling

This project segments real estate buyers for market intelligence using clustering models, customer profiling, and an interactive Streamlit dashboard.
Video Link - https://screenapp.io/app/v/0ugf49YEum
Project Link - 
## Features

- Data cleaning for buyer attributes and duplicate client records
- Age derivation from `date_of_birth`
- Categorical encoding and numeric scaling
- K-Means clustering with elbow and silhouette diagnostics
- Hierarchical clustering validation
- Buyer segment interpretation
- Streamlit dashboard with country, region, purpose, and client-type filters

## Expected Dataset Fields

`client_id`, `client_type`, `gender`, `country`, `region`, `date_of_birth`, `acquisition_purpose`, `loan_applied`, `referral_channel`, `satisfaction_score`

Optional fields such as `income`, `property_value`, or `units_purchased` are used automatically if present.

## Run The Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload the real dataset CSV in the sidebar. If no file is uploaded, the app uses a deterministic sample dataset so every section remains usable.

## Deliverables

- `app.py` - Streamlit analytics dashboard
- `src/segmentation.py` - reusable machine learning pipeline
- `outputs/research_paper.md` - concise research paper with methodology, insights, and recommendations
