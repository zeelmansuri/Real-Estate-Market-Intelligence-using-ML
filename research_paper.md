# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Abstract

This project applies unsupervised machine learning to real estate buyer data to identify actionable buyer segments for marketing, investment targeting, and customer relationship strategy. The workflow cleans client records, engineers buyer age, encodes categorical behavior, scales numeric features, evaluates cluster quality, and profiles each segment using K-Means and hierarchical clustering.

## Business Context

Real estate buyers differ by acquisition purpose, geography, financing behavior, customer type, and satisfaction. Treating all buyers as one market leads to inefficient marketing spend, generic recommendations, and missed investment opportunities. Buyer segmentation helps Parcl identify which audiences are loan dependent, which are investment motivated, and which regions contain high-value or institutional demand.

## Dataset

The required fields are `client_id`, `client_type`, `gender`, `country`, `region`, `date_of_birth`, `acquisition_purpose`, `loan_applied`, `referral_channel`, and `satisfaction_score`. Optional fields such as income, property value, and units purchased improve investment profiling when available.

## Methodology

1. Data cleaning removes duplicate `client_id` records, normalizes categorical labels, handles missing values, and converts `date_of_birth` into age.
2. Feature encoding uses one-hot encoding for categorical variables including client type, country, region, acquisition purpose, loan status, and referral channel.
3. Feature scaling uses standardization for age, satisfaction score, and optional numeric investment fields.
4. K-Means clustering assigns buyers into a configurable number of market segments.
5. Hierarchical clustering is used as a validation view to compare nested grouping behavior against K-Means.
6. Elbow and silhouette diagnostics are calculated to evaluate the number and quality of clusters.
7. Cluster interpretation profiles each group by investment rate, loan rate, age, client type, geography, satisfaction, and property value where available.

## Recommended Buyer Segments

| Segment | Interpretation | Primary Characteristics |
| --- | --- | --- |
| Global Investors | Cross-border, investment-oriented buyers | High investment share, international country mix, moderate loan usage |
| First-Time Buyers | Younger and financing dependent | Lower average age, high loan rate, stronger personal-use behavior |
| Corporate Buyers | Institutional or company purchasers | Corporate client type, investment purpose, potential multi-unit demand |
| Luxury Investors | High-satisfaction, high-value buyers | Higher satisfaction, lower loan reliance, premium property behavior |

## Dashboard Design

The Streamlit application provides five analytical views:

- Segmentation Overview: cluster distribution and buyer age versus satisfaction.
- Investor Behavior: acquisition purpose, loan behavior, and optional property value patterns.
- Geographic Analysis: regional segment distribution and country concentration.
- Segment Insights: descriptive statistics and filtered buyer-level records.
- Model Diagnostics: elbow curve, silhouette score, and K-Means versus hierarchical cross-tab.

## Recommendations

- Target First-Time Buyers with financing partnerships, affordability content, and guided purchase journeys.
- Serve Global Investors with market-entry reports, currency-aware messaging, and remote transaction support.
- Build a corporate account motion for Corporate Buyers, including bulk purchase workflows and portfolio reporting.
- Prioritize Luxury Investors with premium inventory, private advisory, and retention campaigns based on satisfaction.
- Re-run clustering periodically as new buyer records enter the platform because segment definitions can shift with market conditions.

## Conclusion

The project introduces AI-driven buyer intelligence into the Parcl real estate market workflow. Clustering reveals hidden behavioral groups that are difficult to detect through traditional summary reporting alone. These segments can support smarter marketing allocation, stronger property recommendations, and more data-driven real estate investment decisions.
