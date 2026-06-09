from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.segmentation import generate_sample_data, run_segmentation


st.set_page_config(
    page_title="Buyer Segmentation Intelligence",
    page_icon="🏢",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def cached_segmentation(data: pd.DataFrame, clusters: int):
    return run_segmentation(data, n_clusters=clusters)


def multiselect_filter(label: str, values: pd.Series) -> list[str]:
    options = sorted(values.dropna().unique().tolist())
    return st.sidebar.multiselect(label, options=options, default=options)


st.title("Buyer Segmentation and Investment Profiling")

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload buyer dataset CSV", type=["csv"])
cluster_count = st.sidebar.slider("Clusters", min_value=2, max_value=8, value=4, step=1)

raw_df = load_uploaded_csv(uploaded) if uploaded else generate_sample_data()
result = cached_segmentation(raw_df, cluster_count)
df = result.data

countries = multiselect_filter("Country", df["country"])
regions = multiselect_filter("Region", df["region"])
purposes = multiselect_filter("Acquisition purpose", df["acquisition_purpose"])
client_types = multiselect_filter("Client type", df["client_type"])

filtered = df[
    df["country"].isin(countries)
    & df["region"].isin(regions)
    & df["acquisition_purpose"].isin(purposes)
    & df["client_type"].isin(client_types)
]

metric_cols = st.columns(4)
metric_cols[0].metric("Buyers", f"{len(filtered):,}")
metric_cols[1].metric("Segments", filtered["segment"].nunique())
metric_cols[2].metric("Loan Rate", f"{(filtered['loan_applied'].eq('Yes').mean() if len(filtered) else 0):.1%}")
metric_cols[3].metric("Avg Satisfaction", f"{(filtered['satisfaction_score'].mean() if len(filtered) else 0):.1f}")

tab_overview, tab_behavior, tab_geo, tab_insights, tab_model = st.tabs(
    ["Segmentation Overview", "Investor Behavior", "Geographic Analysis", "Segment Insights", "Model Diagnostics"]
)

with tab_overview:
    left, right = st.columns([1, 1])
    with left:
        counts = filtered["segment"].value_counts().reset_index()
        counts.columns = ["segment", "buyers"]
        st.plotly_chart(
            px.bar(counts, x="segment", y="buyers", color="segment", title="Cluster Distribution"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.scatter(
                filtered,
                x="age",
                y="satisfaction_score",
                color="segment",
                symbol="client_type",
                hover_data=["country", "region", "acquisition_purpose", "loan_applied"],
                title="Buyer Age vs Satisfaction",
            ),
            use_container_width=True,
        )

with tab_behavior:
    purpose = filtered.groupby(["segment", "acquisition_purpose"]).size().reset_index(name="buyers")
    loan = filtered.groupby(["segment", "loan_applied"]).size().reset_index(name="buyers")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            px.bar(purpose, x="segment", y="buyers", color="acquisition_purpose", barmode="group", title="Investment Purpose by Segment"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.bar(loan, x="segment", y="buyers", color="loan_applied", barmode="group", title="Loan Behavior by Segment"),
            use_container_width=True,
        )
    if "property_value" in filtered.columns:
        st.plotly_chart(
            px.box(filtered, x="segment", y="property_value", color="segment", title="Property Value Distribution"),
            use_container_width=True,
        )

with tab_geo:
    geo = filtered.groupby(["region", "segment"]).size().reset_index(name="buyers")
    country_geo = filtered.groupby(["country", "segment"]).size().reset_index(name="buyers")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            px.bar(geo, x="region", y="buyers", color="segment", title="Buyer Segments by Region"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.treemap(country_geo, path=["segment", "country"], values="buyers", title="Country Concentration by Segment"),
            use_container_width=True,
        )

with tab_insights:
    st.subheader("Segment Profile")
    display_profile = result.profile.copy()
    for column in ["avg_age", "avg_satisfaction", "loan_rate", "investment_rate", "avg_property_value"]:
        if column in display_profile.columns:
            display_profile[column] = display_profile[column].round(2)
    st.dataframe(display_profile, use_container_width=True, hide_index=True)

    st.subheader("Buyer Records")
    visible_columns = [
        "client_id",
        "segment",
        "client_type",
        "country",
        "region",
        "age",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel",
        "satisfaction_score",
    ]
    optional_columns = [column for column in ["income", "property_value", "units_purchased"] if column in filtered.columns]
    st.dataframe(filtered[visible_columns + optional_columns], use_container_width=True, hide_index=True)

with tab_model:
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(px.line(result.elbow, x="k", y="inertia", markers=True, title="Elbow Method"), use_container_width=True)
    with right:
        st.plotly_chart(px.line(result.silhouette, x="k", y="silhouette_score", markers=True, title="Silhouette Score"), use_container_width=True)

    agreement = pd.crosstab(df["cluster"], df["hierarchical_cluster"])
    st.subheader("K-Means vs Hierarchical Cluster Cross-Tab")
    st.dataframe(agreement, use_container_width=True)
