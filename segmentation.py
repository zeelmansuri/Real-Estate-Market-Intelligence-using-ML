from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = [
    "client_id",
    "client_type",
    "gender",
    "country",
    "region",
    "date_of_birth",
    "acquisition_purpose",
    "loan_applied",
    "referral_channel",
    "satisfaction_score",
]

SEGMENT_NAMES = {
    "global_investors": "Global Investors",
    "first_time_buyers": "First-Time Buyers",
    "corporate_buyers": "Corporate Buyers",
    "luxury_investors": "Luxury Investors",
}


@dataclass(frozen=True)
class SegmentationResult:
    data: pd.DataFrame
    feature_matrix: np.ndarray
    elbow: pd.DataFrame
    silhouette: pd.DataFrame
    profile: pd.DataFrame


def generate_sample_data(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    archetypes = rng.choice(
        ["global", "first_time", "corporate", "luxury"],
        rows,
        p=[0.28, 0.32, 0.18, 0.22],
    )

    countries = {
        "global": ["UAE", "Singapore", "United Kingdom", "Canada", "United States"],
        "first_time": ["United States", "Canada", "India", "United Kingdom"],
        "corporate": ["United States", "UAE", "Singapore", "Germany"],
        "luxury": ["UAE", "Singapore", "United Kingdom", "United States"],
    }
    regions = {
        "global": ["Middle East", "Europe", "Asia Pacific", "North America"],
        "first_time": ["North America", "Asia Pacific", "Europe"],
        "corporate": ["North America", "Middle East", "Asia Pacific", "Europe"],
        "luxury": ["Middle East", "Europe", "North America", "Asia Pacific"],
    }
    channels = ["Broker", "Digital Ads", "Referral", "Property Expo", "Organic Search"]

    records = []
    for idx, kind in enumerate(archetypes, start=1):
        if kind == "first_time":
            age = int(rng.normal(31, 5))
            client_type = "Individual"
            purpose = rng.choice(["Personal use", "Investment"], p=[0.75, 0.25])
            loan = rng.choice(["Yes", "No"], p=[0.78, 0.22])
            satisfaction = np.clip(rng.normal(7.1, 1.2), 1, 10)
            income = rng.normal(85000, 22000)
            property_value = rng.normal(420000, 90000)
            units = 1
        elif kind == "corporate":
            age = int(rng.normal(46, 8))
            client_type = "Corporate"
            purpose = "Investment"
            loan = rng.choice(["Yes", "No"], p=[0.38, 0.62])
            satisfaction = np.clip(rng.normal(7.7, 0.9), 1, 10)
            income = rng.normal(420000, 110000)
            property_value = rng.normal(1100000, 260000)
            units = int(np.clip(rng.normal(4, 1.5), 2, 8))
        elif kind == "luxury":
            age = int(rng.normal(52, 9))
            client_type = "Individual"
            purpose = "Investment"
            loan = rng.choice(["Yes", "No"], p=[0.18, 0.82])
            satisfaction = np.clip(rng.normal(9.0, 0.6), 1, 10)
            income = rng.normal(580000, 140000)
            property_value = rng.normal(1800000, 350000)
            units = int(rng.choice([1, 2, 3], p=[0.55, 0.35, 0.10]))
        else:
            age = int(rng.normal(43, 8))
            client_type = "Individual"
            purpose = "Investment"
            loan = rng.choice(["Yes", "No"], p=[0.34, 0.66])
            satisfaction = np.clip(rng.normal(8.0, 0.8), 1, 10)
            income = rng.normal(260000, 80000)
            property_value = rng.normal(850000, 190000)
            units = int(rng.choice([1, 2, 3], p=[0.70, 0.23, 0.07]))

        birth_year = date.today().year - max(age, 18)
        records.append(
            {
                "client_id": f"C{idx:05d}",
                "client_type": client_type,
                "gender": rng.choice(["Female", "Male", "Non-binary", "Prefer not to say"], p=[0.43, 0.47, 0.04, 0.06]),
                "country": rng.choice(countries[kind]),
                "region": rng.choice(regions[kind]),
                "date_of_birth": f"{birth_year}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}",
                "acquisition_purpose": purpose,
                "loan_applied": loan,
                "referral_channel": rng.choice(channels),
                "satisfaction_score": round(float(satisfaction), 1),
                "income": round(float(max(income, 25000)), 2),
                "property_value": round(float(max(property_value, 150000)), 2),
                "units_purchased": units,
            }
        )
    return pd.DataFrame(records)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip().lower().replace(" ", "_") for column in df.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    df = df.drop_duplicates(subset=["client_id"], keep="first")
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()
        df[column] = df[column].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    categorical_columns = [
        "client_type",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel",
    ]
    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown").str.title()

    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median()).clip(1, 10)
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    current_year = date.today().year
    df["age"] = current_year - df["date_of_birth"].dt.year
    df["age"] = df["age"].where(df["age"].between(18, 100), np.nan)
    df["age"] = df["age"].fillna(df["age"].median()).round().astype(int)

    for column in ["income", "property_value", "units_purchased"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            df[column] = df[column].fillna(df[column].median())

    return df


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = ["age", "satisfaction_score"]
    for optional in ["income", "property_value", "units_purchased"]:
        if optional in df.columns:
            numeric.append(optional)

    categorical = [
        "client_type",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel",
    ]
    return numeric, categorical


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer]:
    numeric, categorical = _feature_columns(df)
    encoder_kwargs = {"handle_unknown": "ignore"}
    try:
        encoder = OneHotEncoder(sparse_output=False, **encoder_kwargs)
    except TypeError:
        encoder = OneHotEncoder(sparse=False, **encoder_kwargs)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", encoder, categorical),
        ],
        remainder="drop",
    )
    matrix = preprocessor.fit_transform(df)
    return matrix, preprocessor


def evaluate_clusters(matrix: np.ndarray, cluster_range: Iterable[int] = range(2, 9)) -> tuple[pd.DataFrame, pd.DataFrame]:
    elbow_rows = []
    silhouette_rows = []
    max_k = min(matrix.shape[0] - 1, max(cluster_range))
    for k in [value for value in cluster_range if value <= max_k]:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(matrix)
        elbow_rows.append({"k": k, "inertia": model.inertia_})
        silhouette_rows.append({"k": k, "silhouette_score": silhouette_score(matrix, labels)})
    return pd.DataFrame(elbow_rows), pd.DataFrame(silhouette_rows)


def assign_segment_labels(df: pd.DataFrame, cluster_col: str = "cluster") -> dict[int, str]:
    summaries = df.groupby(cluster_col).agg(
        client_type_corporate=("client_type", lambda value: (value == "Corporate").mean()),
        investment_share=("acquisition_purpose", lambda value: (value == "Investment").mean()),
        loan_share=("loan_applied", lambda value: (value == "Yes").mean()),
        satisfaction=("satisfaction_score", "mean"),
        age=("age", "mean"),
        property_value=("property_value", "mean") if "property_value" in df.columns else ("satisfaction_score", "mean"),
    )

    labels: dict[int, str] = {}
    remaining = set(summaries.index.tolist())

    if remaining:
        corporate = summaries.loc[list(remaining), "client_type_corporate"].idxmax()
        labels[int(corporate)] = SEGMENT_NAMES["corporate_buyers"]
        remaining.remove(corporate)
    if remaining:
        first_time = summaries.loc[list(remaining)].assign(
            score=summaries.loc[list(remaining), "loan_share"] - summaries.loc[list(remaining), "age"] / 100
        )["score"].idxmax()
        labels[int(first_time)] = SEGMENT_NAMES["first_time_buyers"]
        remaining.remove(first_time)
    if remaining:
        luxury = summaries.loc[list(remaining)].assign(
            score=summaries.loc[list(remaining), "satisfaction"] + summaries.loc[list(remaining), "property_value"].rank(pct=True)
        )["score"].idxmax()
        labels[int(luxury)] = SEGMENT_NAMES["luxury_investors"]
        remaining.remove(luxury)
    for cluster in remaining:
        labels[int(cluster)] = SEGMENT_NAMES["global_investors"]

    return labels


def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    profile = df.groupby("segment").agg(
        buyers=("client_id", "count"),
        avg_age=("age", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        loan_rate=("loan_applied", lambda value: (value == "Yes").mean()),
        investment_rate=("acquisition_purpose", lambda value: (value == "Investment").mean()),
        top_region=("region", lambda value: value.mode().iat[0] if not value.mode().empty else "Unknown"),
        top_country=("country", lambda value: value.mode().iat[0] if not value.mode().empty else "Unknown"),
    )
    if "property_value" in df.columns:
        profile["avg_property_value"] = df.groupby("segment")["property_value"].mean()
    return profile.reset_index().sort_values("buyers", ascending=False)


def run_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> SegmentationResult:
    cleaned = clean_data(df)
    matrix, _ = build_feature_matrix(cleaned)
    elbow, silhouette = evaluate_clusters(matrix)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cleaned["cluster"] = model.fit_predict(matrix)
    hierarchy = AgglomerativeClustering(n_clusters=n_clusters)
    cleaned["hierarchical_cluster"] = hierarchy.fit_predict(matrix)

    labels = assign_segment_labels(cleaned)
    cleaned["segment"] = cleaned["cluster"].map(labels)
    profile = profile_segments(cleaned)

    return SegmentationResult(
        data=cleaned,
        feature_matrix=matrix,
        elbow=elbow,
        silhouette=silhouette,
        profile=profile,
    )
