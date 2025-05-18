import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from cluster_similarity import ClusterSimilarity


# Pipeline for the ratio of two features
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(
            lambda X: X[:, [0]] / X[:, [1]],
            feature_names_out=ratio_name,
        ),
    )


# Pipeline for log transformation
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(
        np.log,
        feature_names_out="one-to-one",  # using one-to-one to keep the same number of features
    ),
    StandardScaler(),
)


# Pipeline for numeric features
def num_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )


# Pipeline for categorical features
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)


# Cluster similarity for geographical features
cluster_similarity = ClusterSimilarity(
    n_clusters=10,
    gamma=1,
    random_state=42,
)


# Default numeric pipeline
def default_numeric_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )


def create_preprocessor():
    return ColumnTransformer(
        [
            ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline(), ["population", "households"]),
            (
                "log",
                log_pipeline,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("geo", cluster_similarity, ["latitude", "longitude"]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ],
        remainder=default_numeric_pipeline(),
    )
