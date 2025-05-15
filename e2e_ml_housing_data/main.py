from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

from sklearn.pipeline import FunctionTransformer
from ramdom_split import split_train_test
from hash_split import split_train_test_by_id_hash
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.metrics.pairwise import rbf_kernel
from cluster_similarity import ClusterSimilarity


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.exists():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as tar:
            tar.extractall(path=tarball_path.parent)
    return pd.read_csv(f"{tarball_path.parent}/housing/housing.csv")


housing = load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
#
# fig = housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# train_set, test_set = split_train_test(housing, 0.2)

# Copy the original DataFrame to avoid modifying it
# housing_with_id = housing.copy()
# Create a unique identifier for each row by combining longitude and latitude
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_train_test_by_id_hash(housing_with_id, 0.2, "id")
# print(f"train_set: {len(train_set)}")
# print(f"test_set: {len(test_set)}")

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel("Income Category")
# plt.ylabel("Number of Districts")
# plt.title("Income Category Distribution")
# plt.show()

# splitter = StratifiedShuffleSplit(
#    n_splits=1,  # Number of re-shuffling iterations
#    test_size=0.2,  # Proportion of the dataset to include in the test split
#    random_state=42,  # Controls the randomness of the split
# )
#
# strat_splits = []
# for train_index, test_index in splitter.split(housing, housing["income_cat"]):
#    strat_train_set_n = housing.iloc[train_index]
#    strat_test_set_n = housing.iloc[test_index]
#    strat_splits.append((strat_train_set_n, strat_test_set_n))
#
# strat_train_set, strat_test_set = strat_splits[0]

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
# housing.plot(
#    kind="scatter",
#    x="longitude",
#    y="latitude",
#    # alpha=0.2,  # Transparency for better visibility
#    grid=True,
#    s=housing["population"] / 100,  # Scale the size of the points
#    label="population",
#    c="median_house_value",
#    cmap="jet",
#    colorbar=True,  # Show color scale
#    legend=True,  # Show legend
#    sharex=False,  # Do not share x-axis with other subplots.
#    figsize=(10, 7),  # Set figure size
# )
# plt.show()

housing_num = housing.select_dtypes(include=[np.number])
corr_matrix = housing_num.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


# attributes = [
#    "median_house_value",
#    "median_income",
#    "total_rooms",
#    "housing_median_age",
# ]
# scatter_matrix = pd.plotting.scatter_matrix(
#    housing[attributes],
#    figsize=(12, 8),
#    grid=True,
# )

# housing.plot(
#    kind="scatter",
#    x="median_income",
#    y="median_house_value",
#    grid=True,
#    alpha=0.1,  # Transparency for better visibility
# )
# plt.xlabel("Median Income")
# plt.show()


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

housing_num = housing.select_dtypes(include=[np.number])
corr_matrix = housing_num.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Drop the original columns after creating new features
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Clean up the data
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
X = imputer.transform(housing_num)
# The result is a NumPy array, so we need to convert it back to a DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

housing_cat = housing[["ocean_proximity"]]
# Head of the DataFrame.
print(housing_cat.head(8))
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:8])
# print(ordinal_encoder.categories_)

# One-hot encoding
one_hot_encoder = OneHotEncoder()
housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat)
print(housing_cat_1hot.toarray()[:8])
print(one_hot_encoder.categories_)

# Scaling and Normalizing
# min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
# housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
# print(housing_num_min_max_scaled[:8])

# Standardization
standard_scaler = StandardScaler()
housing_num_standardized = standard_scaler.fit_transform(housing_num)
print(housing_num_standardized[:8])
print(standard_scaler.mean_)
print(standard_scaler.scale_)

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
print(age_simil_35[:8])

log_transformer = FunctionTransformer(func=np.log, inverse_func=np.exp)
log_pop = log_transformer.fit_transform(housing["population"])

print(log_pop[:8])


rbf_transformer = FunctionTransformer(
    func=rbf_kernel, kw_args={"gamma": 0.1, "Y": [[35]]}
)

age_simil_35 = rbf_transformer.fit_transform(housing[["housing_median_age"]])
print(age_simil_35[:8])


sf_coords = 37.7749, -122.4194
sf_transformer = FunctionTransformer(
    func=rbf_kernel, kw_args={"gamma": 0.1, "Y": [sf_coords]}
)
sf_distances = sf_transformer.fit_transform(housing[["latitude", "longitude"]])
print(sf_distances[:8])

ratio_transformer = FunctionTransformer(func=lambda X: X[:, [0]] / X[:, [1]])
ratios = ratio_transformer.fit_transform(np.array([[1, 2], [3, 4], [5, 6]]))
print(ratios[:8])

# Custom transformer
cluster_similarity = ClusterSimilarity(
    n_clusters=10,
    gamma=1.0,
    random_state=42,
)
similarities = cluster_similarity.fit_transform(
    housing[["latitude", "longitude"]], sample_weight=housing_labels
)

# Print the first 3 rows of the transformed data. Rounded to 2 decimal places.
print(similarities[:3].round(2))
similarities_centers = cluster_similarity.kmeans_.cluster_centers_
print(similarities_centers.round(2))

#
