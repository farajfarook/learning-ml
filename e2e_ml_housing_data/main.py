from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
from ramdom_split import split_train_test
from hash_split import split_train_test_by_id_hash
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


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

numeric_columns = housing.select_dtypes(include=["number"])
corr_matrix = numeric_columns.corr()
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

numeric_columns = housing.select_dtypes(include=["number"])
corr_matrix = numeric_columns.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
