from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
from ramdom_split import split_train_test
from hash_split import split_train_test_by_id_hash


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
housing_with_id = housing.copy()
# Create a unique identifier for each row by combining longitude and latitude
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id_hash(housing_with_id, 0.2, "id")
print(f"train_set: {len(train_set)}")
print(f"test_set: {len(test_set)}")

# Page 108
