from pathlib import Path

import tarfile
import urllib

import pandas as pd


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.exists():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as tar:
            tar.extractall(path=tarball_path.parent)
    return pd.read_csv(f"{tarball_path.parent}/housing/housing.csv")
