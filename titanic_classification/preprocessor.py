from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler  # Fixed import
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Extract deck from cabin
        X["Deck"] = X["Cabin"].str[0]
        X = X.drop(columns=["Cabin"])

        # Extract title from name
        X["Title"] = X["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
        common_titles = ["Mr", "Miss", "Mrs", "Master"]
        X["Title"] = X["Title"].apply(
            lambda x: x if x in common_titles else "Other"
        )  # Remove unnecessary columns
        X = X.drop(columns=["Name", "Ticket", "PassengerId"])

        return X


def create_preprocessor():
    return make_pipeline(
        TitanicPreprocessor(),
        ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(
                        sparse_output=False, drop="first", handle_unknown="ignore"
                    ),
                    ["Sex", "Embarked", "Title"],
                ),
                (
                    "deck",
                    make_pipeline(
                        SimpleImputer(strategy="constant", fill_value="U"),
                        OneHotEncoder(
                            sparse_output=False, drop="first", handle_unknown="ignore"
                        ),
                    ),
                    ["Deck"],
                ),
                (
                    "pclass",
                    OneHotEncoder(
                        sparse_output=False, drop="first", handle_unknown="ignore"
                    ),
                    ["Pclass"],
                ),
            ],
            remainder=make_pipeline(
                SimpleImputer(strategy="mean"),
                StandardScaler(),
            ),
        ),
    )
