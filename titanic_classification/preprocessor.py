from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
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

        # Remove unnecessary columns
        X = X.drop(columns=["Name", "Ticket", "PassengerId"])

        return X


transformer = ColumnTransformer(
    [
        # One-hot encode for categorical variables - Sex and Embarked
        (
            "categorical",
            OneHotEncoder(sparse_output=False, drop="first"),
            ["Sex", "Embarked", "Title"],
        ),
        # Deck: Impute missing values with "U" the most frequent value and one-hot encode
        (
            "deck",
            make_pipeline(
                SimpleImputer(strategy="constant", fill_value="U"),
                OneHotEncoder(sparse_output=False, drop="first"),
            ),
            ["Deck"],
        ),
        # pclass - One-hot encode
        (
            "pclass",
            OneHotEncoder(sparse_output=False, drop="first"),
            ["Pclass"],
        ),
    ],
    # Numerical columns: Age, Fare, SibSp, Parch
    remainder=make_pipeline(
        SimpleImputer(strategy="mean"),  # Impute Mean
        StandardScaler(),  # Standardize numerical columns
    ),
)


def create_preprocessor():
    """
    Create a transformer for the Titanic dataset.

    Returns:
        ColumnTransformer: A transformer that preprocesses the Titanic dataset.
    """
    return make_pipeline(TitanicPreprocessor(), transformer)
