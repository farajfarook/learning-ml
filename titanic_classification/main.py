# load CSV data
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the training data
train_data = pd.read_csv("data/train.csv")

# Split the training data into features and target
y = train_data["Survived"]
x = train_data.drop(columns=["Survived"])

# Split into train and validation sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# take the first letter of cabin - the deck and create a new column and remove the original 'Cabin' column
x_train["Deck"] = x_train["Cabin"].str[0]
x_train = x_train.drop(columns=["Cabin"])

# Extract Mr, Master, Msr, Miss, Mrs, and Ms from the 'Name' column. Example Name: "Braund, Mr. Owen Harris"
x_train["Title"] = x_train["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)

# remove rows with missing values in 'Embarked' column
x_train = x_train.dropna(subset=["Embarked"])
y_train = y_train[x_train.index]  # Ensure y_train matches x_train after dropping rows

# Remove columns that are not needed for the model
x_train = x_train.drop(columns=["Name", "Ticket", "PassengerId"])

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
# Create a pipeline with the transformer and SGDClassifier
pipeline = make_pipeline(transformer, SGDClassifier(random_state=42))

# Fit the pipeline on the training data
pipeline.fit(x_train, y_train)
# Evaluate the model on the test set


x_test = transformer.transform(x_test)
sgd_score = pipeline.score(x_test, y_test)
print(f"SGD Classifier score: {sgd_score:.4f}")
