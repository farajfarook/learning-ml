# load CSV data
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the training data
train_data = pd.read_csv("data/train.csv")

# Split the training data into features and target
y = train_data["Survived"]
x = train_data.drop(columns=["Survived"])

# Split into train and validation sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# get all non numerical columns
non_numerical_columns = x_train.select_dtypes(exclude=["number"]).columns
print(f"Non-numerical columns: {non_numerical_columns.tolist()}")

encoder = OneHotEncoder(
    sparse_output=False,
    drop="first",  # drop first to avoid dummy variable trap
)

x_train_encoded = encoder.fit_transform(x_train["Sex"])
x_test_encoded = encoder.transform(x_test[])

# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(x_train, y_train)
# sgd_score = sgd_clf.score(x_test, y_test)
# print(f"SGD Classifier score: {sgd_score:.4f}")
