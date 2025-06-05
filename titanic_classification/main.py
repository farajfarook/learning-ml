# load CSV data
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from preprocessor import create_preprocessor
from sklearn.model_selection import RandomizedSearchCV

# Load the training data
train_data = pd.read_csv("data/train.csv")

# Split the training data into features and target
y = train_data["Survived"]
x = train_data.drop(columns=["Survived"])

mask = x["Embarked"].notna()
x = x[mask]
y = y[mask]

# Split into train and validation sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

preprocessor = create_preprocessor()

# Create a pipeline with the transformer and SGDClassifier
pipeline = make_pipeline(preprocessor, SGDClassifier(random_state=42))

param_distributions = {
    "sgdclassifier__loss": ["hinge", "log", "modified_huber"],
    "sgdclassifier__penalty": ["l2", "l1", "elasticnet"],
    "sgdclassifier__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "sgdclassifier__max_iter": [1000, 2000, 3000],
}
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="accuracy",
    cv=5,
    random_state=42,
    verbose=1,
)

# Fit the random search on the training data
random_search.fit(x_train, y_train)
# Print the best parameters and score
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", random_search.best_score_)
