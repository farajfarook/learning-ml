# load CSV data
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessor import create_preprocessor

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

# Hyperparameter tuning with RandomizedSearchCV

param_distributions = {
    "sgdclassifier__loss": ["hinge", "log_loss", "modified_huber"],
    "sgdclassifier__alpha": [0.0001, 0.001, 0.01, 0.1],
    "sgdclassifier__learning_rate": ["constant", "optimal", "invscaling"],
    "sgdclassifier__eta0": [0.01, 0.1, 1.0],
    "sgdclassifier__max_iter": [1000, 2000, 3000],
}

# Perform randomized search
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

print("Training model with hyperparameter tuning...")
random_search.fit(x_train, y_train)

# Get the best model
best_pipeline = random_search.best_estimator_
print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

# Evaluate on validation set
y_pred = best_pipeline.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Make predictions on test data
print("\nMaking predictions on test data...")
test_data = pd.read_csv("data/test.csv")

# Make predictions
test_predictions = best_pipeline.predict(test_data)

# Create submission file
submission = pd.DataFrame(
    {"PassengerId": test_data["PassengerId"], "Survived": test_predictions}
)

submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")

# Display first few predictions
print("\nFirst 10 predictions:")
print(submission.head(10))
