import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from scipy.stats import randint

from load_data import load_housing_data
from preprocessor import create_preprocessor
from scipy import stats

housing = load_housing_data()

# Stratified sampling based on income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)
train_set, test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)
train_set.drop("income_cat", axis=1, inplace=True)
test_set.drop("income_cat", axis=1, inplace=True)

# Prepare the data for training
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# Create a preprocessor
preprocessor = create_preprocessor()

## Linear Regression
# lin_reg = make_pipeline(preprocessor, LinearRegression())
# lin_reg.fit(housing, housing_labels)
#
## Evaluate the model
# housing_predictions = lin_reg.predict(housing)
# print("Predictions:", housing_predictions[:5].round(2))
# print("Labels:", housing_labels.iloc[:5].values)
# lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
# print("Root Mean Squared Error:", lin_rmse)
#
## Decision Tree Regressor
# tree_reg = make_pipeline(preprocessor, DecisionTreeRegressor())
# tree_rmse_scores = -cross_val_score(
#    tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10
# )
# print("Decision Tree Regressor RMSE Scores:")
# print(pd.Series(tree_rmse_scores).describe())
#
# Random Forest Regressor
# forest_reg = make_pipeline(
#    preprocessor,
#    RandomForestRegressor(random_state=42, n_jobs=-1),  # Use all CPU cores
# )
# forest_rmse_scores = -cross_val_score(
#    forest_reg,
#    housing,
#    housing_labels,
#    scoring="neg_root_mean_squared_error",
#    cv=10,
#    n_jobs=-1,  # Use all CPU cores for cross-validation
# )
# print("Random Forest Regressor RMSE Scores:")
# print(pd.Series(forest_rmse_scores).describe())

full_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "random_forest",
            RandomForestRegressor(random_state=42, n_jobs=-1),
        ),  # Use all CPU cores
    ]
)

# # Hyperparameter tuning with GridSearchCV
# param_grid = [
#    # First parameter grid. 3x3 = 9 combinations
#    {
#        "preprocessor__geo__n_clusters": [5, 8, 10],
#        "random_forest__max_features": [4, 6, 8],
#    },
#    # Second parameter grid. 2x3 = 6 combinations
#    {
#        "preprocessor__geo__n_clusters": [10, 15],
#        "random_forest__max_features": [6, 8, 10],
#    },
# ]
#
# grid_search = GridSearchCV(
#    full_pipeline,
#    param_grid,
#    cv=3,  # Use 3-fold cross-validation
#    scoring="neg_root_mean_squared_error",
#    n_jobs=-1,
# )
# grid_search.fit(housing, housing_labels)
#
# print("Best parameters found:")
# print(grid_search.best_params_)

# Hyperparameter tuning with RandomizedSearchCV
param_distributions = {
    "preprocessor__geo__n_clusters": randint(3, 50),
    "random_forest__max_features": randint(2, 20),
}

rnd_search = RandomizedSearchCV(
    full_pipeline,
    param_distributions,
    n_iter=10,  # Number of random combinations to try
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
)
rnd_search.fit(housing, housing_labels)

print("Best parameters found:")
print(rnd_search.best_params_)

final_model = rnd_search.best_estimator_
feature_importances = final_model[
    "random_forest"
].feature_importances_  # Feature importances

# Create DataFrame for nicer display
importance_df = pd.DataFrame(
    {
        "Feature": final_model["preprocessor"].get_feature_names_out(),
        "Importance": feature_importances,
    }
)

# Sort by importance, round values
importance_df = importance_df.sort_values("Importance", ascending=False)
importance_df["Importance"] = importance_df["Importance"].round(4)

# Print the formatted result
print("\nFeature Importances:")
print(importance_df)


X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)
print("Final RMSE on test set:", final_rmse)

# Calculate 95% confidence interval for RMSE
# Assuming the errors are normally distributed
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print(
    np.sqrt(
        stats.t.interval(
            confidence,
            len(squared_errors) - 1,  # degrees of freedom
            loc=np.mean(squared_errors),  # sample mean
            scale=stats.sem(squared_errors),  # standard error of the mean
        )
    )
)
