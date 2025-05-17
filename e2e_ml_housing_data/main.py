import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

from load_data import load_housing_data
from preprocessor import create_preprocessor

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

# Linear Regression
preprocessor = create_preprocessor()
lin_reg = make_pipeline(preprocessor, LinearRegression())
lin_reg.fit(housing, housing_labels)

# Evaluate the model
housing_predictions = lin_reg.predict(housing)
print("Predictions:", housing_predictions[:5].round(2))
print("Labels:", housing_labels.iloc[:5].values)
lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print("Root Mean Squared Error:", lin_rmse)

# Decision Tree Regressor
tree_reg = make_pipeline(preprocessor, DecisionTreeRegressor())
tree_reg.fit(housing, housing_labels)

# Evaluate the model
housing_predictions = tree_reg.predict(housing)
print("Predictions:", housing_predictions[:5].round(2))
print("Labels:", housing_labels.iloc[:5].values)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print("Root Mean Squared Error:", tree_rmse)
