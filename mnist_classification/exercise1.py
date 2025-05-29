from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# Load mnist dataset
# minis contains 70,000 images of handwritten digits (0-9)
mnist = fetch_openml(
    "mnist_784",
    as_frame=False,  # Fetch the dataset without converting it to a DataFrame
)

X, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

kn_clf = KNeighborsClassifier(n_neighbors=3)
grid_search = GridSearchCV(
    kn_clf,
    param_grid={"n_neighbors": [1, 3, 5, 7, 9]},
    cv=3,  # 3-fold cross-validation. This means the training set is split into 3 parts, and the model is trained on 2 parts and validated on the remaining part.
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(x_train, y_train)
# Output the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.2f}")
