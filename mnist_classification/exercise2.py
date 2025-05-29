from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import shift
import numpy as np


def shift_image(image, dx, dy):
    """Shift a flattened 28x28 image by dx pixels horizontally and dy pixels vertically."""
    image_2d = image.reshape(28, 28)
    shifted_2d = shift(image_2d, [dy, dx], cval=0)
    return shifted_2d.flatten()


# Load mnist dataset
# minis contains 70,000 images of handwritten digits (0-9)
mnist = fetch_openml(
    "mnist_784",
    as_frame=False,  # Fetch the dataset without converting it to a DataFrame
)

X, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

aug_x_train = []
aug_y_train = []


for i, (image, label) in enumerate(zip(x_train[:1000], y_train[:1000])):
    # Add original image
    aug_x_train.append(image)
    # Add shifted versions using custom shift_image function
    aug_x_train.append(shift_image(image, 1, 0))  # Shift right
    aug_x_train.append(shift_image(image, -1, 0))  # Shift left
    aug_x_train.append(shift_image(image, 0, 1))  # Shift down
    aug_x_train.append(shift_image(image, 0, -1))  # Shift up

    aug_y_train.extend([label] * 5)

x_train = np.array(aug_x_train)
y_train = np.array(aug_y_train)

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
