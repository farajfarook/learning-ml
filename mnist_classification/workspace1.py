from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

from plot_digit import plot_digit

mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist["data"], mnist["target"]

# some_digit = X[0]
# plot_digit(some_digit)
# plt.show()

x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# for i in range(100):
#    plt.subplot(10, 10, i + 1)
#    plot_digit(x_train[i])
#
# plt.show()

y_train_5 = y_train == "5"
y_test_5 = y_test == "5"

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)


# for i in range(9):
#    plt.subplot(3, 3, i + 1)
#    value = sgd_clf.predict([x_train[i]])
#    plt.title(f"{value[0]}")
#    plot_digit(x_train[i])
#
# plt.show()
print("SGDClassifier score: ", sgd_clf.score(x_test, y_test_5))
print(
    "Cross-validation scores: ",
    cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy"),
)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train, y_train_5)
print("DummyClassifier score: ", dummy_clf.score(x_test, y_test_5))
