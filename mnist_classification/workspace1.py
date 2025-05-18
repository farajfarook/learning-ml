from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

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
