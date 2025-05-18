from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from plot_digit import plot_digit

mnist = fetch_openml("mnist_784", as_frame=False)

X, y = mnist["data"], mnist["target"]

# some_digit = X[0]
# plot_digit(some_digit)
# plt.show()

x_train, x_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

for i in range(100):
    plt.subplot(10, 10, i + 1)
    plot_digit(x_train[i])

plt.show()
