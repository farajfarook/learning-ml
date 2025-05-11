import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

model2 = KNeighborsRegressor(n_neighbors=3)

data_root = "https://github.com/ageron/data/raw/main"
lifesat = pd.read_csv(f"{data_root}/lifesat/lifesat.csv")

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat["Life satisfaction"].values

lifesat.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction", grid=True)
plt.axis([23_500, 62_500, 4, 9])
plt.show()


def train_and_predict(model, X, y):
    model.fit(X, y)
    X_new = np.array([[37_655.2]])
    print(model.predict(X_new))


model = LinearRegression()
train_and_predict(model, X, y)

model2 = KNeighborsRegressor(n_neighbors=3)
train_and_predict(model2, X, y)
