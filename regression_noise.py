import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

X = np.c_[.5, 1].T
y = [.5, 1]
test = np.c_[0, 2].T
regression = linear_model.LinearRegression()

np.random.seed(0)
for _ in range(6):
    this_X = .1 * np.random.normal(size=(2, 1)) + X
    regression.fit(this_X, y)
    plt.plot(test, regression.predict(test))
    plt.scatter(this_X, y, s=3)

plt.show()
