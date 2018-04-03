import numpy as np
from sklearn import datasets
from sklearn import linear_model

diabetes = datasets.load_diabetes()
x_train, x_test = diabetes.data[:- 20], diabetes.data[-20:]
y_train, y_test = diabetes.target[:-20], diabetes.target[-20:]

regression = linear_model.LinearRegression()
regression.fit(x_train, y_train)
print(regression.coef_)
print(np.mean((regression.predict(x_test) - y_test) ** 2))
print(regression.score(x_test, y_test))

alphas = np.logspace(-4, -1, 6)
# print([regression.set_params(alpha=alpha).fit(x_train, y_train).score(x_test, y_test)
#       for alpha in alphas])