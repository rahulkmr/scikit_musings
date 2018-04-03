from sklearn import datasets, linear_model

iris = datasets.load_iris()
x_train, x_test = iris.data[:-10], iris.data[-10:]
y_train, y_test = iris.target[:-10], iris.target[-10:]

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(x_train, y_train)
print(logistic.predict(x_test))
print(y_test)