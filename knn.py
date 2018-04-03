import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

np.random.seed(0)
indices = np.random.permutation(len(iris.data))
x_train, x_test = iris.data[indices[:-10]], iris.data[indices[-10:]]
y_train, y_test = iris.target[indices[:-10]], iris.target[indices[-10:]]

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print(knn.predict(x_test))
print(y_test)