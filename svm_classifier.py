from sklearn import datasets, svm

iris = datasets.load_iris()
x_train, x_test = iris.data[:-10], iris.data[-10:]
y_train, y_test = iris.target[:-10], iris.target[-10:]

svc = svm.SVC(kernel='linear')
svc.fit(x_train, y_train)
print(svc.predict(x_test))
print(y_test)