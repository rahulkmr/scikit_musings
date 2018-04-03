from sklearn import datasets, svm
import numpy as np

digits = datasets.load_digits()
X_train, X_test = digits.data[:-100], digits.data[-100:]
y_train, y_test = digits.target[:-100], digits.target[-100:]


svc = svm.SVC(kernel='linear')
print(svc.fit(X_train, y_train).score(X_test, y_test))

NUM_FOLDS = 4
X_folds = np.array_split(digits.data, NUM_FOLDS)
y_folds = np.array_split(digits.target, NUM_FOLDS)
scores = list()

for index in range(NUM_FOLDS):
    X_train = list(X_folds)
    X_test = X_train.pop(index)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(index)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))

print(scores)