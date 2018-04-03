import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, cross_val_score

digits = datasets.load_iris()
svm_classifier = svm.SVC(kernel='linear')
Cs = np.logspace(-6, -1, 10)
classifier = GridSearchCV(estimator=svm_classifier, param_grid=dict(C=Cs))
classifier.fit(digits.data[:1000], digits.target[:1000])

print(classifier.best_score_)
print(classifier.best_estimator_.C)