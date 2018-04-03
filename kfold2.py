import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score

digits = datasets.load_digits()
svm_classifier = svm.SVC(kernel='linear')
k_fold = KFold(n_splits=4, shuffle=True)

for training_indices, test_indices in k_fold.split(digits.data):
    print(svm_classifier
          .fit(digits.data[training_indices], digits.target[training_indices])
          .score(digits.data[test_indices], digits.target[test_indices]))

print(cross_val_score(svm_classifier, digits.data,
                      digits.target, cv=k_fold, n_jobs=-1))
