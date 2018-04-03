from sklearn import linear_model, datasets

lasso_classifier = linear_model.LassoCV()
diabetes = datasets.load_diabetes()

lasso_classifier.fit(diabetes.data[:-100], diabetes.target[:-100])
print(lasso_classifier.score(diabetes.data[-100:], diabetes.target[-100:]))
