import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

start = timer()

data = pd.read_csv('/Users/ganeshs/Documents/SML/train.csv')
data['Lead'].replace({'Male':0, 'Female':1}, inplace=True)
y = data['Lead']
X = data.drop(['Total words', 'Lead', 'Gross', 'Year'], axis = 1)
logscaled = ['Number words female', 'Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors',
       'Number of female actors', 'Number words male']
for i in logscaled:
    X[i] = np.log(X[i] + 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

logModel = LogisticRegression()
scaler = RobustScaler()
pipe = Pipeline(steps=[("scaler", scaler), ("logistic", logModel)])
param_grid = {
    'logistic__max_iter': [1000, 5000, 10000],
    'logistic__solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
    "logistic__C": np.logspace(-4, 4, 4),
    'logistic__penalty': ['l1', 'l2', 'elasticnet', 'none']
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=cross_validation, refit="accuracy",verbose=True)

best_clf = search.fit(X_train, y_train)
print(best_clf.best_estimator_)
print (f'Training Accuracy - : {best_clf.score(X_train,y_train):.3f}')

predictions = best_clf.best_estimator_.predict(X_test)

print('accuracy is :' + str(sklearn.metrics.accuracy_score(y_test,predictions)))
print('f1 score is :'+str(f1_score(y_test, predictions)))
print('Precision score is :' +str(precision_score(y_test, predictions)))
print('Recall score is :' +str(recall_score(y_test, predictions)))

end = timer()

print(end - start)
plot_confusion_matrix(best_clf.best_estimator_, X_test, y_test)
plt.show()
