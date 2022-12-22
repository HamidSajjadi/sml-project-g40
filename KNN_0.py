# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:16:59 2022

@author: Baumann
"""

#%% import
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% save best
import joblib
import json
def save_best_estimator(clf, name):
  best_model_stats = {}
  for key, val in clf.cv_results_.items():
    real_val = val[clf.best_index_]
    if 'numpy' in str(type(real_val)):
      if 'float' in str(type(real_val)):
        best_model_stats[key] = float(real_val) 
      else:
        best_model_stats[key] = int(real_val) 
    else:
      best_model_stats[key] = real_val
  joblib.dump(clf.best_estimator_, f'{name}.pkl', compress = 1)
  with open(f"{name}_stats.json", 'w') as jf:
    json.dump(best_model_stats, jf, indent=4)

#%% data
df_tr = pd.read_csv('data/train.csv')
df_tst = pd.read_csv('data/test.csv')

#% preprocessing
y_tr = df_tr.pop('Lead')
df_tr.pop('Total words')
df_tr.pop('Year')
df_tr.pop('Gross')


names = df_tr.keys()
print('Used features: ', names)

y_tr = y_tr=='Female'   # 'Female' is the positive label

#% scaling
logscaled = ['Number words female', 'Number of words lead',
       'Difference in words lead and co-lead', 'Number of male actors', 'Number of female actors', 
       'Number words male']
for i in logscaled:
    df_tr[i] = np.log(df_tr[i] + 1)

#%% GridSearchCV
random_state = 42
parameters = {'model__n_neighbors':range(1,50), 'model__weights':['distance', 'uniform'], 'model__metric':['manhattan', 'euclidean', 'cosine']}
model = Pipeline([('scaler', RobustScaler()), ('model', KNeighborsClassifier())])

cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

clf = GridSearchCV(model, parameters, n_jobs=4, cv=cross_validation, refit="accuracy", scoring=["f1", "accuracy", "precision","recall"])
clf.fit(X=df_tr, y=y_tr)
model = clf.best_estimator_
save_best_estimator(clf, "KNN")
print (clf.best_score_, clf.best_params_) 

#%% plot result
result = clf.cv_results_
fig, axes = plt.subplots(2,3, figsize=(12,5))

for i, weight in enumerate(['distance', 'uniform']):
    for j, metric in enumerate(['manhattan', 'euclidean', 'cosine']):
        for k in result:
            
            if k[:10] == 'mean_test_':
                print(k)
                temp = []
                for l,m in zip(result[k], result['params']):
                    if m['model__weights'] == weight and m['model__metric'] == metric:
                        temp.append(l)
                axes[i,j].plot(temp, label=k[10:]) 
                axes[i,j].set_title('weight: ' + weight + ', metric: ' + metric)
                axes[i,j].set_xlabel('n_neighbors')
                axes[i,j].legend()
    
fig.tight_layout()

index = clf.best_index_
print('acc: ', round(result['mean_test_accuracy'][index], 4))
print('f1: ', round(result['mean_test_f1'][index], 4))
print('precision: ', round(result['mean_test_precision'][index], 4))
print('recall: ', round(result['mean_test_recall'][index], 4))

#%% Dummy
clf_dumm = DummyClassifier(strategy='most_frequent')
clf_dumm.fit(df_tr, y_tr)
y_dumm = clf_dumm.predict(df_tr)
dumm_scores = {}
dumm_scores['acc'] = metrics.accuracy_score(y_tr, y_dumm)
dumm_scores['f1'] = metrics.f1_score(y_tr, y_dumm)
dumm_scores['precision'] = metrics.precision_score(y_tr, y_dumm)
dumm_scores['recall'] = metrics.recall_score(y_tr, y_dumm)

