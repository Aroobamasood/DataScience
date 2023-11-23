# 23/11/23
# CSC461 – Assignment3 – Machine Learning
# AROOBA MASOOD
# FA20-BSE-092
# This task involves applying a Random Forest classification algorithm using Python's scikit-learn library on a gender prediction dataset. The goal is to predict gender based on various features while utilizing both Monte Carlo and Leave P-Out cross-validation strategies to assess the model's performance, reporting F1 scores for evaluation.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeavePOut, ShuffleSplit
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('gender-prediction.csv')

le = LabelEncoder()
data['beard'] = le.fit_transform(data['beard'])
data['hair_length'] = le.fit_transform(data['hair_length'])
data['scarf'] = le.fit_transform(data['scarf'])
data['eye_color'] = le.fit_transform(data['eye_color'])
data['gender'] = le.fit_transform(data['gender'])

X = data.drop(columns=['gender'])
y = data['gender']

rf = RandomForestClassifier(n_estimators=100, random_state=42)

mc_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
mc_scores = cross_val_score(rf, X, y, cv=mc_cv, scoring='f1_macro')

p_out = 5
lpo_cv = LeavePOut(p=p_out)
lpo_scores = cross_val_score(rf, X, y, cv=lpo_cv, scoring='f1_macro')

print('Monte Carlo CV Scores: ', mc_scores)
print('Leave P-Out CV Scores: ', lpo_scores)
