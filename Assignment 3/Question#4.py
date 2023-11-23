# 23/11/23
# CSC461 – Assignment3 – Machine Learning
# AROOBA MASOOD
# FA20-BSE-092
# This Python code employs Gaussian Naive Bayes classification after encoding categorical data into numerical form using LabelEncoder. It assesses gender predictions based on a dataset, splitting it into training and test sets, then evaluating accuracy, precision, and recall metrics for the classification model.

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('gender-prediction-updated.csv')

le = LabelEncoder()
columns_to_encode = ['beard', 'scarf', 'hair_length', 'eye_color']
for col in columns_to_encode:
    data[col] = le.fit_transform(data[col])

X_train = data.drop(columns=['gender'])[:-10]
y_train = data['gender'][:-10]

X_test = data.drop(columns=['gender'])[-10:]
y_test = data['gender'][-10:]

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
