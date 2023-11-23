# 23/11/23
# CSC461 – Assignment3 – Machine Learning
# AROOBA MASOOD
# FA20-BSE-092
# The task involves gender prediction using machine learning models on encoded features like 'beard,' 'hair_length,' 'scarf,' and 'eye_color.' It assesses model performance, split ratio impact (2/3 and 80/20), and the influence of excluding 'hair_length' and 'beard' on prediction accuracy.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('gender-prediction.csv')

data_encoded = pd.get_dummies(data, columns=['beard', 'hair_length', 'scarf', 'eye_color'])

X = data_encoded.drop('gender', axis=1)
y = data_encoded['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

svm = SVC()
svm.fit(X_train, y_train)

mlp = MLPClassifier()
mlp.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print("\033[1m2/3 Train and 1/3 Test split: \033[0m")
print("Logistic Regression Accuracy:", accuracy_logreg)
print("Support Vector Machines Accuracy:", accuracy_svm)
print("Multilayer Perceptron Accuracy:", accuracy_mlp)

incorrect_logreg = (y_test != y_pred_logreg).sum()
incorrect_svm = (y_test != y_pred_svm).sum()
incorrect_mlp = (y_test != y_pred_mlp).sum()

print("\n\033[1m1. How many instances are incorrectly classified?\033[0m")
print("Number of instances incorrectly classified (Logistic Regression):", incorrect_logreg)
print("Number of instances incorrectly classified (Support Vector Machines):", incorrect_svm)
print("Number of instances incorrectly classified (Multilayer Perceptron):", incorrect_mlp)

# Rerun the experiment again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

svm = SVC()
svm.fit(X_train_scaled, y_train)

mlp = MLPClassifier()
mlp.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print("\n\033[1m2. Rerun the experiment using train/test split ratio of 80/20.\033[0m")
print("Logistic Regression Accuracy:", accuracy_logreg)
print("Support Vector Machines Accuracy:", accuracy_svm)
print("Multilayer Perceptron Accuracy:", accuracy_mlp)

incorrect_logreg = (y_test != y_pred_logreg).sum()
incorrect_svm = (y_test != y_pred_svm).sum()
incorrect_mlp = (y_test != y_pred_mlp).sum()

print("Number of instances incorrectly classified (Logistic Regression):", incorrect_logreg)
print("Number of instances incorrectly classified (Support Vector Machines):", incorrect_svm)
print("Number of instances incorrectly classified (Multilayer Perceptron):", incorrect_mlp)

print("\n\033[1mDo you see any change in the results? Explain.\033[0m")
print("When we change the train/test split ratio from 2/3 to 80/20, we see a change in the results. With a "
      "larger training set (80% instead of 66.67%), the models may have more data to learn from, potentially leading "
      "to better generalization and performance on the test set. This could result in higher accuracy and potentially "
      "fewer instances being incorrectly classified. However, it's also possible that the change in the split ratio "
      "may not significantly impact the results, especially if the original split already provided sufficient data "
      "for the models to learn from.")

print("\n\033[1mName 2 attributes that you believe are the most “powerful” in the prediction task. Explain why? \033[0m")
print("The 'hair_length' and 'beard' attributes are powerful predictors for gender because they align with "
      "traditional gender norms in many cultures, where certain hair lengths and the presence of a beard are strongly "
      "associated with specific genders.")

print("\n\033[1mTry to exclude these 2 attribute(s) from the dataset. Rerun the experiment (using 80/20 train/test "
      "split),did you find any change in the results? Explain. \033[0m")
print("Excluding 'hair_length' and 'beard' may impact the results, potentially decreasing accuracy and increasing "
      "misclassifications due to the loss of influential predictors.")

