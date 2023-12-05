import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data = pd.read_csv('glass(For SVM Program).csv')
data.head()

data.drop('Id' , axis=1)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
svm_linear_predictions = svm_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, svm_linear_predictions)

# SVM with Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3)  # You can adjust the degree as needed
svm_poly.fit(X_train, y_train)
svm_poly_predictions = svm_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, svm_poly_predictions)

# SVM with Radial Basis Function (RBF) Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
svm_rbf_predictions = svm_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, svm_rbf_predictions)

# SVM with sigmoid Kernel
svm_sig = SVC(kernel='sigmoid')
svm_sig.fit(X_train, y_train)
svm_sig_predictions = svm_sig.predict(X_test)
accuracy_sig = accuracy_score(y_test, svm_sig_predictions)

print("SVM with Linear Kernel Accuracy:", accuracy_linear)
print("SVM with Polynomial Kernel Accuracy:", accuracy_poly)
print("SVM with RBF Kernel Accuracy:", accuracy_rbf)
print("SVM with sigmoid Kernel Accuracy:", accuracy_sig)

