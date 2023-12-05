import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv('zoo.csv')

X = data.iloc[:, :-1]  
Y = data.iloc[:, -1]  


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Confusion Matrix:", confusion_matrix(Y_test, Y_pred))

print("Classification Report:")
print(classification_report(Y_test, Y_pred, zero_division=0))

plt.figure(figsize=(25,20))
_=tree.plot_tree(model)