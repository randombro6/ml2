import pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


data = pd.read_csv('random.csv')
data.head()

x = data.iloc[: , : -1]
y = data.iloc[: , -1]




x_train , x_test ,y_train,y_test = train_test_split(x,y)

dt = DecisionTreeClassifier()
rf  = RandomForestClassifier(n_estimators = 30)

dt.fit(x_train,y_train)
rf.fit(x_train,y_train)

# Make predictions on the test set
dt_predictions = dt.predict(x_test)
rf_predictions = rf.predict(x_test)

# Evaluate accuracy
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Print the results
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Visualize one of the decision trees in the Random Forest
plt.figure(figsize=(20, 15))
tree.plot_tree(rf.estimators_[0], feature_names=x.columns, class_names=np.unique(y).astype('str'), filled=True)
plt.show()

