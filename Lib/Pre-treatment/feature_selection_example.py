import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset
iris = sns.load_dataset('iris')

# Inspect the first few rows of the dataset
iris.head()

# Perform exploratory data analysis
sns.pairplot(iris, hue='species')

# Separate the features and target variable
X = iris.drop('species', axis=1)
y = iris['species']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform feature selection using recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=10000)
rfe = RFE(logreg, n_features_to_select=2)
rfe = rfe.fit(X_train, y_train)

# Inspect the selected features
selected_features = X_train.columns[rfe.support_]
print(selected_features)

# Visualize the relationship between the selected features and the target variable
sns.scatterplot(x=selected_features[0], y=selected_features[1], data=X_train.join(y_train))

# Train and evaluate a logistic regression model on the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

logreg.fit(X_train_selected, y_train)

y_pred = logreg.predict(X_test_selected)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
