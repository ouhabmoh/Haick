import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle
import datetime
# Load data
data = pd.read_csv("path/to/dataset.csv")
X = data.drop(columns=["target"])
y = data["target"]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define models
models = [
        ("LR", LinearRegression()),
        ("Ridge", Ridge(random_state=42)),
        ("Lasso", Lasso(random_state=42)),
        ("ElasticNet", ElasticNet(random_state=42)),
        ("SVR", SVR()),
        ("RF", RandomForestRegressor(random_state=42)),
        ("KNN", KNeighborsRegressor()),
        ("ADA", AdaBoostRegressor(random_state=42)),
        ("XGB", XGBRegressor(random_state=42))
    ]
# Define hyperparameters to optimize
params = {
        "LR": {},
        "Ridge": {"model__alpha": np.logspace(-4, 4, 9)},
        "Lasso": {"model__alpha": np.logspace(-4, 4, 9)},
        "ElasticNet": {"model__alpha": np.logspace(-4, 4, 9), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "SVR": {"model__C": np.logspace(-4, 4, 9), "model__kernel": ["linear", "poly", "rbf", "sigmoid"]},
        "RF": {"model__n_estimators": [10, 50, 100, 200, 500]},
        "KNN": {"model__n_neighbors": [3, 5, 7, 9, 11]},
        "ADA": {"model__n_estimators": [10, 50, 100, 200, 500]},
        "XGB": {"model__n_estimators": [10, 50, 100, 200, 500]}
    }
# Create folder to save models and results
model_dir = os.path.join("ML_Regressors", datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# Train models and save to file
for name, model in models:
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    grid_search = GridSearchCV(clf, params[name], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model_path = os.path.join(model_dir, name + ".pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(grid_search,f)
    # Evaluate model on test set
    y_pred = grid_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Save results to file
    result_path = os.path.join(model_dir, name + "_results.txt")
    with open(result_path, 'w') as f:
        f.write(f"MSE: {mse}\n")
        f.write(f"R2 score: {r2}\n")
