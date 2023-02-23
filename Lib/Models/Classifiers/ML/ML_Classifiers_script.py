import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import pickle
import datetime
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config.yaml")
def train_model(cfg: DictConfig):
    
    # Load data
    data = pd.read_csv(cfg.data.path)
    X = data.drop(columns=[cfg.data.target_column])
    y = data[cfg.data.target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.data.test_size, random_state=cfg.seed)

    # Define models
    models = [
        ("LR", LogisticRegression(random_state=cfg.seed)),
        ("SVM", SVC(random_state=cfg.seed)),
        ("RF", RandomForestClassifier(random_state=cfg.seed)),
        ("KNN", KNeighborsClassifier()),
        ("NB", GaussianNB()),
        ("ADA", AdaBoostClassifier(random_state=cfg.seed)),
        ("XGB", XGBClassifier(random_state=cfg.seed))
    ]

    # Define hyperparameters to optimize
    params = {
        "LR": {"model__C": np.logspace(-4, 4, 9)},
        "SVM": {"model__C": np.logspace(-4, 4, 9), "model__kernel": ["linear", "poly", "rbf", "sigmoid"]},
        "RF": {"model__n_estimators": [10, 50, 100, 200, 500]},
        "KNN": {"model__n_neighbors": [3, 5, 7, 9, 11]},
        "NB": {},
        "ADA": {"model__n_estimators": [10, 50, 100, 200, 500]},
        "XGB": {"model__n_estimators": [10, 50, 100, 200, 500]}
    }

    # Create folder to save models and results
    model_dir = os.path.join(cfg.output.model_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
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
            pickle.dump(grid_search.best_estimator_, f)

    # Evaluate models and save results to file
    results_file = os.path.join(model_dir, "results.txt")
    with open(results_file, 'w') as f:
        for name, model in models:
            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
                ])
    with open(os.path.join(model_dir, name + ".pkl"), 'rb') as f2:
        loaded_model = pickle.load(f2)
        y_pred = loaded_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        f.write(f"{name}:\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Confusion matrix: {cm}\n\n")
