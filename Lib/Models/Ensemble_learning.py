import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("path/to/dataset.csv")
X = data.drop(columns=["target"])
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained models
model_dir = "path/to/pretrained_models/"
models = []
for model_file in os.listdir(model_dir):
    if model_file.endswith(".pkl"):
        with open(os.path.join(model_dir, model_file), "rb") as f:
            models.append(pickle.load(f))

# Define weights for models
weights = [0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1] # Change these weights as desired

# Ensemble models and make predictions
y_probs = []
for model in models:
    y_prob = model.predict_proba(X_test)
    y_probs.append(y_prob)

# Weighted probability averaging of predictions
y_probs_ensemble = np.average(y_probs, axis=0, weights=weights)

# Predict the class with the highest probability
y_pred_ensemble = np.argmax(y_probs_ensemble, axis=1)

# Evaluate ensemble model and save results
acc = accuracy_score(y_test, y_pred_ensemble)
cm = confusion_matrix(y_test, y_pred_ensemble)
results_file = "path/to/ensemble_results.txt"
with open(results_file, 'w') as f:
    f.write("Accuracy: {}\n".format(acc))
    f.write("Confusion matrix:\n{}\n".format(cm))

# Save ensemble model
ensemble_model_file = "path/to/ensemble_model.pkl"
with open(ensemble_model_file, "wb") as f:
    pickle.dump((models, weights), f)

print("Ensemble model and results saved in {} and {}".format(ensemble_model_file, results_file))
