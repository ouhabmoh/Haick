import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")

# Load test data
test_data = pd.read_csv("test.csv")

# Separate features
X = data.iloc[:, :-1].values

# Define models
models = [
    ("KMeans", KMeans(n_clusters=2)),
    ("Agglomerative", AgglomerativeClustering(n_clusters=2)),
    ("DBSCAN", DBSCAN(eps=0.5, min_samples=5)),
    ("MeanShift", MeanShift()),
    ("Spectral", SpectralClustering(n_clusters=2)),
    ("GMM", GaussianMixture(n_components=2)),
    ("Affinity", AffinityPropagation())
]

# Evaluate models
scores = []
for name, model in models:
    y_pred = model.fit_predict(X)
    score = silhouette_score(X, y_pred)
    scores.append(score)
    print("{}: {}".format(name, score))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title(name)
    plt.show()

# Save predictions to file
best_model_idx = np.argmax(scores)
best_model_name = models[best_model_idx][0]
best_model = models[best_model_idx][1]
y_pred_test = best_model.fit_predict(test_data.values)
test_data['cluster'] = y_pred_test
test_data.to_csv('predictions.csv', index=False)

print("Predictions saved in predictions.csv")
