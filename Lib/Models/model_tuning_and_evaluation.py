# Hyperparameter tuning using Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)

param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [10, 20, 30, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# Model evaluation using precision, recall, and F1 score
from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = grid_search.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

# Model evaluation using confusion matrix
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Model evaluation using classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# Model evaluation using cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
scores = cross_val_score(lr, X_train, y_train, cv=5)
print('Cross-validation scores:', scores)
print('Average score:', scores.mean())

# Model evaluation using ROC curve and AUC score
from sklearn.metrics import roc_curve, roc_auc_score

lr.fit(X_train, y_train)
y_pred_proba = lr.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

auc_score = roc_auc_score(y_test, y_pred_proba)
print('AUC score:', auc_score)











# Model comparison using cross-validation
from sklearn.model_selection import cross_val_score

# Logistic regression
lr_scores = cross_val_score(lr, X_train, y_train, cv=5)
print(f"Logistic regression CV scores: {lr_scores}")
print(f"Logistic regression mean CV score: {np.mean(lr_scores)}")

# Random forest
rf_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Random forest CV scores: {rf_scores}")
print(f"Random forest mean CV score: {np.mean(rf_scores)}")

# Model selection using grid search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [1, 3, 5]
}

grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding mean CV score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Corresponding mean CV score: {grid_search.best_score_}")

# Model selection using random search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=5)
random_search.fit(X_train, y_train)

# Print the best parameters and the corresponding mean CV score
print(f"Best parameters: {random_search.best_params_}")
print(f"Corresponding mean CV score: {random_search.best_score_}")

# Model evaluation using ROC curve and AUC score
from sklearn.metrics import roc_curve, auc

# Logistic regression
y_score_lr = lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f"Logistic regression (AUC = {roc_auc_lr:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

# Random forest
y_score_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f"Random forest (AUC = {roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

# Model evaluation using confusion matrix and classification report
from sklearn.metrics import confusion_matrix