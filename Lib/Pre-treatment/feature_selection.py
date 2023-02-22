# Feature selection using mutual information
from sklearn.feature_selection import mutual_info_classif

scores = mutual_info_classif(X_train, y_train)
feature_scores = pd.Series(scores, index=X.columns)
feature_scores.plot(kind='barh')
plt.show()

# Feature selection using Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

rfe = RFE(estimator=tree, n_features_to_select=3)
rfe.fit(X_train, y_train)

selected_features = X.columns[rfe.support_]
print('Selected features:', selected_features)

#
# These techniques include feature selection using SelectKBest and chi-squared test to identify
# the most important features, model interpretation using coefficients and feature importance to
# identify the features that have the biggest impact on the model predictions, and model interpretation
# using partial dependence plots and SHAP values to visualize the relationship between the features and
# the model predictions. These techniques can help with feature selection and model interpretation, which
# are important steps in the machine learning pipeline.

# Feature selection using SelectKBest and chi-squared test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

selector = SelectKBest(chi2, k=5)
selector.fit_transform(X_train, y_train)

# Get the scores and p-values of the features
scores = selector.scores_
p_values = selector.pvalues_

# Get the indices of the selected features
selected_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = X_train.columns[selected_indices]

# Print the scores and p-values of the features
for feature, score, p_value in zip(X_train.columns, scores, p_values):
    print(feature, score, p_value)

# Print the names of the selected features
print(selected_features)

# Model interpretation using coefficients and feature importance
coefs = pd.DataFrame({'feature': X_train.columns, 'coefficient': lr.coef_[0]})
coefs = coefs.sort_values('coefficient', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(y=coefs['feature'], width=coefs['coefficient'])
plt.title('Coefficients of logistic regression model')
plt.show()

feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(y=feature_importance['feature'], width=feature_importance['importance'])
plt.title('Feature importance of random forest model')
plt.show()

# Model interpretation using partial dependence plots
from sklearn.inspection import plot_partial_dependence

fig, ax = plt.subplots(figsize=(10, 6))
plot_partial_dependence(rf, X_train, ['Age', 'Fare'], ax=ax)
plt.show()

# Model interpretation using SHAP values
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)


# Correlation-based feature selection
correlation_matrix = df.corr()
relevant_features = correlation_matrix.index[abs(correlation_matrix["target_variable"]) > 0.5]
df = df[relevant_features]