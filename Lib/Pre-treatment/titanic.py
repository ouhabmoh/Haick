# Drop rows with missing values
titanic = titanic.dropna()

# Fill missing values with the mean of the column
titanic['age'] = titanic['age'].fillna(titanic['age'].mean())

# Impute missing values using KNN imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
titanic[['age', 'fare']] = imputer.fit_transform(titanic[['age', 'fare']])


# One-hot encode categorical variables
titanic = pd.get_dummies(titanic, columns=['sex', 'embarked'])

# Label encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
titanic['class'] = le.fit_transform(titanic['class'])

# Hash encoding
from category_encoders import HashingEncoder
hasher = HashingEncoder(cols=['class'], n_components=4)
titanic = hasher.fit_transform(titanic)

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
titanic[['age', 'fare']] = scaler.fit_transform(titanic[['age', 'fare']])

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
titanic[['age', 'fare']] = scaler.fit_transform(titanic[['age', 'fare']])

# RobustScaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
titanic[['age', 'fare']] = scaler.fit_transform(titanic[['age', 'fare']])


# Remove outliers based on z-score
from scipy import stats
z_scores = stats.zscore(titanic[['age', 'fare']])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
titanic = titanic[filtered_entries]

# Transform the data using log transformation
titanic[['age', 'fare']] = np.log(titanic[['age', 'fare']])


# Creating new feature "family_size" from "sibsp" and "parch"
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# Creating interaction term "age * class"
titanic['age_class'] = titanic['age'] * titanic['class']


# Univariate feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = titanic.drop('survived', axis=1)
y = titanic['survived']
k_best = SelectKBest(score_func=chi2, k=3)
X_new = k_best.fit_transform(X, y)

# Upsampling the minority class using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Downsample the majority class using RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)


# PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE for dimensionality reduction
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)


# Check for multicollinearity using VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X = add_constant(X)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
vif = vif.drop('const')

# Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X = X.drop(to_drop, axis=1)