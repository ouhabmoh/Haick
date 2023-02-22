import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
from scipy.stats import ttest_ind, chi2_contingency

data = pd.read_csv('dataset.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

# Fill missing values with mode
df.fillna(df.mode(), inplace=True)

# Identify and handle missing values
# Replace missing values in numeric columns with the median
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    data[col].fillna(data[col].median(), inplace=True)

# Replace missing values in categorical columns with the mode
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Imputation using Mean/Median/Mode
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Imputation using KNN
from fancyimpute import KNN

X_imputed = KNN(k=3).fit_transform(X)

# Check for duplicates
print('Number of duplicates in the dataset:', data.duplicated().sum())

# Remove duplicates
data.drop_duplicates(inplace=True)

# Remove outliers using z-score method
from scipy import stats
z_scores = stats.zscore(df["numerical_column"])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
df = df[filtered_entries]

# Boxplot visualization
import seaborn as sns

sns.boxplot(x=df['Income'])

# Winsorization
from scipy.stats.mstats import winsorize

df['Income'] = winsorize(df['Income'], limits=[0.05, 0.05])

# IQR Method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Apply logarithmic transformation to a column
df["column_to_transform"] = np.log(df["column_to_transform"])

# Apply one-hot encoding to a categorical column
df = pd.get_dummies(df, columns=["categorical_column"])

# Apply label encoding to a categorical column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["categorical_column"] = le.fit_transform(df["categorical_column"])

# Convert categorical columns to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['categorical_column1', 'categorical_column2'])


# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender_Code'] = le.fit_transform(df['Gender'])

# One-Hot Encoding
dummies = pd.get_dummies(df['City'])
df = pd.concat([df, dummies], axis=1)

# Perform dimensionality reduction using principal component analysis (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = data_encoded.drop('target_column', axis=1)
y = data_encoded['target_column']
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# t-SNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df)


# Feature scaling using standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = data_encoded.drop('target_column', axis=1)
y = data_encoded['target_column']
X_scaled = scaler.fit_transform(X)

# Feature scaling using normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Analyze class imbalance
sns.countplot(data['target_column'])
plt.xlabel('Target')
plt.show()

# Address class imbalance using undersampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()
X = data_encoded.drop('target_column', axis=1)
y = data_encoded['target_column']
X_rus, y_rus = rus.fit_resample(X, y)

sns.countplot(y_rus)
plt.xlabel('Target')
plt.show()

# Address class imbalance using oversampling
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X, y)

sns.countplot(y_ros)
plt.xlabel('Target')
plt.show()

# Address class imbalance using SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

sns.countplot(y_smote)
plt.xlabel('Target')
plt.show()


# Oversampling using SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Shape of X_resampled after SMOTE:", X_resampled.shape)
print("Shape of y_resampled after SMOTE:", y_resampled.shape)

# Undersampling using Tomek links
from imblearn.under_sampling import TomekLinks

tomek = TomekLinks()
X_resampled, y_resampled = tomek.fit_resample(X, y)
print("Shape of X_resampled after Tomek links:", X_resampled.shape)
print("Shape of y_resampled after Tomek links:", y_resampled.shape)

# Handling Skewed Data

# Log Transformation
import numpy as np

df['Income_Log'] = np.log(df['Income'])

# Square Root Transformation
df['Income_Sqrt'] = np.sqrt(df['Income'])

# Box-Cox Transformation
from scipy import stats

df['Income_BoxCox'], _ = stats.boxcox(df['Income'])