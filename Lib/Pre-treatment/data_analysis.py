import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
from scipy.stats import ttest_ind, chi2_contingency

data = pd.read_csv('dataset.csv')

# Preview the first few rows of the dataset
print(data.head())

# Check the shape of the dataset
print('Shape of the dataset:', data.shape)

# Check the data types of the columns
print('Data types of the columns:\n', data.dtypes)

# Check the summary statistics of the dataset
print('Summary statistics of the dataset:\n', data.describe())

# Check the missing values in the dataset
print('Missing values in the dataset:\n', data.isnull().sum())

# View the unique values of a categorical column in the dataset
print(data["categorical_column"].unique())

# Visualize the distribution of a numeric column
sns.displot(data['column_name'], kde=False)
plt.title('Distribution of Column Name')
plt.xlabel('Column Name')
plt.ylabel('Count')
plt.show()

# Visualize the relationship between two numeric columns
sns.scatterplot(data=data, x='column1', y='column2')
plt.title('Relationship between Column 1 and Column 2')
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.show()

# Visualize the relationship between a numeric column and a categorical column
sns.boxplot(data=data, x='categorical_column', y='numeric_column')
plt.title('Relationship between Categorical Column and Numeric Column')
plt.xlabel('Categorical Column')
plt.ylabel('Numeric Column')
plt.show()

# Visualize the correlation matrix of the numeric columns
sns.heatmap(data.select_dtypes(include='number').corr(), cmap='coolwarm', annot=True)
plt.title('Correlation Matrix of Numeric Columns')
plt.show()

# Visualize the distribution of a categorical column
sns.countplot(data=data, x='categorical_column')
plt.title('Distribution of Categorical Column')
plt.xlabel('Categorical Column')
plt.ylabel('Count')
plt.show()

# Visualize the relationship between two categorical columns
sns.catplot(data=data, x='categorical_column1', y='categorical_column2', kind='violin')
plt.title('Relationship between Categorical Column 1 and Categorical Column 2')
plt.xlabel('Categorical Column 1')
plt.ylabel('Categorical Column 2')
plt.show()


# Check for outliers in numeric columns using box plots
sns.boxplot(data=data.select_dtypes(include=['int64', 'float64']))
plt.title('Box Plot of Numeric Columns')
plt.show()

# Check the distribution of numeric columns using histograms
data.select_dtypes(include=['int64', 'float64']).hist(bins=20)
plt.suptitle('Histogram of Numeric Columns')
plt.show()

# Check the relationship between two categorical columns using a stacked bar plot
pd.crosstab(data['categorical_column1'], data['categorical_column2']).plot(kind='bar', stacked=True)
plt.title('Relationship between Categorical Column 1 and Categorical Column 2')
plt.xlabel('Categorical Column 1')
plt.ylabel('Count')
plt.show()

# Perform a t-test to compare the means of two numeric columns
t_stat, p_value = ttest_ind(data['numeric_column1'], data['numeric_column2'], equal_var=False)
print('T-statistic:', t_stat)
print('P-value:', p_value)

# Perform a chi-squared test to test for association between two categorical columns
cont_table = pd.crosstab(data['categorical_column1'], data['categorical_column2'])
chi2_stat, p_value, dof, expected = chi2_contingency(cont_table)
print('Chi-squared statistic:', chi2_stat)
print('P-value:', p_value)
print('Degrees of freedom:', dof)

# Analyze feature correlation
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Analyze feature distribution
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    sns.histplot(data[col], kde=True)
    plt.xlabel(col)
    plt.show()

# Analyze feature outliers
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    sns.boxplot(data[col])
    plt.xlabel(col)
    plt.show()

# Analyze class imbalance
sns.countplot(data['target_column'])
plt.xlabel('Target')
plt.show()