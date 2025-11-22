"""
This script explores variable relationships for the dementia dataset we will
be using in our ML pipeline to train our simple classification model.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download the dataset as a pandas DataFrame.
df = pd.read_csv("data/raw/dementia_dataset.csv")
print(df.head(10))
print("")

# Determine correlations between pairs of numerical variables.

# Create an array with the column labels corresponding to numerical columns (found from printed output in assess_data_structure.py)
numerical_vars = ["Visit", "MR Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

# Create a correlation matrix for all the numerical columns.
corr_matrix = df[numerical_vars].corr()

# For each numerical variable, print the other numerical variables it has meaningful correlation with.
for var in numerical_vars:
    print(var + ": ")
    corr_vals = corr_matrix[var]
    meaningful_corr_vals = corr_vals[(abs(corr_vals) >= 0.3) & (corr_vals.index != var)]
    print(meaningful_corr_vals.head(10))
    print("")

# Generate a correlation heatmap from the correlation matrix. You can see a saved version of it as fig1 in the docs/data_plots folder for reference.
sns.heatmap(corr_matrix)
plt.show()