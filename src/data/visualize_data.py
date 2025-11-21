"""
This script generates visualizations for the dementia dataset we will
be using in our ML pipeline to train our simple classification model.
"""
import pandas as pd
import seaborn as sns

# Download the dataset as a pandas DataFrame.
df = pd.read_csv("../../data/raw/dementia_dataset.csv")
print(df.head(10))

# Generate pairplots to uncover interesting patterns or insights.

# Pairplot #1: Differentiate by Group (Demented, Nondemented, Converted)
sns.pairplot(data=df, hue="Group")

# Pairplot #2: Differentiate by Gender (Male, Female)
sns.pairplot(data=df, hue="M/F")

# Pairplot #3: Differentiate by Visit Number
sns.pairplot(data=df, hue="Visit")

# Create an array with the column labels corresponding to numerical columns (found from printed output in assess_data_structure.py)
numerical_vars = ["Visit", "MR Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

# Generate histograms for each of the numerical variables.
import matplotlib.pyplot as plt
for var in numerical_vars:
    fig, ax = plt.subplots()
    ax.hist(df[var].dropna())
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    plt.show()

# Generate box plots for each of the numerical variables.
for var in numerical_vars:
    fig, ax = plt.subplots()
    ax.boxplot(df[var].dropna())
    ax.set_xlabel(var)
    ax.set_ylabel("Value")
    plt.show()