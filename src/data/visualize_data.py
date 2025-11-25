# No AI tools were used to generate any code in this script. 

"""
This script generates visualizations for the dementia dataset we will
be using in our ML pipeline to train our simple classification model.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download the dataset as a pandas DataFrame.
df = pd.read_csv("data/raw/dementia_dataset.csv")

# Generate pairplots to uncover interesting patterns or insights.

# Pairplot #1: Differentiate by Group (Demented, Nondemented, Converted) - You can see a saved version of it as fig2 in the docs/data_plots folder for reference.
sns.pairplot(data=df, hue="Group")
plt.show()

# Pairplot #2: Differentiate by Gender (Male, Female) - You can see a saved version of it as fig3 in the docs/data_plots folder for reference.
sns.pairplot(data=df, hue="M/F")
plt.show()

# Pairplot #3: Differentiate by Visit Number - You can see a saved version of it as fig4 in the docs/data_plots folder for reference.
sns.pairplot(data=df, hue="Visit")
plt.show()

# Create an array with the column labels corresponding to numerical columns (found from printed output in assess_data_structure.py)
numerical_vars = ["Visit", "MR Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

# Generate histograms for each of the numerical variables - You can see a saved versions of them as fig5-fig14 respectively in the docs/data_plots folder for reference.
for var in numerical_vars:
    fig, ax = plt.subplots()
    ax.hist(df[var].dropna())
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    plt.show()

# Generate box plots for each of the numerical variables - You can see a saved versions of them as fig15-fig24 respectively in the docs/data_plots folder for reference.
for var in numerical_vars:
    fig, ax = plt.subplots()
    ax.boxplot(df[var].dropna())
    ax.set_xlabel(var)
    ax.set_ylabel("Value")
    plt.show()