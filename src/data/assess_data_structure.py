"""
This script assesses the data structure of the dementia dataset we will
be using in our ML pipeline to train our simple classification model.
"""
import pandas as pd

# Download the dataset as a pandas DataFrame.
df = pd.read_csv("data/raw/dementia_dataset.csv")

# Report the dimensions of the dataset.
dimensions = df.shape
num_rows = dimensions[0]
num_cols = dimensions[1]
print("Dimensions of Dataset:")
print("Number of rows in dataset:", num_rows)
print("Number of columns in dataset:", num_cols)
print("The dimensions of this dataset are " + str(num_rows) + " x " + str(num_cols))
print("")

# Report the data types of the variables in the dataset.
df_data_types = df.dtypes
print("Data Types in the Dataset:")
print(df_data_types)
print("")

# Report any missing values in the DataFrame.
print("Missing Value Locations in DataFrame:")
print(df.isna())
print("")

# Report counts of missing values per column.
print("Number of Missing Values per Column:")
for col in df.columns:
    print(col + ": " + str(df[col].isna().sum()))