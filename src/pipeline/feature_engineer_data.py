"""
This script feature engineers the preprocessed dementia dataset we will
be using in our ML pipeline to train our simple classification model.
"""
import pandas as pd

def feature_engineer_data(preprocessed_df):
    # Make a copy of the preprocessed DataFrame.
    engineered_df = preprocessed_df.copy()

    """
    Feature Engineering Decision #1: Compute Absolute Brain Volume (ABV)
    """
    engineered_df["ABV"] = engineered_df["eTIV"] * engineered_df["nWBV"]

    """
    Feature Engineering Decision #2: Compute Cognitive Impairment Index (CII)
    """
    engineered_df["CII"] = engineered_df["CDR"] - engineered_df["MMSE"]

    # Output the feature engineered DataFrame as a csv and return it.
    engineered_df.to_csv("data/processed/engineered_dementia_dataset.csv", index=False)
    return engineered_df