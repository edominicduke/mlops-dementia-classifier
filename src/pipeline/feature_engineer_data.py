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

    Rationale:
    1) ABV represents actual brain tissue volume 
    2) Dementia tends to be associated more strongly with ABV compared to just nWBV
    """
    engineered_df["ABV"] = engineered_df["eTIV"] * engineered_df["nWBV"]

    """
    Feature Engineering Decision #2: Compute Cognitive Impairment Index (CII)

    Rationale:
    1) With the way CII is mathematically defined, CII can easily differentiate patients 
    with normal cognition (CII would be more negative due to lower CDR and higher MMSE) 
    from patients with strong impairment (CII would be less negative due to higher CDR and lower MMSE)
    """
    engineered_df["CII"] = engineered_df["CDR"] - engineered_df["MMSE"]

    """
    Feature Engineering Decision #3: Compute Rate of Change of CDR with respect to MR Delay (CDR_RATE)

    Rationale:
    1) By normalizing CDR by MR Delay, CDR_RATE as a metric can quantify the speed of
    dementia progression and subsequently the severity of dementia progression (a higher 
    rate of change would imply a higher severity of dementia progression)
    """
    engineered_df["CDR_RATE"] = (engineered_df["CDR"] / (engineered_df["MR Delay"] + 1e-6))
    engineered_df["CDR_RATE"] = engineered_df["CDR_RATE"].where(engineered_df["MR Delay"] > 0, 0) # Set CDR_RATE to 0 wherever MR Delay is 0 to prevent overly large values of CDR_RATE unnecessarily existing

    # Output the feature engineered DataFrame as a csv and return it.
    engineered_df.to_csv("data/processed/engineered_dementia_dataset.csv", index=False)
    return engineered_df