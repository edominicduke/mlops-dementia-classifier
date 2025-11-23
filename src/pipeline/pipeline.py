from ingest_data import *
from preprocess_data import *
from feature_engineer_data import *

# Ingest data from the cloud and retrieve as a Pandas DataFrame.
df = retrieve_data_as_df()

# Preprocess the DataFrame.
preprocessed_df = preprocess_cleaned_data(df)

# Feature engineer the DataFrame.
engineered_df = feature_engineer_data(preprocessed_df)