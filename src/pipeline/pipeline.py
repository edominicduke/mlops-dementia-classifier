from ingest_data import *
from preprocess_data import *

df = retrieve_data_as_df()
preprocessed_df = preprocess_cleaned_data(df)
print(preprocessed_df.head(10))