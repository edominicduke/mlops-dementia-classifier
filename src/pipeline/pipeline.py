from src.pipeline.ingest_data import *
from src.pipeline.preprocess_data import *
from src.pipeline.feature_engineer_data import *
from src.pipeline.train_model import *
from src.pipeline.evaluate_model import *

def run_pipeline():
    # Ingest data from the cloud and retrieve as a Pandas DataFrame.
    df = retrieve_data_as_df()

    # Preprocess the DataFrame.
    preprocessed_df = preprocess_cleaned_data(df)

    # Feature engineer the DataFrame.
    engineered_df = feature_engineer_data(preprocessed_df)

    # Train the model.
    train_model()

    # Evaluate the model and return relevant metrics.
    metrics = evaluate_model()
    return metrics

if __name__ == "__main__":
    run_pipeline()
