'''
NASA Turbofan Failure Prediction - Project Pipeline

This is the main script of the project.

This script trains and evaluates a series of ML models to predict failure on turbofan engines using data supplied by NASA.  

'''

###################################
# Module Importations (A - Z Format)
###################################
import pandas as pd

import src.ntfp_dataset_baseline as dataset_baseline
import src.ntfp_dataset_exploratory as dataset_eda
import src.ntfp_dataset_import as dataset_import
import src.ntfp_dataset_preprocessing as dataset_preprocessing

###################################
# Main Datapipeline Script
###################################

if __name__ == '__main__':

    # Import & peek data.
    raw_data_df = dataset_import.import_dataset()

    dataset_import.peek_dataset(raw_data_df)
    
    # Visualise engine data, for correlations.
    dataset_eda.visualise_sensor_correlation_all_engine(raw_data_df)

    # Reduce / Eliminate highly-correlated sensors.
    correlation_threshold = 0.9
    
    correlated_data = dataset_preprocessing.find_correlated_data(raw_data_df, correlation_threshold)
    columns_to_be_removed = dataset_preprocessing.list_correlated_data(correlated_data)
    processed_df = dataset_preprocessing.dataset_remove_columns(raw_data_df, columns_to_be_removed)

    print(processed_df.info)

    # Remove data that does not change with time.
    time_independent_columns = dataset_preprocessing.find_time_independent_columns(processed_df)
    processed_df = dataset_preprocessing.dataset_remove_columns(processed_df, time_independent_columns)

    print(processed_df.info)

    # [Sensor Principle Component Analysis].

    # Develop health indicator.

    # Create baseline ML model for health indicator.
    dataset_baseline.create_baseline_model(raw_data_df)

    dataset_baseline.evaluate_baseline_model(model = None)

    # [Model] Hyperparameter Optimisation.


    # Model Evaluation.