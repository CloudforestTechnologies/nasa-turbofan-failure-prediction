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
    dataset_import.visualise_sensor_correlation_all_engine(raw_data_df)

    # Reduce / Eliminate highly-correlated sensors.
    correlation_threshold = 0.9
    high_corrs = dataset_preprocessing.find_correlated_data(raw_data_df, correlation_threshold)

    columns_to_be_removed = ['Sn_05', 'Sn_10', 'Sn_14', 'Sn_16']

    processed_df = dataset_preprocessing.dataset_remove_columns(raw_data_df, columns_to_be_removed)

    print(processed_df.info)

    # [Sensor Principle Component Analysis].

    # Develop health indicator.

    # Create baseline ML model for health indicator.
    dataset_baseline.create_baseline_model(raw_data_df)

    dataset_baseline.evaluate_baseline_model(model = None)

    # [Model] Hyperparameter Optimisation.


    # Model Evaluation.