'''
NASA Turbofan Failure Prediction - Project Pipeline

This is the main script of the project.

This script trains and evaluates a series of ML models to predict failure on turbofan engines using data supplied by NASA.  

'''

###################################
# Module Importations (A - Z Format)
###################################
import pandas as pd

import src.ntfp_dataset_baseline as ntfp_dataset_baseline
import src.ntfp_dataset_import as ntfp_dataset_import
import src.ntfp_dataset_preprocessing as dataset_preprocessing

###################################
# Main Datapipeline Script
###################################

if __name__ == '__main__':

    # Import & visualise data
    raw_data_df = ntfp_dataset_import.import_dataset()

    ntfp_dataset_import.peek_dataset(raw_data_df)

    # Perform Initial Evaluation of Dataset
    dataset_preprocessing.dataset_remove_columns(None, None)

    ntfp_dataset_baseline.create_baseline_model(raw_data_df)

    ntfp_dataset_baseline.evaluate_baseline_model(model = None)

    # [Dataset] Noise Removal


    # [Dataset] Feature Engineering


    # [Model] Hyperparameter Optimisation


    # Model Evaluation