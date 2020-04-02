'''
NASA Turbofan Failure Prediction - Project Pipeline

This is the main script of the project.

This script trains and evaluates a series of ML models to predict failure on turbofan engines using data supplied by NASA.  

'''

###################################
# Module Importations (A - Z Format)
###################################
import pandas as pd

import ntfp_dataset_import

###################################
# Main Datapipeline Script
###################################

if __name__ == '__main__':

    # Import & visualise data
    raw_data_df = ntfp_dataset_import.import_dataset()


    # Perform Initial Evaluation of Dataset


    # [Dataset] Noise Removal


    # [Dataset] Feature Engineering


    # [Model] Hyperparameter Optimisation


    # Model Evaluation