'''
NASA Turbofan Failure Prediction - Project Pipeline

This is the main script of the project.

This script trains and evaluates a series of ML models to predict failure on turbofan engines using data supplied by NASA.  

'''

###################################
# Module Importations (A - Z Format)
###################################
import pandas as pd

import model.ntfp_model_multilayerperceptron as mlp_nn
import model.ntfp_model_randomforest as random_forest
import src.ntfp_dataset_baseline as dataset_baseline
import src.ntfp_dataset_exploratory as dataset_eda
import src.ntfp_dataset_import as dataset_import
import src.ntfp_dataset_preprocessing as dataset_preprocessing

###################################
# Main Datapipeline Script
###################################

if __name__ == '__main__':

    '''
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

    # Add Remaining Useful Life (RUL) to dataset.
    rul_dataset = dataset_preprocessing.add_calculated_rul(processed_df)

    print(rul_dataset.head())

    #dataset_eda.plot_time_history_all_engines(rul_dataset)

    # Remove data columns with no apparent trend.
    data_columns_no_trend = ['Set-1', 'Set-2']
    rul_dataset = dataset_preprocessing.dataset_remove_columns(rul_dataset, data_columns_no_trend)
    
    print(rul_dataset.head())

    rul_dataset.to_pickle(r'C:\Developer\PMetcalf\nasa_turbofan_failure_prediction\data\normalised_data.pkl')
    '''
    
    rul_dataset = pd.read_pickle(r'C:\Developer\PMetcalf\nasa_turbofan_failure_prediction\data\normalised_data.pkl')

    print(rul_dataset.head())

    # Standardise remaining data columns.
    normalised_data = dataset_preprocessing.standardise_columns(rul_dataset)

    # Calculate slope for each data value via linear regression.
    slopes_df, slopes_array = dataset_preprocessing.calculate_slopes_all_engines(rul_dataset, normalised_data)
    print(slopes_df.describe())

    # Order slopes by value.
    slope_order = dataset_preprocessing.return_data_ordered_abs_value(slopes_array, rul_dataset)

    # Drop all data columns except [5] most influential.
    data_columns_not_influential = ['Cycles', 'Sn_21', 'Sn_20', 'Sn_17', 'Sn_02', 'Sn_03', 'Sn_09']
    rul_dataset = dataset_preprocessing.dataset_remove_columns(rul_dataset, data_columns_not_influential)

    print(rul_dataset)

    # Apply normalisation to dataset.

    # Create baseline ML model tracking against RUL.
    baseline_model = dataset_baseline.create_baseline_model(rul_dataset, 'RUL')

    # Train / evaluate random forest regressor.
    random_forest_model = random_forest.train_random_forest_model(rul_dataset, 'RUL')

    # Train / evaluate NN.
    mlp_NN_model = mlp_nn.train_multi_layer_NN_model(rul_dataset, 'RUL')

    # Apply Principle Component Analysis to dataset.

    # Undertake PCA on original dataset.

    # Create new dataset with RUL and PCA components.

    # Retrain and evaluate models.

